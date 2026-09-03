"""
Script de avaliação do pipeline RAG usando RAGAS.

Uso:
    python evaluate.py --split dev          # padrão
    python evaluate.py --split test
    python evaluate.py --split adversarial  # usa métrica de recusa, sem RAGAS
    python evaluate.py --split all          # roda dev + test + adversarial

Saída:
    - Tabela de métricas no console
    - evaluation/results_{split}.json com detalhes por pergunta
"""
import argparse
import asyncio
import importlib
import importlib.metadata
import os
import sys
import json
import math
import hashlib
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset
from ragas import evaluate
# ragas.evaluate() 0.4.3 ainda recebe as classes Metric legadas. As homônimas
# de metrics.collections usam SimpleBaseMetric e não são aceitas por evaluate().
# O projeto fixa ragas==0.4.3; estes imports evitam os avisos do alias obsoleto.
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory
from rag_core.answer_policy import sanitize_answer
from rag_core.numerical_validator import validate_numbers

RAG_NAME = "rag_principal"
RAGAS_JUDGE_MODEL = "sabia-4"
RAGAS_JUDGE_BASE_URL = "https://chat.maritaca.ai/api"

DATASET_PATHS = {
    "dev":         "data/golden_dataset_dev.json",
    "test":        "data/golden_dataset_test.json",
    "adversarial": "data/golden_dataset_adversarial.json",
}

RESULTS_DIR = "evaluation"

REFUSAL_KEYWORDS = [
    "não consta",
    "não está disponível",
    "não foi possível encontr",
    "não há informação",
    "não encontr",
    "não tenho informação",
    "não está nos documentos",
    "informação não consta",
    "não fornecem evidência suficiente",
    "não é possível determinar com segurança",
]

_RUN_METADATA: dict = {}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile), len(ordered) - 1)
    return ordered[index]


def _dataset_hash(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _run_metadata(args) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    packages = {}
    for name in ("ragas", "datasets", "llama-index", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "seed": args.seed,
        "rag": RAG_NAME,
        "use_graph": args.use_graph,
        "limit": args.limit,
        "git_commit": commit,
        "python": platform.python_version(),
        "packages": packages,
        "dataset_sha256": {
            split: _dataset_hash(path) for split, path in DATASET_PATHS.items()
            if os.path.exists(path)
        },
        "ragas_judge_model": os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL),
        "ragas_judge_provider": "maritaca",
        "ragas_judge_base_url": os.getenv(
            "RAGAS_JUDGE_BASE_URL", RAGAS_JUDGE_BASE_URL
        ),
    }


def _ragas_judge_llm():
    model = os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL)
    api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("MARITACA_API_KEY")
    base_url = os.getenv("RAGAS_JUDGE_BASE_URL", RAGAS_JUDGE_BASE_URL)
    if not api_key:
        raise EnvironmentError(
            "Defina MARITACA_API_KEY ou RAGAS_JUDGE_API_KEY para executar o judge RAGAS."
        )
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    return model, llm_factory(
        model, client=client, max_tokens=8192, temperature=0
    )


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = [d for d in data if d.get("question", "").strip()]
    gt_count = sum(1 for d in valid if d.get("ground_truth", "").strip())
    print(f"  {len(valid)} perguntas carregadas ({gt_count} com ground_truth)")
    return valid


def run_question(question: str, engine, interp_llm, interpret_query) -> tuple[str, list, dict]:
    interp = interpret_query(question, interp_llm)
    answer, source_nodes = asyncio.run(
        engine.answer(
            question=question,
            sources=interp["sources"],
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        )
    )
    answer = sanitize_answer(answer, question=question)
    return answer, source_nodes, interp


def is_refusal(response: str) -> bool:
    lowered = response.lower()
    return any(kw in lowered for kw in REFUSAL_KEYWORDS)


def run_ragas_split(split: str, dataset: list[dict], engine, interp_llm, interpret_query):
    print(f"\n  Rodando {len(dataset)} perguntas ({split})...")
    records = []
    diagnostics = []
    for i, item in enumerate(dataset, 1):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "").strip()
        q_type       = item.get("type", "?")

        print(f"  [{i}/{len(dataset)}] ({q_type}) {question[:70]}...")
        started = time.perf_counter()
        try:
            answer, source_nodes, interp = run_question(question, engine, interp_llm, interpret_query)
            contexts = [n.get_content() for n in source_nodes]
            number_checks = validate_numbers(answer, source_nodes)
            records.append({
                "user_input":        question,
                "response":          answer,
                "retrieved_contexts": contexts,
                "reference":         ground_truth or "",
            })
            diagnostics.append({
                "id": item.get("id"),
                "type": q_type,
                "domain": item.get("domain"),
                "expected_source_files": item.get("source_files", []),
                "expected_source_pages": item.get("source_pages", []),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "numeric_precision": (
                    sum(check.verified for check in number_checks) / len(number_checks)
                    if number_checks else None
                ),
            })
            print(f"         ✅ OK | fontes: {interp['sources']} | chunks: {len(contexts)}")
        except Exception as exc:
            print(f"         ❌ ERRO: {type(exc).__name__}: {exc}")
            records.append({
                "user_input":        question,
                "response":          f"[ERRO] {exc}",
                "retrieved_contexts": [],
                "reference":         ground_truth or "",
            })
            diagnostics.append({
                "id": item.get("id"),
                "type": q_type,
                "domain": item.get("domain"),
                "expected_source_files": item.get("source_files", []),
                "expected_source_pages": item.get("source_pages", []),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "numeric_precision": None,
            })

    ragas_model, ragas_llm = _ragas_judge_llm()
    print(f"\n  Computando métricas RAGAS (judge: {ragas_model})...")
    metrics = [Faithfulness(llm=ragas_llm), ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
    ds = Dataset.from_list(records)
    scores = evaluate(ds, metrics=metrics)

    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
    details = scores.to_pandas().to_dict(orient="records")
    for detail, diagnostic in zip(details, diagnostics):
        detail.update(diagnostic)

    latencies = [item["latency_ms"] for item in diagnostics]
    numeric = [item["numeric_precision"] for item in diagnostics if item["numeric_precision"] is not None]
    scores_dict.update({
        "numeric_precision": sum(numeric) / len(numeric) if numeric else math.nan,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    })

    return scores_dict, details


def run_adversarial_split(dataset: list[dict], engine, interp_llm, interpret_query):
    print(f"\n  Rodando {len(dataset)} perguntas adversariais...")
    details = []
    refusals = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset)}] {question[:70]}...")
        started = time.perf_counter()
        try:
            answer, _, _interp = run_question(question, engine, interp_llm, interpret_query)
            refused = is_refusal(answer)
            refusals += int(refused)
            mark = "✅" if refused else "❌ ALUCINAÇÃO"
            print(f"         {mark}")
        except Exception as exc:
            print(f"         ❌ ERRO: {exc}")
            answer = f"[ERRO] {exc}"
            refused = False

        details.append({
            "id":         item.get("id"),
            "type":       item.get("type"),
            "domain":     item.get("domain"),
            "user_input": question,
            "response":   answer,
            "refused":    refused,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        })

    refusal_accuracy = refusals / len(dataset) if dataset else 0.0
    scores_dict = {"refusal_accuracy": refusal_accuracy}
    return scores_dict, details


def print_results(scores_dict: dict):
    print("\n" + "=" * 55)
    print(" RESULTADOS")
    print("=" * 55)
    for metric, value in scores_dict.items():
        if isinstance(value, float) and math.isnan(value):
            print(f"  {metric:<30} N/A")
        elif metric.endswith("_ms"):
            print(f"  {metric:<30} {value:.1f} ms")
        else:
            bar = "█" * int(value * 20)
            print(f"  {metric:<30} {value:.3f}  {bar}")


def save_results(split: str, scores_dict: dict, details: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_{split}.json")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split":     split,
        "run":       _RUN_METADATA,
        "summary":   scores_dict,
        "details":   details,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Resultados salvos em: {path}\n")


def run_split(split: str, engine, interp_llm, interpret_query, limit: int | None = None):
    print(f"\n{'=' * 55}")
    print(f" SPLIT: {split.upper()}")
    print(f"{'=' * 55}")

    print(f"\n1. Carregando dataset ({split})...")
    dataset = load_dataset(DATASET_PATHS[split])
    if limit is not None:
        dataset = dataset[:max(limit, 0)]

    if split == "adversarial":
        scores_dict, details = run_adversarial_split(dataset, engine, interp_llm, interpret_query)
    else:
        scores_dict, details = run_ragas_split(split, dataset, engine, interp_llm, interpret_query)

    print_results(scores_dict)
    save_results(split, scores_dict, details)
    return scores_dict


def main():
    parser = argparse.ArgumentParser(description="Avaliação RAG com RAGAS")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "adversarial", "all"],
        default="dev",
        help="Split a avaliar (padrão: dev)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da avaliação (padrão: 42)")
    parser.add_argument("--limit", type=int, default=None, help="Limita exemplos por split")
    parser.add_argument(
        "--use-graph",
        action="store_true",
        default=False,
        help="Habilita o GraphRetriever como 4ª fonte (apenas rag_principal)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit deve ser maior ou igual a 1")

    load_dotenv()
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    global _RUN_METADATA
    _RUN_METADATA = _run_metadata(args)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rag_dir  = os.path.join(root_dir, RAG_NAME)

    if not os.path.isdir(rag_dir):
        print(f"[ERRO] Pasta '{RAG_NAME}' não encontrada em {root_dir}")
        sys.exit(1)

    # Importa initialize e interpret_query do rag_principal.
    sys.path.insert(0, rag_dir)
    startup_mod = importlib.import_module("src.startup")
    interp_mod  = importlib.import_module("src.query_interpreter")
    initialize      = startup_mod.initialize
    interpret_query = interp_mod.interpret_query

    print(f"Inicializando pipeline RAG ({RAG_NAME})...")
    data_dir = os.path.join(root_dir, "data")
    init_kwargs = {"data_dir": data_dir}
    if args.use_graph:
        init_kwargs["use_graph"] = True
    engine, interp_llm = initialize(rag_dir, **init_kwargs)

    splits = ["dev", "test", "adversarial"] if args.split == "all" else [args.split]

    all_scores = {}
    for split in splits:
        all_scores[split] = run_split(
            split, engine, interp_llm, interpret_query, limit=args.limit
        )

    if args.split == "all":
        print("\n" + "=" * 55)
        print(" RESUMO GERAL")
        print("=" * 55)
        for split, scores in all_scores.items():
            print(f"\n  [{split.upper()}]")
            for metric, value in scores.items():
                if isinstance(value, float) and not math.isnan(value):
                    if metric.endswith("_ms"):
                        print(f"    {metric:<30} {value:.1f} ms")
                    else:
                        bar = "█" * int(value * 20)
                        print(f"    {metric:<30} {value:.3f}  {bar}")


if __name__ == "__main__":
    main()
