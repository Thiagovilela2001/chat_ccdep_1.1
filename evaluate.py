"""
Script de avaliação do pipeline RAG usando RAGAS.

Fases:
    1. Geração: executa o pipeline RAG, coleta respostas, contextos, diagnósticos e validação numérica.
    2. Julgamento: computa métricas RAGAS (faithfulness, context_precision, context_recall).

Uso:
    python evaluate.py --split dev                         # geração + julgamento
    python evaluate.py --split dev --generation-only       # apenas geração e validação numérica
    python evaluate.py --judge-only caminho/geracao.json   # apenas julgamento sobre artefato salvo
    python evaluate.py --ids dev_economy_01,dev_demography_03 # focado por IDs
"""
import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory
from rag_core import token_usage
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


def _sanitize_for_json(obj):
    """Substitui float('nan') e float('inf') por None para conformidade com JSON RFC 8259."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    sanitized = _sanitize_for_json(data)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _sanitize_url(url: str) -> str:
    if not url:
        return ""
    parsed_base = url.split("?")[0].rstrip("/")
    sanitized = re.sub(r"://[^@]+@", "://", parsed_base)
    return sanitized


def _sanitize_error_message(msg: str) -> str:
    """Oculta tokens ou credenciais de URLs e mensagens de erro."""
    return re.sub(
        r"(key|token|bearer|secret)[=:\s]+[A-Za-z0-9_\-\.]{8,}",
        r"\1=***REDACTED***",
        msg,
        flags=re.IGNORECASE,
    )


def _derive_judge_provider(base_url: str) -> str:
    env_provider = os.getenv("RAGAS_JUDGE_PROVIDER")
    if env_provider:
        return env_provider
    lowered = (base_url or "").lower()
    if "deepseek.com" in lowered:
        return "deepseek"
    if "maritaca.ai" in lowered:
        return "maritaca"
    if "openrouter.ai" in lowered:
        return "openrouter"
    if "openai.com" in lowered:
        return "openai"
    return "custom"


def _get_git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            check=True, timeout=5,
        )
        is_dirty = bool(status_proc.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        is_dirty = False

    diff_hash = None
    if is_dirty:
        try:
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD"], capture_output=True,
                check=True, timeout=10,
            )
            diff_hash = hashlib.sha256(diff_proc.stdout).hexdigest()
        except (OSError, subprocess.SubprocessError):
            diff_hash = "unknown"

    return {
        "git_commit": commit,
        "git_dirty": is_dirty,
        "git_diff_sha256": diff_hash,
    }


def _run_metadata(args) -> dict:
    git_info = _get_git_info()
    packages = {}
    for name in ("ragas", "datasets", "llama-index", "llama-index-core", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    base_url = os.getenv("RAGAS_JUDGE_BASE_URL", RAGAS_JUDGE_BASE_URL)
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL)
    judge_provider = _derive_judge_provider(base_url)

    return {
        "seed": getattr(args, "seed", 42),
        "rag": RAG_NAME,
        "use_graph": getattr(args, "use_graph", False),
        "limit": getattr(args, "limit", None),
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "git_diff_sha256": git_info["git_diff_sha256"],
        "python": platform.python_version(),
        "packages": packages,
        "dataset_sha256": {
            split: _dataset_hash(path) for split, path in DATASET_PATHS.items()
            if os.path.exists(path)
        },
        "models": {
            "main": os.getenv("RAG_LLM_MODEL"),
            "interp": os.getenv("RAG_INTERP_MODEL"),
            "extract": os.getenv("RAG_EXTRACT_MODEL"),
        },
        "reasoning_effort": {
            "main": os.getenv("RAG_LLM_REASONING_EFFORT"),
            "interp": os.getenv("RAG_INTERP_REASONING_EFFORT"),
            "extract": os.getenv("RAG_EXTRACT_REASONING_EFFORT"),
        },
        "query_fusion_queries": os.getenv("RAG_QUERY_FUSION_QUERIES"),
        "strict_inference": os.getenv("RAG_STRICT_INFERENCE"),
        "direction_check": os.getenv("RAG_DIRECTION_CHECK"),
        "rerank_enabled": os.getenv("RAG_LLM_RERANK"),
        "ragas_judge_model": judge_model,
        "ragas_judge_provider": judge_provider,
        "ragas_judge_base_url": _sanitize_url(base_url),
    }


def _ragas_judge_llm():
    model, client = judge_client()
    return model, llm_factory(
        model, client=client, max_tokens=8192, temperature=0
    )


def judge_client() -> tuple[str, OpenAIClient]:
    """Modelo e cliente OpenAI-compatível do judge, resolvidos pelo ambiente."""
    model = os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL)
    base_url = os.getenv("RAGAS_JUDGE_BASE_URL", RAGAS_JUDGE_BASE_URL)
    provider = _derive_judge_provider(base_url)

    api_key = os.getenv("RAGAS_JUDGE_API_KEY")
    if not api_key:
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("RAG_LLM_API_KEY")
        elif provider == "maritaca":
            api_key = os.getenv("MARITACA_API_KEY")
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            f"Defina RAGAS_JUDGE_API_KEY ou a chave para {provider} ({provider.upper()}_API_KEY) para executar o judge RAGAS."
        )
    return model, OpenAIClient(api_key=api_key, base_url=base_url)


def load_dataset(path: str, ids: set[str] | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = [d for d in data if d.get("question", "").strip()]
    if ids:
        valid = [d for d in valid if d.get("id") in ids]
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


# ── Fase 1: Geração ──────────────────────────────────────────────────────────

def generate_split(
    split: str,
    dataset: list[dict],
    engine,
    interp_llm,
    interpret_query,
) -> tuple[list[dict], list[dict]]:
    print(f"\n  Rodando geração para {len(dataset)} perguntas ({split})...")
    records = []
    diagnostics = []
    for i, item in enumerate(dataset, 1):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "").strip()
        q_type       = item.get("type", "?")

        print(f"  [{i}/{len(dataset)}] ({q_type}) {question[:70]}...")
        started = time.perf_counter()
        usage_before = token_usage.snapshot()
        try:
            answer, source_nodes, interp = run_question(question, engine, interp_llm, interpret_query)
            contexts = [n.get_content() for n in source_nodes]
            number_checks = validate_numbers(answer, source_nodes, question=question)
            applicable_checks = [c for c in number_checks if getattr(c, "applicable", True)]
            num_prec = (
                sum(check.verified for check in applicable_checks) / len(applicable_checks)
                if applicable_checks else None
            )
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
                "numeric_precision": num_prec,
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
        diagnostics[-1]["token_usage"] = token_usage.diff(token_usage.snapshot(), usage_before)

    return records, diagnostics


# ── Fase 2: Julgamento RAGAS ──────────────────────────────────────────────────

def judge_records(
    records: list[dict],
    diagnostics: list[dict],
    judge_model: str,
    judge_llm,
) -> tuple[dict, list[dict], bool, Optional[str]]:
    print(f"\n  Computando métricas RAGAS (judge: {judge_model})...")
    metrics = [
        Faithfulness(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    ds_records = []
    for r in records:
        ds_records.append({
            "user_input": r.get("user_input", ""),
            "response": r.get("response", ""),
            "retrieved_contexts": r.get("retrieved_contexts", []),
            "reference": r.get("reference", ""),
        })
    ds = Dataset.from_list(ds_records)

    try:
        scores = evaluate(ds, metrics=metrics)
        df = scores.to_pandas()

        has_nans = False
        nan_details = []
        for metric_name in ("faithfulness", "context_precision", "context_recall"):
            if metric_name in df.columns:
                nans_count = df[metric_name].isna().sum()
                if nans_count > 0:
                    has_nans = True
                    nan_details.append(f"{metric_name}: {nans_count}/{len(df)} NaN")

        scores_dict = df.mean(numeric_only=True).to_dict()
        details_list = df.to_dict(orient="records")

        combined_details = []
        for det, diag in zip(details_list, diagnostics):
            item = dict(det)
            item.update(diag)
            for m_key in ("faithfulness", "context_precision", "context_recall"):
                if m_key in item and isinstance(item[m_key], float) and math.isnan(item[m_key]):
                    item[m_key] = None
            combined_details.append(item)

        for m_key in ("faithfulness", "context_precision", "context_recall"):
            if m_key in scores_dict and isinstance(scores_dict[m_key], float) and math.isnan(scores_dict[m_key]):
                scores_dict[m_key] = None

        if has_nans:
            err_msg = f"Judge RAGAS produziu valores NaN ({', '.join(nan_details)})"
            return scores_dict, combined_details, False, err_msg

        return scores_dict, combined_details, True, None

    except Exception as exc:
        err_msg = _sanitize_error_message(f"{type(exc).__name__}: {exc}")
        scores_dict = {
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
        }
        combined_details = []
        for rec, diag in zip(records, diagnostics):
            item = dict(rec)
            item.update(diag)
            item.update({"faithfulness": None, "context_precision": None, "context_recall": None})
            combined_details.append(item)
        return scores_dict, combined_details, False, err_msg


# ── Execução em modo --judge-only ─────────────────────────────────────────────

def run_judge_only(
    artifact_path: str,
    output_path: Optional[str] = None,
    ids: Optional[set[str]] = None,
) -> tuple[dict, bool]:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artefato de geração não encontrado: {artifact_path}")

    with open(artifact_path, encoding="utf-8") as f:
        data = json.load(f)

    all_details = data.get("details", [])
    if ids:
        target_details = [d for d in all_details if d.get("id") in ids]
        if not target_details:
            raise ValueError(f"Nenhum registro encontrado correspondente aos IDs: {ids}")
    else:
        target_details = all_details

    records = []
    diagnostics = []
    for d in target_details:
        records.append({
            "user_input": d.get("user_input", ""),
            "response": d.get("response", ""),
            "retrieved_contexts": d.get("retrieved_contexts", []),
            "reference": d.get("reference", ""),
        })
        diag = {
            k: v for k, v in d.items()
            if k not in (
                "user_input", "response", "retrieved_contexts", "reference",
                "faithfulness", "context_precision", "context_recall"
            )
        }
        diagnostics.append(diag)

    judge_model, judge_llm = _ragas_judge_llm()
    scores_dict, judged_details, success, error_msg = judge_records(
        records, diagnostics, judge_model, judge_llm
    )

    if ids and not output_path:
        judged_by_id = {d.get("id"): d for d in judged_details}
        final_details = [judged_by_id.get(d.get("id"), d) for d in all_details]
    else:
        final_details = judged_details

    existing_summary = data.get("summary", {})
    if "latency_p50_ms" in existing_summary:
        scores_dict["latency_p50_ms"] = existing_summary["latency_p50_ms"]
    if "latency_p95_ms" in existing_summary:
        scores_dict["latency_p95_ms"] = existing_summary["latency_p95_ms"]
    if "numeric_precision" in existing_summary:
        scores_dict["numeric_precision"] = existing_summary["numeric_precision"]

    split = data.get("split", "dev")
    run_meta = dict(data.get("run", {}))
    base_url = os.getenv("RAGAS_JUDGE_BASE_URL", RAGAS_JUDGE_BASE_URL)
    run_meta["ragas_judge_model"] = judge_model
    run_meta["ragas_judge_provider"] = _derive_judge_provider(base_url)
    run_meta["ragas_judge_base_url"] = _sanitize_url(base_url)

    out_file = output_path or artifact_path
    output_obj = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "status": "completed" if success else "failed",
        "run": run_meta,
        "summary": scores_dict,
        "details": final_details,
    }
    if not success and error_msg:
        output_obj["error"] = error_msg

    _atomic_write_json(out_file, output_obj)
    print_results(scores_dict)
    if not success:
        print(f"\n  ❌ FALHA NO JUDGE: {error_msg}")
    else:
        print(f"\n  ✅ Julgamento concluído com sucesso e salvo em: {out_file}")

    return scores_dict, success


# ── Split Adversarial ─────────────────────────────────────────────────────────

def run_adversarial_split(dataset: list[dict], engine, interp_llm, interpret_query):
    print(f"\n  Rodando {len(dataset)} perguntas adversariais...")
    details = []
    refusals = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset)}] {question[:70]}...")
        started = time.perf_counter()
        usage_before = token_usage.snapshot()
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
            "token_usage": token_usage.diff(token_usage.snapshot(), usage_before),
        })

    refusal_accuracy = refusals / len(dataset) if dataset else 0.0
    scores_dict = {"refusal_accuracy": refusal_accuracy}
    return scores_dict, details


def print_results(scores_dict: dict):
    print("\n" + "=" * 55)
    print(" RESULTADOS")
    print("=" * 55)
    for metric, value in scores_dict.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            print(f"  {metric:<30} N/A")
        elif metric.endswith("_ms"):
            print(f"  {metric:<30} {value:.1f} ms")
        else:
            bar = "█" * int(value * 20)
            print(f"  {metric:<30} {value:.3f}  {bar}")


def save_results(
    split: str,
    scores_dict: dict,
    details: list,
    output_path: str | None = None,
    success: bool = True,
):
    path = output_path or os.path.join(RESULTS_DIR, f"results_{split}.json")
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split":     split,
        "status":    "completed" if success else "failed",
        "run":       _RUN_METADATA,
        "summary":   scores_dict,
        "token_usage": token_usage.merge(*(d.get("token_usage") for d in details)),
        "details":   details,
    }
    _atomic_write_json(path, output)
    print(f"\n  Resultados salvos em: {path}\n")


def run_ragas_split(
    split: str,
    dataset: list[dict],
    engine,
    interp_llm,
    interpret_query,
    generation_only: bool = False,
    output_path: Optional[str] = None,
) -> tuple[dict, list[dict], bool]:
    records, diagnostics = generate_split(split, dataset, engine, interp_llm, interpret_query)

    latencies = [item["latency_ms"] for item in diagnostics]
    numeric = [item["numeric_precision"] for item in diagnostics if item["numeric_precision"] is not None]

    base_out = output_path or os.path.join(RESULTS_DIR, f"results_{split}.json")
    checkpoint_path = base_out + ".generation_checkpoint.json"
    checkpoint_summary = {
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "numeric_precision": sum(numeric) / len(numeric) if numeric else None,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }
    checkpoint_details = []
    for rec, diag in zip(records, diagnostics):
        item = dict(rec)
        item.update(diag)
        item.update({"faithfulness": None, "context_precision": None, "context_recall": None})
        checkpoint_details.append(item)

    _atomic_write_json(checkpoint_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "status": "generated",
        "run": _RUN_METADATA,
        "summary": checkpoint_summary,
        "token_usage": token_usage.merge(*(d.get("token_usage") for d in checkpoint_details)),
        "details": checkpoint_details,
    })
    print(f"  Checkpoint da geração gravado em: {checkpoint_path}")

    if generation_only:
        print("\n  [generation-only] Pulando métricas RAGAS; preservando respostas e diagnósticos.")
        return checkpoint_summary, checkpoint_details, True

    judge_model, judge_llm = _ragas_judge_llm()
    scores_dict, details, success, error_msg = judge_records(
        records, diagnostics, judge_model, judge_llm
    )
    scores_dict.update({
        "numeric_precision": sum(numeric) / len(numeric) if numeric else None,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    })
    if not success and error_msg:
        print(f"\n  ❌ {error_msg}")
    return scores_dict, details, success


def run_split(
    split: str,
    engine,
    interp_llm,
    interpret_query,
    limit: int | None = None,
    ids: set[str] | None = None,
    generation_only: bool = False,
    output_path: str | None = None,
) -> tuple[dict, bool]:
    print(f"\n{'=' * 55}")
    print(f" SPLIT: {split.upper()}")
    print(f"{'=' * 55}")

    print(f"\n1. Carregando dataset ({split})...")
    dataset = load_dataset(DATASET_PATHS[split], ids=ids)
    if limit is not None:
        dataset = dataset[:max(limit, 0)]

    if split == "adversarial":
        scores_dict, details = run_adversarial_split(dataset, engine, interp_llm, interpret_query)
        success = True
    else:
        scores_dict, details, success = run_ragas_split(
            split,
            dataset,
            engine,
            interp_llm,
            interpret_query,
            generation_only=generation_only,
            output_path=output_path,
        )

    print_results(scores_dict)
    save_results(split, scores_dict, details, output_path=output_path, success=success)
    return scores_dict, success


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
        "--ids",
        type=str,
        default=None,
        help="Filtra por IDs específicos separados por vírgula (ex.: dev_economy_01,dev_demography_03)",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        default=False,
        help="Executa apenas geração e validação numérica, sem chamar o judge RAGAS",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        default=False,
        help="Sinônimo de --generation-only",
    )
    parser.add_argument(
        "--judge-only",
        type=str,
        default=None,
        help="Caminho para arquivo JSON de geração para executar apenas o judge RAGAS",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho personalizado para salvar o arquivo JSON de resultado",
    )
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
    token_usage.install()
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    global _RUN_METADATA
    _RUN_METADATA = _run_metadata(args)

    target_ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None

    # Modo 1: Judge-Only (não inicializa pipeline RAG)
    if args.judge_only:
        print("Modo --judge-only ativado. Ignorando inicialização do pipeline RAG.")
        scores, success = run_judge_only(
            artifact_path=args.judge_only,
            output_path=args.output,
            ids=target_ids,
        )
        sys.exit(0 if success else 1)

    # Modo 2 & 3: Geração completa ou generation-only
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rag_dir  = os.path.join(root_dir, RAG_NAME)

    if not os.path.isdir(rag_dir):
        print(f"[ERRO] Pasta '{RAG_NAME}' não encontrada em {root_dir}")
        sys.exit(1)

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
    gen_only = bool(args.generation_only or args.skip_ragas)

    all_scores = {}
    all_success = True
    for split in splits:
        scores, success = run_split(
            split,
            engine,
            interp_llm,
            interpret_query,
            limit=args.limit,
            ids=target_ids,
            generation_only=gen_only,
            output_path=args.output,
        )
        all_scores[split] = scores
        if not success:
            all_success = False

    if args.split == "all":
        print("\n" + "=" * 55)
        print(" RESUMO GERAL")
        print("=" * 55)
        for split, scores in all_scores.items():
            print(f"\n  [{split.upper()}]")
            for metric, value in scores.items():
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    if metric.endswith("_ms"):
                        print(f"    {metric:<30} {value:.1f} ms")
                    else:
                        bar = "█" * int(value * 20)
                        print(f"    {metric:<30} {value:.3f}  {bar}")

    if not all_success:
        print("\n[ERRO] Avaliação concluiu com falhas no judge.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
