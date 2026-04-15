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
import os
import sys
import json
import math
from datetime import datetime
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory

from src.startup import initialize
from src.query_interpreter import interpret_query

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
]


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = [d for d in data if d.get("question", "").strip()]
    gt_count = sum(1 for d in valid if d.get("ground_truth", "").strip())
    print(f"  {len(valid)} perguntas carregadas ({gt_count} com ground_truth)")
    return valid


def run_question(question: str, engine, interp_llm) -> tuple[str, list, dict]:
    interp = interpret_query(question, interp_llm)
    answer, source_nodes = asyncio.run(
        engine.answer(
            question=question,
            sources=interp["sources"],
            rewritten_query=interp["rewritten_query"],
            is_labor_market=interp.get("is_labor_market", False),
        )
    )
    return answer, source_nodes, interp


def is_refusal(response: str) -> bool:
    lowered = response.lower()
    return any(kw in lowered for kw in REFUSAL_KEYWORDS)


def run_ragas_split(split: str, dataset: list[dict], engine, interp_llm):
    print(f"\n  Rodando {len(dataset)} perguntas ({split})...")
    records = []
    for i, item in enumerate(dataset, 1):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "").strip()
        q_type       = item.get("type", "?")

        print(f"  [{i}/{len(dataset)}] ({q_type}) {question[:70]}...")
        try:
            answer, source_nodes, interp = run_question(question, engine, interp_llm)
            contexts = [n.get_content() for n in source_nodes]
            records.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth or "",
            })
            print(f"         ✅ OK | fontes: {interp['sources']} | chunks: {len(contexts)}")
        except Exception as exc:
            print(f"         ❌ ERRO: {exc}")
            records.append({
                "question":     question,
                "answer":       f"[ERRO] {exc}",
                "contexts":     [],
                "ground_truth": ground_truth or "",
            })

    ragas_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-5-chat-latest")
    print(f"\n  Computando métricas RAGAS (judge: {ragas_model})...")
    ragas_llm = llm_factory(ragas_model, client=OpenAIClient(), max_tokens=8192)
    metrics = [Faithfulness(llm=ragas_llm), ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
    ds = Dataset.from_list(records)
    scores = evaluate(ds, metrics=metrics)

    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
    details = scores.to_pandas().rename(columns={"question": "user_input"}).to_dict(orient="records")

    return scores_dict, details


def run_adversarial_split(dataset: list[dict], engine, interp_llm):
    print(f"\n  Rodando {len(dataset)} perguntas adversariais...")
    details = []
    refusals = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset)}] {question[:70]}...")
        try:
            answer, _, _interp = run_question(question, engine, interp_llm)
            refused = is_refusal(answer)
            refusals += int(refused)
            mark = "✅" if refused else "❌ ALUCINAÇÃO"
            print(f"         {mark}")
        except Exception as exc:
            print(f"         ❌ ERRO: {exc}")
            answer = f"[ERRO] {exc}"
            refused = False

        details.append({
            "user_input": question,
            "response":   answer,
            "refused":    refused,
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
        else:
            bar = "█" * int(value * 20)
            print(f"  {metric:<30} {value:.3f}  {bar}")


def save_results(split: str, scores_dict: dict, details: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_{split}.json")
    output = {
        "timestamp": datetime.now().isoformat(),
        "split":     split,
        "summary":   scores_dict,
        "details":   details,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Resultados salvos em: {path}\n")


def run_split(split: str, engine, interp_llm):
    print(f"\n{'=' * 55}")
    print(f" SPLIT: {split.upper()}")
    print(f"{'=' * 55}")

    print(f"\n1. Carregando dataset ({split})...")
    dataset = load_dataset(DATASET_PATHS[split])

    if split == "adversarial":
        scores_dict, details = run_adversarial_split(dataset, engine, interp_llm)
    else:
        scores_dict, details = run_ragas_split(split, dataset, engine, interp_llm)

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
    args = parser.parse_args()

    load_dotenv()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Inicializando pipeline RAG...")
    engine, interp_llm = initialize(base_dir)

    splits = ["dev", "test", "adversarial"] if args.split == "all" else [args.split]

    all_scores = {}
    for split in splits:
        all_scores[split] = run_split(split, engine, interp_llm)

    if args.split == "all":
        print("\n" + "=" * 55)
        print(" RESUMO GERAL")
        print("=" * 55)
        for split, scores in all_scores.items():
            print(f"\n  [{split.upper()}]")
            for metric, value in scores.items():
                if isinstance(value, float) and not math.isnan(value):
                    bar = "█" * int(value * 20)
                    print(f"    {metric:<30} {value:.3f}  {bar}")


if __name__ == "__main__":
    main()
