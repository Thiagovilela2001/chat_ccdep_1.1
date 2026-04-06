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
import os
import sys
import json
import math
from datetime import datetime
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index, load_nodes_cache
from src.qa_chain import get_query_engine, answer_question
from src.query_router import classify_query
from src.calculation_engine import CalculationEngine
from llama_index.core import Settings

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


def build_pipeline():
    """Inicializa o pipeline RAG."""
    load_dotenv()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")

    db = chromadb.PersistentClient(path=DB_PATH)
    col = db.get_or_create_collection("estatisticas")

    nodes = []
    if col.count() == 0:
        print("  Banco vazio — indexando documentos...")
        docs = load_documents(DATA_DIR)
        nodes = process_documents(docs)

    index = create_or_load_index(nodes, db_path=DB_PATH)
    bm25_nodes = load_nodes_cache()
    query_engine, retriever, reranker, rewrite_llm = get_query_engine(index, nodes=bm25_nodes)

    llm = Settings.llm  # instância já criada por setup_llm() dentro de get_query_engine()
    calc_engine = CalculationEngine(retriever=retriever, reranker=reranker, llm=llm)

    return query_engine, calc_engine, llm, rewrite_llm


def run_question(question: str, query_engine, calc_engine, llm, rewrite_llm=None) -> tuple[str, list[str]]:
    query_type = classify_query(question, llm)

    if query_type == "calculo":
        answer, source_nodes = calc_engine.answer(question)
        contexts = [n.get_content() for n in source_nodes]
    else:
        response = answer_question(query_engine, question, rewrite_llm=rewrite_llm)
        answer = response.response
        contexts = [n.get_content() for n in response.source_nodes]

    return answer, contexts


def is_refusal(response: str) -> bool:
    """Verifica se a resposta indica corretamente que a informação não está nos documentos."""
    lowered = response.lower()
    return any(kw in lowered for kw in REFUSAL_KEYWORDS)


def run_ragas_split(split: str, dataset: list[dict], query_engine, calc_engine, llm, rewrite_llm=None):
    """Roda RAGAS em um split (dev ou test). Retorna dict com métricas e detalhes."""
    print(f"\n  Rodando {len(dataset)} perguntas ({split})...")
    records = []
    for i, item in enumerate(dataset, 1):
        question    = item["question"]
        ground_truth = item.get("ground_truth", "").strip()
        q_type      = item.get("type", "?")

        print(f"  [{i}/{len(dataset)}] ({q_type}) {question[:70]}...")
        try:
            answer, contexts = run_question(question, query_engine, calc_engine, llm, rewrite_llm=rewrite_llm)
            records.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth or "",
            })
            print("         ✅ OK")
        except Exception as exc:
            print(f"         ❌ ERRO: {exc}")
            records.append({
                "question":     question,
                "answer":       f"[ERRO] {exc}",
                "contexts":     [],
                "ground_truth": ground_truth or "",
            })

    print("\n  Computando métricas RAGAS (judge: gpt-5-chat-latest)...")
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-5-chat-latest", temperature=0.0))
    metrics = [Faithfulness(llm=ragas_llm), ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
    ds = Dataset.from_list(records)
    scores = evaluate(ds, metrics=metrics)

    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
    details = scores.to_pandas().rename(columns={"question": "user_input"}).to_dict(orient="records")

    return scores_dict, details


def run_adversarial_split(dataset: list[dict], query_engine, calc_engine, llm, rewrite_llm=None):
    """Roda o split adversarial e computa refusal_accuracy."""
    print(f"\n  Rodando {len(dataset)} perguntas adversariais...")
    details = []
    refusals = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset)}] {question[:70]}...")
        try:
            answer, _ = run_question(question, query_engine, calc_engine, llm, rewrite_llm=rewrite_llm)
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


def run_split(split: str, query_engine, calc_engine, llm, rewrite_llm=None):
    print(f"\n{'=' * 55}")
    print(f" SPLIT: {split.upper()}")
    print(f"{'=' * 55}")

    print(f"\n1. Carregando dataset ({split})...")
    dataset = load_dataset(DATASET_PATHS[split])

    if split == "adversarial":
        scores_dict, details = run_adversarial_split(dataset, query_engine, calc_engine, llm, rewrite_llm=rewrite_llm)
    else:
        scores_dict, details = run_ragas_split(split, dataset, query_engine, calc_engine, llm, rewrite_llm=rewrite_llm)

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

    print("Inicializando pipeline RAG...")
    query_engine, calc_engine, llm, rewrite_llm = build_pipeline()

    splits = ["dev", "test", "adversarial"] if args.split == "all" else [args.split]

    all_scores = {}
    for split in splits:
        all_scores[split] = run_split(split, query_engine, calc_engine, llm, rewrite_llm=rewrite_llm)

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
