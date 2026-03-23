"""
Script de avaliação do pipeline RAG usando RAGAS.

Uso:
    python evaluate.py

Saída:
    - Tabela de métricas no console
    - evaluation_results.json com detalhes por pergunta

Métricas computadas:
    Sempre:        faithfulness, answer_relevancy
    Com ground_truth: context_precision, context_recall
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index, load_nodes_cache
from src.qa_chain import get_query_engine, answer_question
from src.query_router import classify_query
from src.calculation_engine import CalculationEngine
from llama_index.llms.openai import OpenAI

GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
RESULTS_PATH = "evaluation/results.json"


def load_golden_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = [d for d in data if d.get("question", "").strip()]
    print(f"  {len(valid)} perguntas carregadas ({sum(1 for d in valid if d.get('ground_truth', '').strip())} com ground_truth)")
    return valid


def build_rag_pipeline():
    """Inicializa o pipeline RAG (igual ao main.py, sem o loop interativo)."""
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
    query_engine, retriever, reranker = get_query_engine(index, nodes=bm25_nodes)

    llm = OpenAI(model="gpt-4.1", temperature=0.0)
    calc_engine = CalculationEngine(retriever=retriever, reranker=reranker, llm=llm)

    return query_engine, calc_engine, llm


def run_question(question: str, query_engine, calc_engine, llm) -> tuple[str, list[str]]:
    """Roda uma pergunta no pipeline e retorna (resposta, contextos)."""
    query_type = classify_query(question, llm)

    if query_type == "calculo":
        answer, source_nodes = calc_engine.answer(question)
        contexts = [n.get_content() for n in source_nodes]
    else:
        response = answer_question(query_engine, question)
        answer = response.response
        contexts = [n.get_content() for n in response.source_nodes]

    return answer, contexts


def run_evaluation():
    print("=" * 55)
    print(" AVALIAÇÃO RAG — RAGAS")
    print("=" * 55)

    print("\n1. Carregando golden dataset...")
    golden = load_golden_dataset(GOLDEN_DATASET_PATH)

    has_ground_truth = any(d.get("ground_truth", "").strip() for d in golden)
    if not has_ground_truth:
        print("\n  ⚠️  Nenhum ground_truth encontrado.")
        print("  Serão computadas apenas: faithfulness, answer_relevancy")
        print("  Preencha o campo 'ground_truth' em data/golden_dataset.json")
        print("  para ativar: context_precision, context_recall\n")

    print("\n2. Inicializando pipeline RAG...")
    query_engine, calc_engine, llm = build_rag_pipeline()

    print(f"\n3. Rodando {len(golden)} perguntas no pipeline...")
    records = []
    for i, item in enumerate(golden, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "").strip()
        query_type = item.get("type", "?")

        print(f"  [{i}/{len(golden)}] ({query_type}) {question[:70]}...")

        try:
            answer, contexts = run_question(question, query_engine, calc_engine, llm)
            record = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth or None,
            }
            records.append(record)
            print(f"         ✅ OK")
        except Exception as exc:
            print(f"         ❌ ERRO: {exc}")
            records.append({
                "question": question,
                "answer": f"[ERRO] {exc}",
                "contexts": [],
                "ground_truth": ground_truth or None,
            })

    print("\n4. Computando métricas RAGAS...")
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics += [context_precision, context_recall]

    # RAGAS espera ground_truth como string vazia ou preenchida
    for r in records:
        if r["ground_truth"] is None:
            r["ground_truth"] = ""

    dataset = Dataset.from_list(records)
    scores = evaluate(dataset, metrics=metrics)

    print("\n" + "=" * 55)
    print(" RESULTADOS")
    print("=" * 55)
    scores_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
    for metric, value in scores_dict.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:<25} {value:.3f}  {bar}")

    # Salva resultados detalhados
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": scores_dict,
        "details": scores.to_pandas().to_dict(orient="records"),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Resultados detalhados salvos em: {RESULTS_PATH}\n")


if __name__ == "__main__":
    run_evaluation()
