"""
Startup — inicialização compartilhada do sistema RAG.

Usado tanto pela API (FastAPI lifespan) quanto pelo CLI interativo.
Centraliza: detecção de mudanças, indexação, criação de LLMs e retrievers.
"""
import os
import json

import chromadb
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index, load_nodes_cache
from src.text_retriever import build_hybrid_retriever, TextRetriever
from src.tables_retriever import TablesRetriever
from src.timeseries_retriever import TimeSeriesRetriever
from src.analysis_engine import AnalysisEngine
from src.labor_market_skill import LaborMarketSkill

MANIFEST_FILE = "chroma_db/indexed_manifest.json"


# ── Detecção de mudanças ──────────────────────────────────────────────────────

def _get_data_snapshot(data_dir: str) -> dict:
    snapshot = {}
    if not os.path.isdir(data_dir):
        return snapshot
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            snapshot[fname] = os.path.getmtime(fpath)
    return snapshot


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(snapshot: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def _detect_changes(snapshot: dict, manifest: dict) -> list[str]:
    return [
        fname for fname, mtime in snapshot.items()
        if fname not in manifest or manifest[fname] != mtime
    ]


# ── Inicialização principal ───────────────────────────────────────────────────

def initialize(base_dir: str) -> tuple[AnalysisEngine, object]:
    """
    Inicializa o sistema completo e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do projeto (onde ficam /data e /chroma_db).

    Retorna
    -------
    engine : AnalysisEngine
        Engine pronta para receber queries.
    interp_llm : OpenAI
        LLM leve usado pelo Query Interpreter.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY não encontrada. "
            "Crie um arquivo .env com: OPENAI_API_KEY=sua_chave"
        )

    data_dir = os.path.join(base_dir, "data")
    db_path = os.path.join(base_dir, "chroma_db")

    print("=" * 52)
    print(" INICIALIZANDO RAG ESTATÍSTICO SP")
    print("=" * 52)

    # 1. Detecção de mudanças nos documentos
    snapshot = _get_data_snapshot(data_dir)
    manifest = _load_manifest()
    changed = _detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    # 2. Ingestão e processamento (somente se necessário)
    nodes = []
    if changed:
        print(f"\n[1] Documentos novos/modificados: {changed}")
        print("    Reindexando banco de dados...")
        db.delete_collection("estatisticas")
        col = db.get_or_create_collection("estatisticas")
        docs = load_documents(data_dir)
        if docs:
            nodes = process_documents(docs)
    elif not already_indexed:
        print("\n[1] Banco vazio. Carregando documentos...")
        docs = load_documents(data_dir)
        if docs:
            nodes = process_documents(docs)
        else:
            print("    Nenhum documento encontrado em /data.")
    else:
        print(f"\n[1] Banco atualizado ({col.count()} vetores). Sem mudanças detectadas.")

    # 3. Indexação vetorial
    print("\n[2] Indexando vetores no ChromaDB...")
    index = create_or_load_index(nodes, db_path=db_path)

    if changed or not already_indexed:
        _save_manifest(snapshot)

    # 4. LLMs
    print("\n[3] Carregando modelos de linguagem...")
    llm = OpenAI(model="gpt-5-chat-latest", temperature=0.0)
    Settings.llm = llm
    interp_llm  = OpenAI(model="gpt-5-mini", temperature=0.0)
    nano_llm    = OpenAI(model="gpt-5-nano",  temperature=0.0)

    # 5. Retriever híbrido compartilhado + reranker
    print("\n[4] Inicializando retrievers...")
    bm25_nodes = load_nodes_cache()
    retriever  = build_hybrid_retriever(index, bm25_nodes)
    reranker   = LLMRerank(top_n=5, choice_batch_size=10, llm=nano_llm)

    # 6. Três retrievers especializados
    text_ret   = TextRetriever(retriever, reranker)
    tables_ret = TablesRetriever(retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(retriever, reranker, llm)

    # 7. Labor Market Skill (opcional — carrega se o arquivo existir)
    labor_skill = LaborMarketSkill(base_dir)
    if labor_skill.is_loaded():
        print("\n[5] Skill de mercado de trabalho carregada.")
    else:
        print("\n[5] Skill de mercado de trabalho não encontrada (opcional).")

    # 8. Analysis Engine
    engine = AnalysisEngine(text_ret, tables_ret, ts_ret, llm, labor_market_skill=labor_skill)

    print("\n[OK] Sistema pronto.\n")
    return engine, interp_llm
