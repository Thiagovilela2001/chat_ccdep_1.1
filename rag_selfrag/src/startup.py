"""
Startup — inicialização do Self-RAG.

Idêntico ao rag_agentic/src/startup.py, exceto que instancia
SelfRAGEngine em vez de AgenticEngine.
"""
import os
import json

import chromadb
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from src.logger import get_logger, setup_logging
from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index, load_nodes_cache
from src.text_retriever import build_hybrid_retriever, TextRetriever
from src.tables_retriever import TablesRetriever
from src.timeseries_retriever import TimeSeriesRetriever
from src.self_rag_engine import SelfRAGEngine
from src.labor_market_skill import LaborMarketSkill

log = get_logger(__name__)

MANIFEST_FILE = "chroma_db/indexed_manifest.json"


def _get_data_snapshot(data_dir: str) -> dict:
    snapshot = {}
    if not os.path.isdir(data_dir):
        return snapshot
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            snapshot[fname] = os.path.getmtime(fpath)
    return snapshot


def _load_manifest(db_path: str) -> dict:
    path = os.path.join(db_path, "indexed_manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(db_path: str, snapshot: dict) -> None:
    os.makedirs(db_path, exist_ok=True)
    path = os.path.join(db_path, "indexed_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def _detect_changes(snapshot: dict, manifest: dict) -> list[str]:
    return [
        fname for fname, mtime in snapshot.items()
        if fname not in manifest or manifest[fname] != mtime
    ]


def initialize(base_dir: str, data_dir: str | None = None) -> tuple[SelfRAGEngine, object]:
    """
    Inicializa o Self-RAG e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    data_dir : str | None
        Diretório dos documentos PDF. Se None, usa base_dir/data.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY não encontrada. "
            "Crie um arquivo .env com: OPENAI_API_KEY=sua_chave"
        )

    setup_logging()
    data_dir = data_dir or os.path.join(base_dir, "data")
    db_path  = os.path.join(base_dir, "chroma_db")

    log.info("Inicializando Self-RAG")

    snapshot = _get_data_snapshot(data_dir)
    manifest = _load_manifest(db_path)
    changed  = _detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    nodes = []
    if changed:
        log.info("[1] Documentos novos/modificados: %s — reindexando", changed)
        db.delete_collection("estatisticas")
        col = db.get_or_create_collection("estatisticas")
        docs = load_documents(data_dir)
        if docs:
            nodes = process_documents(docs)
    elif not already_indexed:
        log.info("[1] Banco vazio — carregando documentos")
        docs = load_documents(data_dir)
        if docs:
            nodes = process_documents(docs)
        else:
            log.warning("Nenhum documento encontrado em /data")
    else:
        log.info("[1] Banco atualizado (%d vetores) — sem mudanças", col.count())

    log.info("[2] Indexando vetores no ChromaDB")
    index = create_or_load_index(nodes, db_path=db_path)

    if changed or not already_indexed:
        _save_manifest(db_path, snapshot)

    log.info("[3] Carregando modelos de linguagem")
    llm        = OpenAI(model="gpt-5-chat-latest", temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm = OpenAI(model="gpt-5-mini", temperature=0.0, timeout=30.0)

    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache()
    retriever  = build_hybrid_retriever(index, bm25_nodes)
    reranker   = LLMRerank(top_n=10, choice_batch_size=30, llm=interp_llm)

    text_ret   = TextRetriever(retriever, reranker)
    tables_ret = TablesRetriever(retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(retriever, reranker, llm)

    labor_skill = LaborMarketSkill(base_dir)
    if labor_skill.is_loaded():
        log.info("[5] Skill de mercado de trabalho carregada")
    else:
        log.info("[5] Skill de mercado de trabalho não encontrada (opcional)")

    engine = SelfRAGEngine(text_ret, tables_ret, ts_ret, llm, labor_market_skill=labor_skill)

    log.info("Self-RAG pronto")
    return engine, interp_llm
