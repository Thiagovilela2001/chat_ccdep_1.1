"""
Startup — inicialização do Self-RAG.

Idêntico ao rag_agentic/src/startup.py, exceto que instancia
SelfRAGEngine em vez de AgenticEngine.
"""
import os

import chromadb
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from rag_core.llm import make_llm, require_api_key
from rag_core.logger import get_logger, setup_logging
from rag_core.ingestion import load_documents
from rag_core.processing import process_documents
from rag_core.index_manifest import (
    data_snapshot,
    detect_changes,
    load_manifest,
    resolve_data_dir,
    save_manifest,
)
from rag_core.indexing import create_or_load_index, load_nodes_cache, reset_nodes_cache
from rag_core.text_retriever import build_hybrid_retriever, TextRetriever
from rag_core.tables_retriever import TablesRetriever
from rag_core.timeseries_retriever import TimeSeriesRetriever
from .self_rag_engine import SelfRAGEngine
from rag_core.labor_market_skill import LaborMarketSkill

log = get_logger(__name__)

def initialize(base_dir: str, data_dir: str | None = None) -> tuple[SelfRAGEngine, object]:
    """
    Inicializa o Self-RAG e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    data_dir : str | None
        Diretório dos documentos. Se None, resolve a base compartilhada local
        ou base_dir/data (container).
    """
    require_api_key()

    setup_logging()
    data_dir = resolve_data_dir(base_dir, data_dir)
    db_path  = os.path.join(base_dir, "chroma_db")

    log.info("Inicializando Self-RAG")

    snapshot = data_snapshot(data_dir)
    manifest = load_manifest(db_path)
    changed  = detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    nodes = []
    if changed:
        log.info("[1] Documentos novos/modificados: %s — reindexando", changed)
        docs = load_documents(data_dir)
        if not docs:
            raise RuntimeError(
                f"Nenhum documento encontrado em '{data_dir}'; índice anterior preservado."
            )
        nodes = process_documents(docs)
        if not nodes:
            raise RuntimeError("Processamento não produziu nós; índice anterior preservado.")
        reset_nodes_cache(db_path)
        db.delete_collection("estatisticas")
        col = db.get_or_create_collection("estatisticas")
    elif not already_indexed:
        log.info("[1] Banco vazio — carregando documentos")
        reset_nodes_cache(db_path)
        docs = load_documents(data_dir)
        if docs:
            nodes = process_documents(docs)
        else:
            raise RuntimeError(f"Nenhum documento encontrado em '{data_dir}'.")
    else:
        log.info("[1] Banco atualizado (%d vetores) — sem mudanças", col.count())

    log.info("[2] Indexando vetores no ChromaDB")
    index = create_or_load_index(nodes, db_path=db_path)

    if changed or not already_indexed:
        save_manifest(db_path, snapshot)

    log.info("[3] Carregando modelos de linguagem")
    llm        = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm = make_llm(interp=True, temperature=0.0, timeout=30.0)

    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache(db_path)
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
