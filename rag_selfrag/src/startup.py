"""
Startup — inicialização do Self-RAG.

Idêntico ao rag_agentic/src/startup.py, exceto que instancia
SelfRAGEngine em vez de AgenticEngine.
"""
import os

from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from rag_core.llm import make_llm, require_api_key
from rag_core.logger import get_logger, setup_logging
from rag_core.index_manifest import (
    resolve_data_dir,
    resolve_db_dir,
)
from rag_core.index_sync import sync_standard_index
from rag_core.indexing import load_nodes_cache
from rag_core.text_retriever import build_hybrid_retriever, rerank_top_n, TextRetriever
from rag_core.tables_retriever import TablesRetriever
from rag_core.timeseries_retriever import TimeSeriesRetriever
from .self_rag_engine import SelfRAGEngine
from rag_core.domain_skills import DomainSkillRegistry

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
    db_path  = resolve_db_dir(base_dir)

    log.info("Inicializando Self-RAG")

    index, _changed = sync_standard_index(data_dir, db_path, log)

    log.info("[3] Carregando modelos de linguagem")
    llm        = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm = make_llm(interp=True, temperature=0.0, timeout=30.0)

    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache(db_path)
    text_retriever = build_hybrid_retriever(
        index, bm25_nodes, node_type="text", llm=interp_llm
    )
    table_retriever = build_hybrid_retriever(
        index, bm25_nodes, node_type="table", llm=interp_llm
    )
    reranker = LLMRerank(top_n=rerank_top_n(), choice_batch_size=30, llm=interp_llm)

    text_ret   = TextRetriever(text_retriever, reranker)
    tables_ret = TablesRetriever(table_retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(table_retriever, reranker, llm)

    domain_skills = DomainSkillRegistry(base_dir)
    if domain_skills.is_loaded():
        log.info("[5] Skills de domínio carregadas: %s", domain_skills.available_domains())
    else:
        log.info("[5] Skills de domínio não encontradas (opcional)")

    engine = SelfRAGEngine(text_ret, tables_ret, ts_ret, llm, domain_skills=domain_skills)

    log.info("Self-RAG pronto")
    return engine, interp_llm
