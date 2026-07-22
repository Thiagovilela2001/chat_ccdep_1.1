"""
Startup — inicialização compartilhada do sistema RAG.

Usado tanto pela API (FastAPI lifespan) quanto pelo CLI interativo.
Centraliza: detecção de mudanças, indexação, criação de LLMs e retrievers.
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
from rag_core.text_retriever import (
    build_hybrid_retriever,
    llm_reranking_enabled,
    ScoreReranker,
    TextRetriever,
)
from rag_core.tables_retriever import TablesRetriever
from rag_core.timeseries_retriever import TimeSeriesRetriever
from .graph_indexing import build_or_load_graph
from .graph_retriever import GraphRetriever
from .analysis_engine import AnalysisEngine
from rag_core.labor_market_skill import LaborMarketSkill

log = get_logger(__name__)


def graph_enabled_by_env() -> bool:
    """True se RAG_USE_GRAPH estiver ligada (1/true/yes/on)."""
    return os.getenv("RAG_USE_GRAPH", "").strip().lower() in {"1", "true", "yes", "on"}


# ── Detecção de mudanças ──────────────────────────────────────────────────────

# ── Inicialização principal ───────────────────────────────────────────────────

def initialize(base_dir: str, data_dir: str | None = None, use_graph: bool = False) -> tuple[AnalysisEngine, object]:
    """
    Inicializa o sistema completo e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    data_dir : str | None
        Diretório dos documentos. Se None, resolve a base local compartilhada
        ou base_dir/data (container).
        Útil quando evaluate.py roda de um diretório diferente do RAG.
    use_graph : bool
        Se True, constrói/carrega o grafo de conhecimento e habilita a
        4ª fonte "graph" no AnalysisEngine.

    Retorna
    -------
    engine : AnalysisEngine
        Engine pronta para receber queries.
    interp_llm : OpenAI
        LLM leve usado pelo Query Interpreter.
    """
    require_api_key()

    setup_logging()
    data_dir = resolve_data_dir(base_dir, data_dir)
    db_path = os.path.join(base_dir, "chroma_db")

    log.info("Inicializando RAG Estatistico SP")

    # 1. Detecção de mudanças nos documentos
    snapshot = data_snapshot(data_dir)
    manifest = load_manifest(db_path)
    changed = detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    # 2. Ingestão e processamento (somente se necessário)
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
        log.info("[1] Banco atualizado (%d vetores) — sem mudancas", col.count())

    # 3. Indexação vetorial
    log.info("[2] Indexando vetores no ChromaDB")
    index = create_or_load_index(nodes, db_path=db_path)

    if changed or not already_indexed:
        save_manifest(db_path, snapshot)

    # 4. LLMs
    log.info("[3] Carregando modelos de linguagem")
    llm = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm  = make_llm(interp=True, temperature=0.0, timeout=30.0)

    # 5. Retriever híbrido compartilhado + reranker
    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache(db_path)
    retriever  = build_hybrid_retriever(index, bm25_nodes)
    if llm_reranking_enabled():
        # Em provedores com janela/latência adequadas, todos os candidatos são
        # comparados numa única chamada para evitar viés entre lotes.
        reranker = LLMRerank(top_n=10, choice_batch_size=30, llm=interp_llm)
    else:
        # Ollama local: preserva o score híbrido e evita prompts maiores que a
        # janela ativa do modelo, além de eliminar uma chamada lenta por busca.
        reranker = ScoreReranker(top_n=5)
        log.info("Reranking por LLM desativado; usando ranking hibrido Vector+BM25")

    # 6. Três retrievers especializados
    text_ret   = TextRetriever(retriever, reranker)
    tables_ret = TablesRetriever(retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(retriever, reranker, llm)

    # 7. Labor Market Skill (opcional — carrega se o arquivo existir)
    labor_skill = LaborMarketSkill(base_dir)
    if labor_skill.is_loaded():
        log.info("[5] Skill de mercado de trabalho carregada")
    else:
        log.info("[5] Skill de mercado de trabalho nao encontrada (opcional)")

    # 8. Grafo de conhecimento (opcional — apenas se use_graph=True)
    graph_ret = None
    if use_graph:
        log.info("[6] Inicializando grafo de conhecimento")
        all_nodes = load_nodes_cache(db_path)
        force_rebuild = bool(changed)
        graph_index, graph_vec = build_or_load_graph(
            all_nodes, base_dir, llm, force_rebuild=force_rebuild
        )
        graph_ret = GraphRetriever(graph_index, interp_llm, vector_store=graph_vec)
        log.info("[6] Grafo pronto")
    else:
        log.info("[6] Grafo desativado (use --graph no CLI ou RAG_USE_GRAPH=1)")

    # 9. Analysis Engine
    engine = AnalysisEngine(
        text_ret, tables_ret, ts_ret, llm,
        labor_market_skill=labor_skill,
        graph_retriever=graph_ret,
    )

    log.info("Sistema pronto")
    return engine, interp_llm
