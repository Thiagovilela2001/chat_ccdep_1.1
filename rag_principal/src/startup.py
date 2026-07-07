"""
Startup — inicialização compartilhada do sistema RAG.

Usado tanto pela API (FastAPI lifespan) quanto pelo CLI interativo.
Centraliza: detecção de mudanças, indexação, criação de LLMs e retrievers.
"""
import os
import json

import chromadb
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank

from rag_core.llm import make_llm, require_api_key
from rag_core.logger import get_logger, setup_logging
from rag_core.ingestion import load_documents
from rag_core.processing import process_documents
from rag_core.indexing import create_or_load_index, load_nodes_cache
from rag_core.text_retriever import build_hybrid_retriever, TextRetriever
from rag_core.tables_retriever import TablesRetriever
from rag_core.timeseries_retriever import TimeSeriesRetriever
from src.graph_indexing import build_or_load_graph
from src.graph_retriever import GraphRetriever
from src.analysis_engine import AnalysisEngine
from rag_core.labor_market_skill import LaborMarketSkill

log = get_logger(__name__)

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


# ── Inicialização principal ───────────────────────────────────────────────────

def initialize(base_dir: str, data_dir: str | None = None, use_graph: bool = False) -> tuple[AnalysisEngine, object]:
    """
    Inicializa o sistema completo e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    data_dir : str | None
        Diretório dos documentos PDF. Se None, usa base_dir/data.
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
    data_dir = data_dir or os.path.join(base_dir, "data")
    db_path = os.path.join(base_dir, "chroma_db")

    log.info("Inicializando RAG Estatistico SP")

    # 1. Detecção de mudanças nos documentos
    snapshot = _get_data_snapshot(data_dir)
    manifest = _load_manifest(db_path)
    changed = _detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    # 2. Ingestão e processamento (somente se necessário)
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
        log.info("[1] Banco atualizado (%d vetores) — sem mudancas", col.count())

    # 3. Indexação vetorial
    log.info("[2] Indexando vetores no ChromaDB")
    index = create_or_load_index(nodes, db_path=db_path)

    if changed or not already_indexed:
        _save_manifest(db_path, snapshot)

    # 4. LLMs
    log.info("[3] Carregando modelos de linguagem")
    llm = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm  = make_llm(interp=True, temperature=0.0, timeout=30.0)

    # 5. Retriever híbrido compartilhado + reranker
    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache(db_path)
    retriever  = build_hybrid_retriever(index, bm25_nodes)
    # choice_batch_size=30 iguala o top_k do retriever → todos os candidatos
    # são comparados num único LLM call (sem viés de batch)
    reranker   = LLMRerank(top_n=10, choice_batch_size=30, llm=interp_llm)

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
        graph_index = build_or_load_graph(all_nodes, base_dir, llm, force_rebuild=force_rebuild)
        graph_ret = GraphRetriever(graph_index, interp_llm)
        log.info("[6] Grafo pronto")
    else:
        log.info("[6] Grafo desativado (use --use-graph para habilitar)")

    # 9. Analysis Engine
    engine = AnalysisEngine(
        text_ret, tables_ret, ts_ret, llm,
        labor_market_skill=labor_skill,
        graph_retriever=graph_ret,
    )

    log.info("Sistema pronto")
    return engine, interp_llm
