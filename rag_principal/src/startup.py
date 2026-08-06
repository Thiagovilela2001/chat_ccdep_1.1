"""
Startup — inicialização compartilhada do sistema RAG.

Usado tanto pela API (FastAPI lifespan) quanto pelo CLI interativo.
Centraliza: detecção de mudanças, indexação, criação de LLMs e retrievers.
"""
import os
from pathlib import Path

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
from scripts.index_artifact import (
    DEFAULT_RELEASE_ASSET,
    DEFAULT_RELEASE_REPO,
    DEFAULT_RELEASE_TAG,
    ensure_release_index,
)
from rag_core.text_retriever import (
    build_hybrid_retriever,
    llm_reranking_enabled,
    rerank_top_n,
    ScoreReranker,
    TextRetriever,
)
from rag_core.tables_retriever import TablesRetriever
from rag_core.timeseries_retriever import TimeSeriesRetriever
from .graph_indexing import build_or_load_graph
from .graph_retriever import GraphRetriever
from .analysis_engine import AnalysisEngine
from rag_core.domain_skills import DomainSkillRegistry

log = get_logger(__name__)


def graph_enabled_by_env() -> bool:
    """True se RAG_USE_GRAPH estiver ligada (1/true/yes/on)."""
    return os.getenv("RAG_USE_GRAPH", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def ensure_principal_index(db_path: str) -> None:
    """Baixa índice portátil quando banco local ainda não existe."""
    if not _env_enabled("RAG_INDEX_AUTO_DOWNLOAD"):
        return

    os.environ.setdefault("RAG_INDEX_READ_ONLY", "1")
    repo = os.getenv("RAG_INDEX_REPO", DEFAULT_RELEASE_REPO)
    tag = os.getenv("RAG_INDEX_TAG", DEFAULT_RELEASE_TAG)
    asset = os.getenv("RAG_INDEX_ASSET", DEFAULT_RELEASE_ASSET)
    token = os.getenv("GITHUB_TOKEN") or None
    try:
        timeout = float(os.getenv("RAG_INDEX_DOWNLOAD_TIMEOUT", "600"))
    except ValueError as exc:
        raise RuntimeError("RAG_INDEX_DOWNLOAD_TIMEOUT deve ser numérico.") from exc

    log.info("[0] Verificando índice vetorial portátil")
    count, downloaded = ensure_release_index(
        target=Path(db_path),
        repo=repo,
        tag=tag,
        asset=asset,
        token=token,
        timeout=timeout,
    )
    if downloaded:
        log.info("[0] Índice baixado e validado (%d vetores)", count)
    else:
        log.info("[0] Índice local válido (%d vetores); download dispensado", count)


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
    db_path = resolve_db_dir(base_dir)

    log.info("Inicializando RAG Estatistico SP")

    # 0. Bootstrap portátil: baixa uma vez, valida e impede reindexação implícita.
    ensure_principal_index(db_path)

    # 1–3. Detecção, ingestão seletiva e sincronização vetorial/BM25
    index, changed = sync_standard_index(data_dir, db_path, log)

    # 4. LLMs
    log.info("[3] Carregando modelos de linguagem")
    llm = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm  = make_llm(interp=True, temperature=0.0, timeout=30.0)

    # 5. Retriever híbrido compartilhado + reranker
    log.info("[4] Inicializando retrievers")
    bm25_nodes = load_nodes_cache(db_path)
    text_retriever = build_hybrid_retriever(
        index, bm25_nodes, node_type="text", llm=interp_llm
    )
    table_retriever = build_hybrid_retriever(
        index, bm25_nodes, node_type="table", llm=interp_llm
    )
    if llm_reranking_enabled():
        # Em provedores com janela/latência adequadas, o LLM refina os
        # candidatos híbridos em lotes controlados.
        reranker = LLMRerank(top_n=rerank_top_n(), choice_batch_size=30, llm=interp_llm)
    else:
        # Ollama local: preserva o score híbrido e evita prompts maiores que a
        # janela ativa do modelo, além de eliminar uma chamada lenta por busca.
        reranker = ScoreReranker(top_n=rerank_top_n())
        log.info("Reranking por LLM desativado; usando ranking hibrido Vector+BM25")

    # 6. Três retrievers especializados
    text_ret   = TextRetriever(text_retriever, reranker)
    tables_ret = TablesRetriever(table_retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(table_retriever, reranker, llm)

    # 7. Skills de domínio (opcionais — descobertas em .agents/skills)
    domain_skills = DomainSkillRegistry(base_dir)
    if domain_skills.is_loaded():
        log.info("[5] Skills de domínio carregadas: %s", domain_skills.available_domains())
    else:
        log.info("[5] Skills de domínio não encontradas (opcional)")

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
        domain_skills=domain_skills,
        graph_retriever=graph_ret,
    )

    log.info("Sistema pronto")
    return engine, interp_llm
