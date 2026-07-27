"""
Startup — inicialização do RAPTOR RAG.

Diferenças em relação ao rag_agentic/src/startup.py:
1. Após processar documentos (leaf nodes), constrói a árvore RAPTOR via
   build_raptor_tree() — agrupa por similaridade semântica e sumariza cada
   cluster com LLM, gerando nós de resumo em múltiplos níveis.
2. Indexa TODOS os nós (folhas + resumos) na coleção "estatisticas_raptor"
   do ChromaDB — separada das outras variantes para não misturar os índices.
3. Instancia RaptorEngine em vez de AgenticEngine.

O custo extra (sumarização dos clusters) ocorre apenas na primeira indexação
ou quando os documentos mudam. Nas execuções seguintes, tudo é carregado do
cache em chroma_db/bm25_nodes.pkl.
"""
import os

import chromadb
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank
from rag_core.index_manifest import (
    data_snapshot,
    detect_changes,
    load_manifest,
    resolve_data_dir,
    resolve_db_dir,
    save_manifest,
)
from rag_core.indexing import (
    create_or_load_index,
    load_nodes_cache,
    reset_nodes_cache,
    setup_embeddings,
)
from rag_core.ingestion import load_documents
from rag_core.labor_market_skill import LaborMarketSkill
from rag_core.llm import interp_model, make_llm, require_api_key
from rag_core.logger import get_logger, setup_logging
from rag_core.processing import process_documents
from .raptor_engine import RaptorEngine
from .raptor_indexing import build_raptor_tree
from rag_core.tables_retriever import TablesRetriever
from rag_core.text_retriever import TextRetriever, build_hybrid_retriever, rerank_top_n
from rag_core.timeseries_retriever import TimeSeriesRetriever

log = get_logger(__name__)

_RAPTOR_COLLECTION = "estatisticas_raptor"


def initialize(base_dir: str, data_dir: str | None = None) -> tuple[RaptorEngine, object]:
    """
    Inicializa o RAPTOR RAG e retorna (engine, interp_llm).

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz do RAG (onde fica /chroma_db).
    data_dir : str | None
        Diretório dos documentos PDF. Se None, usa base_dir/data.
    """
    require_api_key()

    setup_logging()
    data_dir = resolve_data_dir(base_dir, data_dir)
    db_path  = resolve_db_dir(base_dir)

    log.info("Inicializando RAPTOR RAG")

    snapshot = data_snapshot(data_dir)
    manifest = load_manifest(db_path)
    changed  = detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection(_RAPTOR_COLLECTION)
    already_indexed = col.count() > 0

    # ── Fase 1: decidir se precisa reindexar ─────────────────────────────────
    needs_rebuild = bool(changed) or not already_indexed

    raptor_nodes: list = []

    if needs_rebuild:
        if changed:
            log.info("[1] Documentos novos/modificados: %s — reindexando", changed)
        else:
            log.info("[1] Banco vazio — indexando documentos pela primeira vez")

        docs = load_documents(data_dir)
        if not docs:
            raise RuntimeError(
                f"Nenhum documento encontrado em '{data_dir}'; índice anterior preservado."
            )
        log.info("[2] Processando documentos em nós folha")
        leaf_nodes = process_documents(docs)
        if not leaf_nodes:
            raise RuntimeError("Processamento não produziu nós; índice anterior preservado.")

        # ── Fase 2: construção da árvore RAPTOR ──────────────────────────
        log.info("[3] Configurando embeddings para clustering RAPTOR")
        embed_model = setup_embeddings()

        log.info(
            "[3] Construindo árvore RAPTOR sobre %d leaf nodes "
            "(clustering + sumarização LLM)...",
            len(leaf_nodes),
        )
        raptor_nodes = build_raptor_tree(
            leaf_nodes,
            embed_model=embed_model,
            api_key="",  # legado — a chave vem da config central do provedor
            llm_model=interp_model(),
            max_levels=3,
            min_cluster_size=4,
        )
        if not raptor_nodes:
            raise RuntimeError(
                "Construção RAPTOR não produziu nós; índice anterior preservado."
            )

        # Só substitui o índice depois que ingestão e árvore terminaram com sucesso.
        reset_nodes_cache(db_path)
        if already_indexed:
            db.delete_collection(_RAPTOR_COLLECTION)
            col = db.get_or_create_collection(_RAPTOR_COLLECTION)
    else:
        log.info("[1] Banco atualizado (%d vetores) — sem mudanças", col.count())

    # ── Fase 3: indexar todos os nós RAPTOR (ou carregar do cache) ───────────
    log.info("[4] Indexando %d nós RAPTOR no ChromaDB", len(raptor_nodes))
    index = create_or_load_index(
        raptor_nodes if needs_rebuild else [],
        db_path=db_path,
        collection_name=_RAPTOR_COLLECTION,
    )
    # create_or_load_index salva automaticamente em chroma_db/bm25_nodes.pkl

    if needs_rebuild and raptor_nodes:
        save_manifest(db_path, snapshot)

    # ── Fase 4: modelos de linguagem ─────────────────────────────────────────
    log.info("[5] Carregando modelos de linguagem")
    llm        = make_llm(temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm = make_llm(interp=True, temperature=0.0, timeout=30.0)

    # ── Fase 5: retrievers ───────────────────────────────────────────────────
    log.info("[6] Inicializando retrievers sobre índice RAPTOR")
    # BM25 usa todos os nós do índice RAPTOR (folhas + resumos)
    all_nodes = load_nodes_cache(db_path)
    text_retriever = build_hybrid_retriever(
        index, all_nodes, node_type="text", llm=interp_llm
    )
    table_retriever = build_hybrid_retriever(
        index, all_nodes, node_type="table", llm=interp_llm
    )
    reranker = LLMRerank(top_n=rerank_top_n(), choice_batch_size=30, llm=interp_llm)

    text_ret   = TextRetriever(text_retriever, reranker)
    tables_ret = TablesRetriever(table_retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(table_retriever, reranker, llm)

    # ── Fase 6: skill de mercado de trabalho ─────────────────────────────────
    labor_skill = LaborMarketSkill(base_dir)
    if labor_skill.is_loaded():
        log.info("[7] Skill de mercado de trabalho carregada")
    else:
        log.info("[7] Skill de mercado de trabalho não encontrada (opcional)")

    engine = RaptorEngine(text_ret, tables_ret, ts_ret, llm, labor_market_skill=labor_skill)

    log.info("RAPTOR RAG pronto")
    return engine, interp_llm
