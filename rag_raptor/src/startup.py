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
import json
import os

import chromadb
from llama_index.core import Settings
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI

from rag_core.indexing import create_or_load_index, load_nodes_cache, setup_embeddings
from rag_core.ingestion import load_documents
from rag_core.labor_market_skill import LaborMarketSkill
from rag_core.logger import get_logger, setup_logging
from rag_core.processing import process_documents
from src.raptor_engine import RaptorEngine
from src.raptor_indexing import build_raptor_tree
from rag_core.tables_retriever import TablesRetriever
from rag_core.text_retriever import TextRetriever, build_hybrid_retriever
from rag_core.timeseries_retriever import TimeSeriesRetriever

log = get_logger(__name__)

_RAPTOR_COLLECTION = "estatisticas_raptor"


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
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY não encontrada. "
            "Crie um arquivo .env com: OPENAI_API_KEY=sua_chave"
        )

    setup_logging()
    data_dir = data_dir or os.path.join(base_dir, "data")
    db_path  = os.path.join(base_dir, "chroma_db")

    log.info("Inicializando RAPTOR RAG")

    snapshot = _get_data_snapshot(data_dir)
    manifest = _load_manifest(db_path)
    changed  = _detect_changes(snapshot, manifest)

    db = chromadb.PersistentClient(path=db_path)
    col = db.get_or_create_collection(_RAPTOR_COLLECTION)
    already_indexed = col.count() > 0

    # ── Fase 1: decidir se precisa reindexar ─────────────────────────────────
    needs_rebuild = bool(changed) or not already_indexed

    raptor_nodes: list = []

    if needs_rebuild:
        if changed:
            log.info("[1] Documentos novos/modificados: %s — reindexando", changed)
            db.delete_collection(_RAPTOR_COLLECTION)
            col = db.get_or_create_collection(_RAPTOR_COLLECTION)
        else:
            log.info("[1] Banco vazio — indexando documentos pela primeira vez")

        docs = load_documents(data_dir)
        if docs:
            log.info("[2] Processando documentos em nós folha")
            leaf_nodes = process_documents(docs)

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
                api_key=os.environ["OPENAI_API_KEY"],
                llm_model=os.getenv("RAG_INTERP_MODEL", "gpt-5-mini"),
                max_levels=3,
                min_cluster_size=4,
            )
        else:
            log.warning("Nenhum documento encontrado em /data")
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
        _save_manifest(db_path, snapshot)

    # ── Fase 4: modelos de linguagem ─────────────────────────────────────────
    log.info("[5] Carregando modelos de linguagem")
    llm        = OpenAI(model=os.getenv("RAG_LLM_MODEL", "gpt-5-chat-latest"), temperature=0.0, timeout=60.0)
    Settings.llm = llm
    interp_llm = OpenAI(model=os.getenv("RAG_INTERP_MODEL", "gpt-5-mini"), temperature=0.0, timeout=30.0)

    # ── Fase 5: retrievers ───────────────────────────────────────────────────
    log.info("[6] Inicializando retrievers sobre índice RAPTOR")
    # BM25 usa todos os nós do índice RAPTOR (folhas + resumos)
    all_nodes = load_nodes_cache(db_path)
    retriever = build_hybrid_retriever(index, all_nodes)
    reranker  = LLMRerank(top_n=10, choice_batch_size=30, llm=interp_llm)

    text_ret   = TextRetriever(retriever, reranker)
    tables_ret = TablesRetriever(retriever, reranker, llm)
    ts_ret     = TimeSeriesRetriever(retriever, reranker, llm)

    # ── Fase 6: skill de mercado de trabalho ─────────────────────────────────
    labor_skill = LaborMarketSkill(base_dir)
    if labor_skill.is_loaded():
        log.info("[7] Skill de mercado de trabalho carregada")
    else:
        log.info("[7] Skill de mercado de trabalho não encontrada (opcional)")

    engine = RaptorEngine(text_ret, tables_ret, ts_ret, llm, labor_market_skill=labor_skill)

    log.info("RAPTOR RAG pronto")
    return engine, interp_llm
