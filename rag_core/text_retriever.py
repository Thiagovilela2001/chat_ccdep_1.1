"""
Text Retriever — retrieval híbrido (Vector + BM25) para chunks de texto narrativo.

Expõe também build_hybrid_retriever(), usado pelos outros retrievers.
"""
import os
import re

from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from .logger import get_logger
from .llm import provider_name

log = get_logger(__name__)
_FALLBACK_TOP_N = 5


def llm_reranking_enabled() -> bool:
    """Define se os candidatos devem ser reranqueados por um LLM.

    No Ollama local, o lote de candidatos excede facilmente a janela ativa do
    modelo e o processamento em CPU adiciona dezenas de segundos à consulta.
    O ranking híbrido Vector+BM25 é usado por padrão nesse caso.
    """
    raw = os.getenv("RAG_LLM_RERANK")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return provider_name() != "ollama"


class ScoreReranker:
    """Reranker determinístico que preserva a ordem do retriever híbrido."""

    def __init__(self, top_n: int = _FALLBACK_TOP_N):
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_str: str | None = None):
        del query_str
        return list(nodes[:self.top_n])


def _sanitize(text: str) -> str:
    """Remove caracteres de controle inválidos que podem quebrar o JSON da API."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def build_hybrid_retriever(index, bm25_nodes):
    """
    Cria retriever híbrido (Vector + BM25 com fusão Reciprocal Rank).
    Se não houver nós BM25, retorna apenas o retriever vetorial.
    """
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=30)

    if bm25_nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=30,
        )
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=30,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

    return vector_retriever


class TextRetriever:
    """
    Recupera chunks de texto narrativo relevantes para a query.

    Fluxo: retrieve (top-30) → filtra texto → rerank → top-N texto
    Fallback: se o reranker falhar ou retornar vazio, retorna os top-5 por score.
    """

    def __init__(self, retriever, reranker):
        self._retriever = retriever
        self._reranker = reranker

    def retrieve(self, question: str) -> list:
        """Retorna nodes de texto reranqueados. Lista vazia se sem resultados."""
        nodes = self._retriever.retrieve(question)

        # Filtra apenas chunks narrativos (não tabelas)
        text_nodes = [n for n in nodes if n.metadata.get("type") != "table"]
        if not text_nodes:
            return []

        # Sanitiza conteúdo antes do reranker (evita erro 400 na API)
        for n in text_nodes:
            n.node.text = _sanitize(n.node.text)

        try:
            reranked = self._reranker.postprocess_nodes(text_nodes, query_str=question)
        except Exception as exc:
            log.warning(
                "Reranker falhou (%s: %s) — usando fallback por score",
                type(exc).__name__,
                exc,
                extra={"fallback": True},
            )
            reranked = []

        # Fallback: reranker vazio → top-N por score de recuperação
        if not reranked:
            log.warning(
                "Reranker retornou vazio — usando fallback top-%d", _FALLBACK_TOP_N,
                extra={"fallback": True},
            )
            return text_nodes[:_FALLBACK_TOP_N]

        return reranked
