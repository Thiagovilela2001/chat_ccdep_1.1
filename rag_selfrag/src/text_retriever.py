"""
Text Retriever — retrieval híbrido (Vector + BM25) para chunks de texto narrativo.
"""
import re

from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from src.logger import get_logger

log = get_logger(__name__)
_FALLBACK_TOP_N = 5


def _sanitize(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def build_hybrid_retriever(index, bm25_nodes):
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
    def __init__(self, retriever, reranker):
        self._retriever = retriever
        self._reranker = reranker

    def retrieve(self, question: str) -> list:
        nodes = self._retriever.retrieve(question)

        text_nodes = [n for n in nodes if n.metadata.get("type") != "table"]
        if not text_nodes:
            return []

        for n in text_nodes:
            n.node.text = _sanitize(n.node.text)

        try:
            reranked = self._reranker.postprocess_nodes(text_nodes, query_str=question)
        except Exception:
            log.warning("Reranker falhou — usando fallback por score", extra={"fallback": True})
            reranked = []

        if not reranked:
            log.warning(
                "Reranker retornou vazio — usando fallback top-%d", _FALLBACK_TOP_N,
                extra={"fallback": True},
            )
            return text_nodes[:_FALLBACK_TOP_N]

        return reranked