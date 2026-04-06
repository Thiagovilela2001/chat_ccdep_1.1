"""
Text Retriever — retrieval híbrido (Vector + BM25) para chunks de texto narrativo.

Expõe também build_hybrid_retriever(), usado pelos outros retrievers.
"""
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


def build_hybrid_retriever(index, bm25_nodes):
    """
    Cria retriever híbrido (Vector + BM25 com fusão Reciprocal Rank).
    Se não houver nós BM25, retorna apenas o retriever vetorial.
    """
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=20)

    if bm25_nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=20,
        )
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=20,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

    return vector_retriever


class TextRetriever:
    """
    Recupera chunks de texto narrativo relevantes para a query.

    Fluxo: retrieve (top-20, todos os tipos) → filtra texto → rerank → top-5 texto
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

        return self._reranker.postprocess_nodes(text_nodes, query_str=question)
