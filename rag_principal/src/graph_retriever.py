"""
Graph Retriever — recupera nós de texto via travessia do grafo de conhecimento.

Combina dois sub-retrievers (PGRetriever):
  - LLMSynonymRetriever: o LLM gera sinônimos da query, casados literalmente
    com os nomes das entidades (normalizados na indexação para o mesmo
    formato .capitalize() que o retriever aplica às keywords);
  - VectorContextRetriever: fallback vetorial sobre os embeddings das
    entidades (modelo local) — cobre os casos em que o sinônimo gerado não
    bate com o nome exato. Só ativa se o grafo foi construído com embeddings.

Deduplicação: remove nós já presentes no contexto de outros retrievers para
evitar repetição de conteúdo na síntese.
"""
from llama_index.core import Settings
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    PGRetriever,
    VectorContextRetriever,
)
from rag_core.logger import get_logger

log = get_logger(__name__)


class GraphRetriever:
    """
    Wrapper sobre PGRetriever (sinônimos + vetorial) para integração com o
    AnalysisEngine.
    """

    def __init__(self, graph_index, llm, vector_store=None):
        graph_store = graph_index.property_graph_store

        sub_retrievers = [
            LLMSynonymRetriever(
                graph_store=graph_store,
                include_text=True,
                max_keywords=10,
                path_depth=2,
                llm=llm,
            )
        ]

        if vector_store is not None:
            sub_retrievers.append(
                VectorContextRetriever(
                    graph_store=graph_store,
                    vector_store=vector_store,
                    embed_model=Settings.embed_model,
                    include_text=True,
                    similarity_top_k=4,
                    path_depth=2,
                )
            )
        else:
            log.info(
                "[Graph] Sem vector store de entidades — usando apenas sinônimos"
            )

        self._retriever = PGRetriever(sub_retrievers=sub_retrievers, use_async=False)

    def retrieve(self, question: str, exclude_ids: set | None = None) -> list:
        """
        Retorna nós de texto recuperados via grafo.

        Parâmetros
        ----------
        question : str
            Query (reescrita pelo interpreter).
        exclude_ids : set | None
            IDs de nós já presentes em outros retrievers — serão filtrados
            para evitar duplicação no contexto.

        Retorna
        -------
        list[NodeWithScore] com conteúdo não-vazio e não-duplicado.
        """
        try:
            nodes = self._retriever.retrieve(question)
        except Exception as exc:
            log.warning("[Graph] Falha no retrieval: %s", exc)
            return []

        seen = exclude_ids or set()
        result = []
        for n in nodes:
            nid = getattr(n.node, "node_id", None) or id(n)
            if nid not in seen and n.get_content().strip():
                result.append(n)
                seen.add(nid)

        log.info("[Graph] %d nós recuperados pelo grafo", len(result))
        return result
