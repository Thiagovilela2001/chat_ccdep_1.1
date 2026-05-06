"""
Graph Retriever — recupera nós de texto via travessia do grafo de conhecimento.

Usa LLMSynonymRetriever: dado uma query, o LLM gera sinônimos e termos
relacionados, que são usados para encontrar entidades no grafo e retornar
os chunks de texto originais que as mencionam.

Deduplicação: remove nós já presentes no contexto de outros retrievers para
evitar repetição de conteúdo na síntese.
"""
from llama_index.core.indices.property_graph import LLMSynonymRetriever
from src.logger import get_logger

log = get_logger(__name__)


class GraphRetriever:
    """
    Wrapper sobre LLMSynonymRetriever para integração com AnalysisEngine.

    O retriever expande a query com sinônimos de entidades econômicas
    (indicadores, setores, fontes) e busca nós conectados no grafo.
    """

    def __init__(self, graph_index, llm):
        self._retriever = LLMSynonymRetriever(
            graph_store=graph_index.property_graph_store,
            include_text=True,
            max_keywords=10,
            path_depth=2,
            llm=llm,
        )

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
