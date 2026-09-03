from llama_index.core.schema import NodeWithScore, TextNode

from rag_core.text_retriever import (
    _diversify_by_document,
    _nodes_by_type,
    deduplicate_nodes,
    llm_reranking_enabled,
    query_fusion_queries,
    retrieval_top_k,
    ScoreReranker,
    complete_coverage_chunks_per_document,
)

def test_ollama_uses_hybrid_scores_without_llm_by_default(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("RAG_LLM_RERANK", raising=False)

    assert llm_reranking_enabled() is False


def test_ollama_llm_reranking_can_be_enabled(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RAG_LLM_RERANK", "1")

    assert llm_reranking_enabled() is True


def test_score_reranker_preserves_hybrid_order_and_limit():
    nodes = [
        NodeWithScore(node=TextNode(text=f"Trecho {index}"), score=1.0 / index)
        for index in range(1, 8)
    ]

    reranked = ScoreReranker(top_n=5).postprocess_nodes(nodes, query_str="pergunta")

    assert reranked == nodes[:5]


def test_recuperacao_ampla_e_expansao_sao_configuraveis(monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_TOP_K", "120")
    monkeypatch.setenv("RAG_QUERY_FUSION_QUERIES", "3")

    assert retrieval_top_k() == 120
    assert query_fusion_queries() == 3


def test_pools_separam_texto_de_tabela():
    text = TextNode(text="Narrativa", metadata={"type": "text"})
    table = TextNode(text="Tabela", metadata={"type": "table"})

    assert _nodes_by_type([text, table], "text") == [text]
    assert _nodes_by_type([text, table], "table") == [table]


def test_diversidade_prioriza_documentos_sem_perder_resultados(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "1")
    nodes = [
        NodeWithScore(node=TextNode(text="A1", metadata={"source_file": "a.pdf"}), score=1),
        NodeWithScore(node=TextNode(text="A2", metadata={"source_file": "a.pdf"}), score=.9),
        NodeWithScore(node=TextNode(text="B1", metadata={"source_file": "b.pdf"}), score=.8),
        NodeWithScore(node=TextNode(text="C1", metadata={"source_file": "c.pdf"}), score=.7),
    ]

    selected = _diversify_by_document(nodes, limit=4)

    assert [node.node.text for node in selected] == ["A1", "B1", "C1", "A2"]


def test_cobertura_temporal_permite_mais_trechos_do_mesmo_documento(monkeypatch):
    monkeypatch.setenv("RAG_COMPLETE_COVERAGE_CHUNKS_PER_DOCUMENT", "8")
    assert complete_coverage_chunks_per_document() == 8


def test_deduplicate_nodes_removes_duplicate_ids_and_texts():
    n1 = NodeWithScore(node=TextNode(id_="id1", text="Texto duplicado"), score=1.0)
    n2 = NodeWithScore(node=TextNode(id_="id1", text="Texto duplicado"), score=0.9)
    n3 = NodeWithScore(node=TextNode(id_="id2", text="Texto duplicado"), score=0.8)
    n4 = NodeWithScore(node=TextNode(id_="id3", text="Texto unico"), score=0.7)

    deduped = deduplicate_nodes([n1, n2, n3, n4])
    assert len(deduped) == 2
    assert [n.node.id_ for n in deduped] == ["id1", "id3"]

