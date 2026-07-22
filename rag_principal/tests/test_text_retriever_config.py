from llama_index.core.schema import NodeWithScore, TextNode

from rag_core.text_retriever import llm_reranking_enabled, ScoreReranker


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
