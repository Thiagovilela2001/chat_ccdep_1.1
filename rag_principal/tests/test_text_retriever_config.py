import pytest
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from rag_core.text_retriever import (
    _diagnosing_parser,
    _diversify_by_document,
    _nodes_by_type,
    _RerankDiagnosis,
    build_llm_reranker,
    deduplicate_nodes,
    llm_reranking_enabled,
    query_fusion_queries,
    rerank_top_n,
    retrieval_top_k,
    ScoreReranker,
    TolerantLLMReranker,
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


class _ScriptedReranker:
    """Reranker de teste que registra a janela recebida e devolve o roteiro."""

    def __init__(self, script):
        self.script = script
        self.windows: list[int] = []

    def postprocess_nodes(self, nodes, query_str=None):
        self.windows.append(len(nodes))
        return self.script(nodes)


def _scored(count):
    return [
        NodeWithScore(node=TextNode(text=f"Trecho {index}"), score=float(count - index))
        for index in range(count)
    ]


def test_lote_recusado_volta_na_ordem_hibrida():
    nodes = _scored(40)
    aceitos = [nodes[2], nodes[5]]
    primary = _ScriptedReranker(
        lambda batch: aceitos if len(batch) > 10 else []
    )

    result = TolerantLLMReranker(primary, top_n=24).postprocess_nodes(nodes)

    assert result[:2] == aceitos
    assert [node.node.text for node in result[2:]] == [
        node.node.text for node in nodes if node not in aceitos
    ]
    assert len(result) == 40


def test_aceitos_sao_ordenados_por_nota_e_cortados_em_top_n():
    nodes = _scored(12)
    primary = _ScriptedReranker(lambda batch: [nodes[7], nodes[3]])

    result = TolerantLLMReranker(primary, top_n=1).postprocess_nodes(nodes)

    # _scored dá nota maior ao índice menor: nodes[3] tem score 9, nodes[7] tem 5
    assert result[0] is nodes[3]
    assert len(result) == 12


def test_repeticao_em_janela_menor_quando_nenhum_lote_tem_veredito():
    nodes = _scored(40)
    aceitos = [nodes[1]]
    primary = _ScriptedReranker(lambda batch: [])
    retry = _ScriptedReranker(lambda batch: aceitos)

    result = TolerantLLMReranker(primary, retry, top_n=24).postprocess_nodes(nodes)

    assert result[0] is aceitos[0]
    assert primary.windows == [30, 10]
    # a repetição reexamina os 20 primeiros em dois lotes de 10
    assert retry.windows == [10, 10]


def test_sem_repeticao_quando_nao_ha_janela_menor():
    nodes = _scored(18)
    primary = _ScriptedReranker(lambda batch: [])
    retry = _ScriptedReranker(lambda batch: [nodes[0]])

    result = TolerantLLMReranker(primary, retry, top_n=24).postprocess_nodes(nodes)

    assert result == nodes
    assert retry.windows == []


def test_falha_em_um_lote_nao_derruba_os_outros():
    nodes = _scored(40)

    def script(batch):
        if len(batch) > 10:
            raise TimeoutError("LLM nao respondeu")
        return [nodes[35]]

    result = TolerantLLMReranker(_ScriptedReranker(script), top_n=24).postprocess_nodes(nodes)

    assert result[0] is nodes[35]
    assert len(result) == 40


def test_sem_veredito_algum_preserva_todos_os_candidatos():
    nodes = _scored(40)
    primary = _ScriptedReranker(lambda batch: [])

    result = TolerantLLMReranker(primary, primary, top_n=24).postprocess_nodes(nodes)

    assert result == nodes


def test_parser_registra_resposta_sem_escolhas():
    diagnosis = _RerankDiagnosis()
    parse = _diagnosing_parser(diagnosis)

    choices, _ = parse("Doc: 4, Relevance: 8\n\nApenas o Documento 4 trata do tema.", 40)
    assert choices == [4]

    prosa = (
        "None of the provided documents are relevant to answering the question. "
        "Relevance: 0 for all documents."
    )
    choices, relevances = parse(prosa, 40)
    assert choices == [] and relevances == []
    assert diagnosis.raw == prosa


def test_factory_configura_lote_maior_e_repeticao_menor():
    reranker = build_llm_reranker(MockLLM(), batch_size=30)

    assert isinstance(reranker, TolerantLLMReranker)
    assert reranker._batch_size == 30
    assert reranker._top_n == rerank_top_n()
    assert reranker._batch_reranker.choice_batch_size == 30
    assert reranker._retry_reranker.choice_batch_size == 10

