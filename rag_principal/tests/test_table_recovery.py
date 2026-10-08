from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag_core.query_service import _with_temporal_tables
from rag_core.tables_retriever import TablesRetriever, _is_static_table
from rag_core.timeseries_retriever import TimeSeriesRetriever, _is_temporal_table


def _node(text: str, *, granularidade=None, tipo="table", node_id="node"):
    metadata = {"type": tipo}
    if granularidade is not None:
        metadata["table_granularidade"] = granularidade
    return NodeWithScore(
        node=TextNode(text=text, metadata=metadata, id_=node_id),
        score=0.9,
    )


class _IndiceFalso:
    def __init__(self, nodes):
        self.nodes = nodes

    def retrieve(self, question):
        return list(self.nodes)


class _RerankerFalso:
    def __init__(self):
        self.recebidos = []

    def postprocess_nodes(self, nodes, query_str=None):
        self.recebidos.append(list(nodes))
        return list(nodes)


class _LLMFalso:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def complete(self, prompt):
        self.chamadas.append(prompt)
        return SimpleNamespace(text=self.respostas.pop(0))


@pytest.mark.parametrize("granularidade", ["mensal", "trimestral", "anual"])
def test_granularidade_temporal_e_reconhecida_pelos_dois_classificadores(
    granularidade,
):
    node = _node("Tabela temporal", granularidade=granularidade)

    assert not _is_static_table(node)
    assert _is_temporal_table(node)


@pytest.mark.parametrize("granularidade", [None, ""])
def test_granularidade_vazia_ou_ausente_e_tratada_como_estatica(granularidade):
    node = _node("Fragmento de tabela", granularidade=granularidade)

    assert _is_static_table(node)
    assert not _is_temporal_table(node)


def test_node_que_nao_e_tabela_e_rejeitado_pelos_dois_classificadores():
    node = _node("Trecho narrativo", tipo="text")

    assert not _is_static_table(node)
    assert not _is_temporal_table(node)


@pytest.mark.xfail(
    reason="tabela temporal descartada quando series nao e escolhido: PNAD 76/80"
)
def test_tables_retriever_sozinho_recupera_tabela_temporal():
    temporal = _node(
        "Periodo | Renda\n2023 | 2.900\n2024 | 3.100",
        granularidade="anual",
        node_id="pnad-anual",
    )
    llm = _LLMFalso(
        '{"columns": ["Periodo", "Renda"], "rows": [["2023", 2900], ["2024", 3100]]}',
        '{"resultado": "A renda passou de 2.900 para 3.100."}',
    )
    retriever = TablesRetriever(
        _IndiceFalso([temporal]),
        _RerankerFalso(),
        llm,
    )

    resultado = retriever.retrieve("Como evoluiu a renda da PNAD?")

    assert resultado is not None
    structured, nodes = resultado
    assert structured == "A renda passou de 2.900 para 3.100."
    assert nodes == [temporal]


def test_fontes_adicionam_timeseries_quando_tables_e_selecionado():
    fontes = ["text", "tables"]

    resultado = _with_temporal_tables(fontes)

    assert resultado == ["text", "tables", "timeseries"]
    assert fontes == ["text", "tables"]
    assert resultado is not fontes


def test_fontes_com_timeseries_sao_idempotentes_e_nao_mutadas():
    fontes = ["tables", "timeseries"]

    resultado = _with_temporal_tables(fontes)

    assert resultado == fontes
    assert fontes == ["tables", "timeseries"]
    assert resultado is not fontes


def test_fontes_sem_tables_ficam_inalteradas_e_nao_sao_mutadas():
    fontes = ["text", "graph"]

    resultado = _with_temporal_tables(fontes)

    assert resultado == fontes
    assert fontes == ["text", "graph"]
    assert resultado is not fontes


@pytest.mark.parametrize(
    "payload",
    [
        '{"columns": ["Setor", "Valor"], "rows": []}',
        "{}",
    ],
    ids=["rows-vazio", "objeto-vazio"],
)
def test_rows_invalido_retorna_resultado_vazio_preserva_nodes_e_nao_calcula(payload):
    tabela = _node(
        "Setor | Valor\nIndustria | 10",
        granularidade="",
        node_id="tabela-estatica",
    )
    llm = _LLMFalso(payload)
    retriever = TablesRetriever(
        _IndiceFalso([tabela]),
        _RerankerFalso(),
        llm,
    )

    structured, nodes = retriever.retrieve("Qual o valor da industria?")

    assert structured is None
    assert nodes == [tabela]
    assert len(llm.chamadas) == 1


def test_timeseries_seleciona_so_node_temporal_e_aceita_columns_rows():
    estatica = _node(
        "Setor | Valor\nIndustria | 10",
        granularidade="",
        node_id="estatica",
    )
    temporal = _node(
        "Periodo | Taxa\n2023 | 8.0\n2024 | 7.0",
        granularidade="trimestral",
        node_id="temporal",
    )
    reranker = _RerankerFalso()
    llm = _LLMFalso(
        '{"columns": ["Periodo", "Taxa"], "rows": [["2023", 8.0], ["2024", 7.0]]}',
        '{"resultado": "A taxa caiu de 8,00 para 7,00."}',
    )
    retriever = TimeSeriesRetriever(
        _IndiceFalso([estatica, temporal]),
        reranker,
        llm,
    )

    structured, nodes = retriever.retrieve("Como evoluiu a taxa?")

    assert structured == "A taxa caiu de 8,00 para 7,00."
    assert nodes == [temporal]
    assert reranker.recebidos == [[temporal]]
    assert len(llm.chamadas) == 2
