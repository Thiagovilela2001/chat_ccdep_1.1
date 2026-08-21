import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_core.demographic_indicators import (
    DemographicCalculationError,
    calculate_demographic_indicators,
    is_demographic_indicator_query,
)
from rag_core.numerical_validator import validate_numbers
from rag_core.tables_retriever import TablesRetriever
from rag_orchestrator.src.orchestrator import Orchestrator
from rag_principal.src.query_interpreter import interpret_query


def _frame(rows=None):
    values = rows or [
        [0, 14, 200, "Estado de São Paulo", 2022, "pessoas"],
        [15, 64, 500, "Estado de São Paulo", 2022, "pessoas"],
        [65, None, 100, "Estado de São Paulo", 2022, "pessoas"],
    ]
    values = [
        [*row, "demografia.pdf", 10] if len(row) == 6 else row
        for row in values
    ]
    return pd.DataFrame(
        values,
        columns=[
            "idade_inicial",
            "idade_final",
            "populacao",
            "territorio",
            "periodo",
            "unidade",
            "fonte",
            "pagina",
        ],
    )


def test_calcula_indice_e_razoes_com_formulas_explicitas():
    result = calculate_demographic_indicators(
        "Calcule o índice de envelhecimento e as razões de dependência.",
        _frame(),
    )

    assert "P(0–14)=200" in result
    assert "P(15–64)=500" in result
    assert "P(65+)=100" in result
    assert "100 / 200 × 100 = 50,00" in result
    assert "200 / 500 × 100 = 40,00" in result
    assert "100 / 500 × 100 = 20,00" in result
    assert "(200 + 100) / 500 × 100 = 60,00" in result


def test_agrega_faixas_detalhadas_sem_usar_llm_para_conta():
    result = calculate_demographic_indicators(
        "Calcule o índice de envelhecimento.",
        _frame(
            [
                [0, 4, 50, "SP", 2022, "pessoas"],
                [5, 9, 70, "SP", 2022, "pessoas"],
                [10, 14, 80, "SP", 2022, "pessoas"],
                [15, 39, 250, "SP", 2022, "pessoas"],
                [40, 64, 250, "SP", 2022, "pessoas"],
                [65, 79, 60, "SP", 2022, "pessoas"],
                [80, None, 40, "SP", 2022, "pessoas"],
            ]
        ),
    )

    assert "P(0–14)=200" in result
    assert "100 / 200 × 100 = 50,00" in result


def test_bloqueia_faixa_que_cruza_limite_padrao():
    frame = _frame(
        [
            [0, 14, 200, "SP", 2022, "pessoas"],
            [15, 59, 450, "SP", 2022, "pessoas"],
            [60, None, 150, "SP", 2022, "pessoas"],
        ]
    )

    with pytest.raises(DemographicCalculationError, match="cruza os limites"):
        calculate_demographic_indicators("Razão de dependência", frame)


def test_bloqueia_faixas_com_lacuna():
    frame = _frame(
        [
            [0, 14, 200, "SP", 2022, "pessoas"],
            [16, 64, 500, "SP", 2022, "pessoas"],
            [65, None, 100, "SP", 2022, "pessoas"],
        ]
    )

    with pytest.raises(DemographicCalculationError, match="lacuna ou sobreposição"):
        calculate_demographic_indicators("Razão de dependência", frame)


def test_nao_combina_faixas_de_documentos_diferentes():
    frame = _frame(
        [
            [0, 14, 200, "SP", 2022, "pessoas", "a.pdf", 1],
            [15, 64, 500, "SP", 2022, "pessoas", "b.pdf", 2],
            [65, None, 100, "SP", 2022, "pessoas", "b.pdf", 2],
        ]
    )

    with pytest.raises(DemographicCalculationError):
        calculate_demographic_indicators("Razão de dependência", frame)


def test_detecta_apenas_pergunta_atual_quando_ha_memoria():
    contextual = """
    <HISTORICO_DA_CONVERSA>
    USUÁRIO: Calcule o índice de envelhecimento.
    </HISTORICO_DA_CONVERSA>
    <PERGUNTA_ATUAL>Qual foi o PIB?</PERGUNTA_ATUAL>
    """
    assert not is_demographic_indicator_query(contextual)


class _ExtractLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.payload)


def test_tables_retriever_nao_delega_calculo_demografico_ao_llm():
    llm = _ExtractLLM(
        '{"columns":["idade_inicial","idade_final","populacao","territorio",'
        '"periodo","unidade","fonte","pagina"],'
        '"rows":[[0,14,200,"SP",2022,"pessoas","demo.pdf",10],'
        '[15,64,500,"SP",2022,"pessoas","demo.pdf",10],'
        '[65,null,100,"SP",2022,"pessoas","demo.pdf",10]]}'
    )
    retriever = TablesRetriever(object(), object(), llm)

    result = retriever._extract_and_calculate(
        "Calcule o índice de envelhecimento e as razões de dependência.",
        "Tabela etária",
    )

    assert len(llm.prompts) == 1
    assert "idade_inicial" in llm.prompts[0]
    assert "Índice de envelhecimento" in result
    assert "Razão de dependência total" in result


def test_tables_retriever_explica_bloqueio_sem_segunda_chamada_llm():
    llm = _ExtractLLM("JSON inválido")
    retriever = TablesRetriever(object(), object(), llm)

    result = retriever._extract_and_calculate(
        "Calcule o índice de envelhecimento.",
        "Tabela incompleta",
    )

    assert len(llm.prompts) == 1
    assert "Status: não realizado" in result
    assert "Objeto JSON não encontrado" in result


def test_resultados_demograficos_derivados_sao_validados_pela_formula():
    result = calculate_demographic_indicators(
        "Calcule o índice de envelhecimento e as razões de dependência.",
        _frame(),
    )
    node = SimpleNamespace(
        metadata={"type": "table"},
        get_content=lambda: (
            "Faixas e populações: 0–14: 200; 15–64: 500; 65+: 100; período: 2022."
        ),
    )

    checks = validate_numbers(result, [node])
    by_value = {check.value: check for check in checks}

    for value in ("50,00", "40,00", "20,00", "60,00"):
        assert by_value[value].verified
        assert by_value[value].derived


def test_resultado_demografico_incorreto_e_rejeitado():
    node = SimpleNamespace(
        metadata={"type": "table"},
        get_content=lambda: "População 0–14: 200; população 65+: 100.",
    )

    checks = validate_numbers(
        "Índice de envelhecimento: 100 / 200 × 100 = 51,00.",
        [node],
    )

    by_value = {check.value: check for check in checks}
    assert not by_value["51,00"].verified


def test_orquestrador_forca_engine_deterministica_mesmo_em_modo_multi():
    analyzer = SimpleNamespace(
        model="fake",
        analyze=lambda _question: {
            "query_type": "ampla",
            "confidence": 0.2,
            "in_scope": False,
            "priority": "abrangencia",
            "retrieval_need": "semantica",
        },
    )

    result = asyncio.run(
        Orchestrator(analyzer=analyzer, multi_engine=True).route_only(
            "Calcule o índice de envelhecimento e as razões de dependência."
        )
    )

    assert result["route"]["mode"] == "single_best"
    assert result["route"]["engines_used"] == ["principal"]
    assert result["analysis"]["query_type"] == "tabular"
    assert result["analysis"]["in_scope"] is True


@pytest.mark.parametrize(
    "llm_output",
    ['{"sources":["text"],"rewritten_query":"estrutura etária SP 2022"}', "inválido"],
)
def test_interpretador_sempre_inclui_tabelas_para_calculo_demografico(llm_output):
    llm = SimpleNamespace(complete=lambda _prompt: SimpleNamespace(text=llm_output))

    result = interpret_query("Calcule o índice de envelhecimento.", llm)

    assert "tables" in result["sources"]
