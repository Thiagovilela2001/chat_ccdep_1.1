import asyncio
from types import SimpleNamespace

from rag_principal.src.analysis_engine import AnalysisEngine
from rag_agentic.src.agent_engine import _critic_rounds, _max_iterations, _max_tool_calls
from rag_selfrag.src.self_rag_engine import _max_retries
from rag_core.query_service import execute_engine_query


class _Retriever:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def retrieve(self, _query):
        if self.error:
            raise self.error
        return self.result


class _LLM:
    def complete(self, _prompt):
        return SimpleNamespace(text="Resposta parcial baseada na tabela.")


def test_analysis_engine_tolera_falha_parcial_de_retriever():
    node = SimpleNamespace(get_content=lambda: "Fonte: tabela.csv\nValor: 10")
    engine = AnalysisEngine(
        _Retriever(error=RuntimeError("texto indisponível")),
        _Retriever(result=("Valor: 10", [node])),
        _Retriever(result=None),
        _LLM(),
    )
    answer, sources = asyncio.run(engine.answer(
        question="Qual o valor?",
        sources=["text", "tables", "timeseries"],
        rewritten_query="valor",
    ))
    assert answer == "Resposta parcial baseada na tabela."
    assert sources == [node]


def test_limites_de_iteracao_sao_saturados(monkeypatch):
    monkeypatch.setenv("RAG_AGENTIC_MAX_ITERATIONS", "999")
    monkeypatch.setenv("RAG_AGENTIC_MAX_TOOL_CALLS", "999")
    monkeypatch.setenv("RAG_AGENTIC_CRITIC_ROUNDS", "999")
    monkeypatch.setenv("RAG_SELFRAG_MAX_RETRIES", "999")
    assert _max_iterations() == 12
    assert _max_tool_calls() == 32
    assert _critic_rounds() == 3
    assert _max_retries() == 2


def test_servico_compartilhado_preserva_contrato_http(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.9,
        get_content=lambda: "A taxa foi 7,9%.",
    )

    class Engine:
        async def answer(self, **_kwargs):
            return "A taxa foi 7,9% (Fonte: boletim.pdf, p. 2).", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    async def popup_explanations(_citations):
        return {0: "A taxa informada foi de 7,9%."}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        popup_explanations,
    )
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Qual foi a taxa?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))
    assert response.sources[0].file == "boletim.pdf"
    assert response.sources[0].page == 2
    assert response.sources[0].excerpt == "A taxa foi 7,9%."
    assert response.numeric_citations[0].value == "7,9%"
    assert response.numeric_citations[0].file == "boletim.pdf"
    assert response.numeric_citations[0].page == 2
    assert response.numeric_citations[0].snippet
    assert "A taxa foi 7,9%" in response.numeric_citations[0].claim
    assert response.numeric_citations[0].explanation == "A taxa informada foi de 7,9%."
    assert response.answer == "A taxa foi 7,9%."
    assert response.validation.verified == 1
    assert diagnostics.chunks == 1


def test_servico_http_nunca_omite_proveniencia_de_calculo(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "tabela.pdf", "page": 4},
        score=0.9,
        get_content=lambda: "Valores documentados: 10 e 20.",
    )

    class Engine:
        async def answer(self, **_kwargs):
            return "A soma é 10 + 20 = 30.", [node]

    def interpreter(question, _llm):
        return {"sources": ["tables"], "rewritten_query": question}

    async def no_popup(_citations):
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        no_popup,
    )
    response, _diagnostics = asyncio.run(
        execute_engine_query(
            question="Calcule a soma de 10 e 20.",
            engine=Engine(),
            interp_llm=object(),
            interpreter=interpreter,
            rag_type="test",
            rag_label="Test RAG",
        )
    )

    assert "Proveniência obrigatória do cálculo" in response.answer
    assert "não foi criado pela LLM" in response.answer
    assert "Documentos dos quais os valores foram extraídos" in response.answer
    assert "tabela.pdf, p./aba 4" in response.answer
    assert "10 + 20 = 30" in response.answer
    assert response.validation.verified == response.validation.total == 3


def test_servico_http_converte_ambiguidade_em_pedido_de_direcionamento(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "demografia.pdf", "page": 8},
        score=0.9,
        get_content=lambda: "Índices disponíveis: 43,3 e 49,0.",
    )

    class Engine:
        async def answer(self, **_kwargs):
            return (
                "A diferença seria 49,0 - 43,3 = 5,7, mas os documentos não "
                "especificam os períodos correspondentes."
            ), [node]

    def interpreter(question, _llm):
        return {"sources": ["tables"], "rewritten_query": question}

    async def no_popup(_citations):
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        no_popup,
    )
    response, _diagnostics = asyncio.run(
        execute_engine_query(
            question="Compare os índices e calcule a diferença absoluta.",
            engine=Engine(),
            interp_llm=object(),
            interpreter=interpreter,
            rag_type="test",
            rag_label="Test RAG",
        )
    )

    assert response.answer.startswith("Quais períodos você deseja comparar?")
    assert "5,7" not in response.answer
    assert response.validation.total == 0
