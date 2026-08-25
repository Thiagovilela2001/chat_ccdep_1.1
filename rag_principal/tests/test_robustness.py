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


def test_servico_compartilhado_bloqueia_resposta_com_numero_nao_verificado(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.9,
        get_content=lambda: "A taxa foi 7,9%.",
    )

    class Engine:
        calls = 0

        async def answer(self, **_kwargs):
            self.calls += 1
            return "A taxa foi 7,9%, mas o saldo chegou a 99.", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    popup_called = False

    async def popup_explanations(_citations):
        nonlocal popup_called
        popup_called = True
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        popup_explanations,
    )
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Qual foi a taxa e o saldo?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert "evidência suficiente" in response.answer
    assert response.validation.verified == 1
    assert response.validation.total == 2
    assert response.validation.unverified == ["99"]
    assert response.numeric_citations == []
    assert not popup_called
    assert response.sources_used == ["text"]
    assert diagnostics.verified == 1
    assert diagnostics.total == 2


def test_servico_compartilhado_recupera_numeros_com_busca_ampliada(monkeypatch):
    partial_node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.8,
        get_content=lambda: "A taxa foi 7,9%.",
    )
    complete_node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 3},
        score=0.95,
        get_content=lambda: "A taxa foi 7,9% e o saldo chegou a 99.",
    )

    class Engine:
        calls = 0

        async def answer(self, **kwargs):
            self.calls += 1
            if "graph" in kwargs["sources"]:
                return "A taxa foi 7,9%, e o saldo chegou a 99.", [complete_node]
            return "A taxa foi 7,9%, e o saldo chegou a 99.", [partial_node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    async def popup_explanations(_citations):
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        popup_explanations,
    )
    engine = Engine()
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Qual foi a taxa e o saldo?",
        engine=engine,
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert engine.calls == 2
    assert response.answer == "A taxa foi 7,9%, e o saldo chegou a 99."
    assert response.validation.verified == 2
    assert response.validation.total == 2
    assert response.validation.unverified == []
    assert response.sources_used == ["text", "tables", "timeseries", "graph"]
    assert diagnostics.verified == 2


def test_servico_compartilhado_bloqueia_argumento_textual_sem_suporte():
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 4},
        score=0.9,
        get_content=lambda: "As admissoes superaram os desligamentos na industria.",
    )

    class Engine:
        calls = 0

        async def answer(self, **_kwargs):
            self.calls += 1
            return (
                "A composicao setorial reforca recuperacao parcial apos retracao "
                "de fim de ano.",
                [node],
            )

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    engine = Engine()
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Como foi a composicao setorial?",
        engine=engine,
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert engine.calls == 2
    assert "evidência suficiente" in response.answer
    assert diagnostics.unsupported_arguments
    assert response.numeric_citations == []
