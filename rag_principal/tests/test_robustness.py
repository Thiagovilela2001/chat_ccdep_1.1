import asyncio
from types import SimpleNamespace

from rag_principal.src.analysis_engine import AnalysisEngine
from rag_agentic.src.agent_engine import _critic_rounds, _max_iterations, _max_tool_calls
from rag_selfrag.src.self_rag_engine import _max_retries
from rag_core.answer_policy import REFUSAL_TEXT
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


def test_servico_preserva_numeros_validados_e_nao_validados(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.9,
        get_content=lambda: "A taxa foi 7,9%.",
    )

    class Engine:
        calls = 0

        async def answer(self, **kwargs):
            self.calls += 1
            if "ORIENTAÇÃO OBRIGATÓRIA" in kwargs["question"]:
                return "A taxa foi 7,9%.", [node]
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

    assert response.answer == "A taxa foi 7,9%, mas o saldo chegou a 99."
    assert response.validation.verified == 1
    assert response.validation.total == 2
    assert response.validation.unverified == ["99"]
    assert [citation.value for citation in response.numeric_citations] == ["7,9%"]
    assert popup_called
    assert response.sources_used == ["text"]
    assert diagnostics.verified == 1
    assert diagnostics.total == 2
    assert diagnostics.unverified == ["99"]


def test_servico_responde_regioes_qualitativamente_quando_valores_falham(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "trabalho.pdf", "page": 5},
        score=0.9,
        get_content=lambda: (
            "Campinas e Baixada Santista apresentaram maior dinamismo "
            "no mercado de trabalho."
        ),
    )

    class Engine:
        async def answer(self, **kwargs):
            if "ORIENTAÇÃO OBRIGATÓRIA" in kwargs["question"]:
                return (
                    "Campinas e Baixada Santista apresentaram maior dinamismo "
                    "no mercado de trabalho.",
                    [node],
                )
            return "Campinas liderou com saldo de 9 mil postos.", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": True}

    async def popup_explanations(_citations):
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        popup_explanations,
    )
    response, _ = asyncio.run(execute_engine_query(
        question="Quais regiões paulistas apresentaram maior dinamismo no mercado de trabalho?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert response.answer == "Campinas liderou com saldo de 9 mil postos."
    assert response.validation.total == 1
    assert response.validation.unverified == ["9"]


def test_servico_preserva_resultado_qualitativo_mesmo_com_parafrase_lexical(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "trabalho.pdf", "page": 6},
        score=0.9,
        get_content=lambda: (
            "Campinas e Sorocaba aparecem entre as areas com maiores saldos."
        ),
    )

    class Engine:
        async def answer(self, **kwargs):
            if "ORIENTAÇÃO OBRIGATÓRIA" in kwargs["question"]:
                return (
                    "Campinas e Sorocaba se destacaram como polos de expansao.",
                    [node],
                )
            return "Campinas e Sorocaba criaram 99 mil postos.", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": True}

    response, diagnostics = asyncio.run(execute_engine_query(
        question="Quais regioes apresentaram maior dinamismo?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert response.answer == "Campinas e Sorocaba criaram 99 mil postos."
    assert response.validation.total == 1
    assert response.validation.unverified == ["99"]
    assert diagnostics.unsupported_arguments


def test_servico_recupera_resultado_sem_numeros_quando_retry_recusa(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "trabalho.pdf", "page": 8},
        score=0.9,
        get_content=lambda: (
            "Campinas e Sorocaba aparecem entre as areas com maiores saldos."
        ),
    )

    class Engine:
        async def answer(self, **kwargs):
            if "ORIENTAÇÃO OBRIGATÓRIA" in kwargs["question"]:
                return REFUSAL_TEXT, [node]
            return (
                "O saldo informado foi de 99 mil postos. "
                "Campinas e Sorocaba aparecem entre as areas de maior destaque.",
                [node],
            )

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": True}

    response, _ = asyncio.run(execute_engine_query(
        question="Quais regioes apresentaram maior dinamismo?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert response.answer == (
        "O saldo informado foi de 99 mil postos. "
        "Campinas e Sorocaba aparecem entre as areas de maior destaque."
    )
    assert response.validation.unverified == ["99"]
    assert response.answer != REFUSAL_TEXT


def test_servico_compartilhado_preserva_resposta_parcial_verificada(monkeypatch):
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.9,
        get_content=lambda: "A taxa foi 7,9%. O indice foi 12,5%.",
    )

    class Engine:
        async def answer(self, **_kwargs):
            return "A taxa foi 7,9%. O indice foi 12,5%. O saldo chegou a 99.", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    async def popup_explanations(_citations):
        return {}

    monkeypatch.setattr(
        "rag_core.query_service.generate_popup_explanations",
        popup_explanations,
    )
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Quais foram os indicadores?",
        engine=Engine(),
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert response.answer == "A taxa foi 7,9%. O indice foi 12,5%. O saldo chegou a 99."
    assert response.validation.verified == 2
    assert response.validation.total == 3
    assert response.validation.unverified == ["99"]
    assert [citation.value for citation in response.numeric_citations] == ["7,9%", "12,5%"]
    assert diagnostics.unsupported_arguments == []


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


def test_servico_compartilhado_recupera_quando_resposta_inicial_recusa():
    node = SimpleNamespace(
        metadata={"source_file": "boletim.pdf", "page": 2},
        score=0.9,
        get_content=lambda: "A taxa foi 7,9%.",
    )

    class Engine:
        calls = 0

        async def answer(self, **kwargs):
            self.calls += 1
            if "graph" in kwargs["sources"]:
                return "A taxa foi 7,9%.", [node]
            return REFUSAL_TEXT, [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    engine = Engine()
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Qual foi a taxa?",
        engine=engine,
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert engine.calls == 2
    assert response.answer == "A taxa foi 7,9%."
    assert response.sources_used == ["text", "tables", "timeseries", "graph"]
    assert diagnostics.verified == 1


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

    assert engine.calls == 3
    assert response.answer == REFUSAL_TEXT
    assert diagnostics.unsupported_arguments
    assert response.numeric_citations == []


def test_servico_preserva_parafrase_com_numeros_totalmente_verificados():
    node = SimpleNamespace(
        metadata={"source_file": "demografia.pdf", "page": 11},
        score=0.9,
        get_content=lambda: (
            "Em 2015, a razao de dependencia foi de 39,8 pessoas "
            "para cada 100 individuos."
        ),
    )

    class Engine:
        calls = 0

        async def answer(self, **_kwargs):
            self.calls += 1
            return (
                "A reconfiguracao estrutural culminou em 2015, com 39,8 "
                "dependentes para cada 100 ativos.",
                [node],
            )

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    engine = Engine()
    response, diagnostics = asyncio.run(execute_engine_query(
        question="Como evoluiu a razao de dependencia?",
        engine=engine,
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert engine.calls == 2
    assert response.answer == (
        "A reconfiguracao estrutural culminou em 2015, com 39,8 "
        "dependentes para cada 100 ativos."
    )
    assert response.validation.verified == response.validation.total == 3
    assert diagnostics.unsupported_arguments
    assert len(response.numeric_citations) == 3


def test_pergunta_regional_ampla_exige_todos_os_periodos_disponiveis():
    node = SimpleNamespace(
        metadata={"source_file": "economia_regional.pdf", "page": 3},
        score=0.9,
        get_content=lambda: (
            "No primeiro trimestre, houve avanço em 14 das 20 regiões. "
            "No segundo trimestre, houve avanço em 16 das 20 regiões."
        ),
    )

    class Engine:
        calls = 0
        seen_sources = []
        seen_queries = []

        async def answer(self, **kwargs):
            self.calls += 1
            self.seen_sources.append(kwargs["sources"])
            self.seen_queries.append(kwargs["rewritten_query"])
            if "resposta anterior ficou incompleta" in kwargs["question"]:
                return (
                    "No primeiro trimestre, houve avanço em 14 das 20 regiões. "
                    "No segundo trimestre, houve avanço em 16 das 20 regiões.",
                    [node],
                )
            return "No primeiro trimestre, houve avanço em 14 das 20 regiões.", [node]

    def interpreter(question, _llm):
        return {"sources": ["text"], "rewritten_query": question, "is_labor_market": False}

    engine = Engine()
    response, _ = asyncio.run(execute_engine_query(
        question="Quais regiões paulistas apresentaram maior dinamismo econômico?",
        engine=engine,
        interp_llm=object(),
        interpreter=interpreter,
        rag_type="test",
        rag_label="Test RAG",
    ))

    assert engine.calls == 2
    assert all(sources == ["text", "tables", "timeseries", "graph"] for sources in engine.seen_sources)
    assert all(
        "todos os periodos disponiveis" in query for query in engine.seen_queries
    ), engine.seen_queries
    assert "primeiro trimestre" in response.answer.lower()
    assert "segundo trimestre" in response.answer.lower()
    assert response.validation.verified == response.validation.total == 4
