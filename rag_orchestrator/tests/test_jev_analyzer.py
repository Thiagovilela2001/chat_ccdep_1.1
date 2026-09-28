"""
Testes do classificador Jev do orquestrador (JevQueryAnalyzer).

Cobrem o contrato do dict consumido pelo `router`, a agregação de confiança, a
validação das opções devolvidas, a delegação ao analisador LLM quando a Jev não
responde — e a integração real com `route()` (recusa por escopo só com
confiança alta).
"""
import pytest

from rag_core.jev import JevResponse
from rag_orchestrator.src.jev_analyzer import (
    _CHOICE_DIMENSIONS,
    _NOUL_DIMENSIONS,
    JevQueryAnalyzer,
    build_questions,
    build_state,
)
from rag_orchestrator.src.query_analyzer import _DEFAULTS, QueryAnalyzer, make_analyzer
from rag_orchestrator.src.router import route

_FIRM_ANSWERS = {
    "intent": {"type": "choice", "choice": "consulta_dado", "confidence": 0.91},
    "query_type": {"type": "choice", "choice": "temporal", "confidence": 0.88},
    "semantic_domain": {"type": "choice", "choice": "emprego", "confidence": 0.83},
    "specificity": {"type": "choice", "choice": "especifica", "confidence": 0.80},
    "expected_answer": {"type": "choice", "choice": "serie", "confidence": 0.79},
    "priority": {"type": "choice", "choice": "precisao", "confidence": 0.90},
    "retrieval_need": {"type": "choice", "choice": "lexical", "confidence": 0.86},
    "complexity": {"type": "choice", "choice": "media", "confidence": 0.77},
    "is_labor_market": {"type": "noul", "noul": 0.97},
    "in_scope": {"type": "noul", "noul": 0.99},
}

# Menor grau de decisão entre as dimensões que o router consome.
_FIRM_CONFIDENCE = 0.77


def _payload(**overrides) -> dict:
    answers = {key: dict(value) for key, value in _FIRM_ANSWERS.items()}
    answers.update(overrides)
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 4}}


class _FakeJevClient:
    """Cliente Jev de mentira: devolve o roteiro e registra as chamadas."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if not self._responses:
            return None
        item = self._responses.pop(0)
        return None if item is None else JevResponse.from_payload(item)


class _StubFallback:
    """Analisador reserva: registra que foi chamado."""

    def __init__(self):
        self.calls = 0

    def analyze(self, question: str) -> dict:
        self.calls += 1
        return dict(_DEFAULTS, confidence=0.3, reasoning="stub-llm")


# ── Contrato do dict ─────────────────────────────────────────────────────────

def test_classificacao_completa_em_uma_unica_requisicao():
    client = _FakeJevClient([_payload()])
    result = JevQueryAnalyzer(client=client).analyze("taxa de desocupação em 2024")

    assert set(result) == set(_DEFAULTS)
    assert result["intent"] == "consulta_dado"
    assert result["query_type"] == "temporal"
    assert result["semantic_domain"] == "emprego"
    assert result["specificity"] == "especifica"
    assert result["expected_answer"] == "serie"
    assert result["priority"] == "precisao"
    assert result["retrieval_need"] == "lexical"
    assert result["complexity"] == "media"
    assert result["needs_multi_hop"] is False
    assert result["is_labor_market"] is True
    assert result["in_scope"] is True

    assert len(client.calls) == 1
    questions = client.calls[0]["questions"]
    assert set(questions) == set(_CHOICE_DIMENSIONS) | set(_NOUL_DIMENSIONS)
    assert client.calls[0]["state"]["pergunta"] == "taxa de desocupação em 2024"
    # Campos sem equivalente em Jev (não gera texto) ficam nos defaults.
    assert result["entities"] == [] and result["period"] is None


def test_todas_as_dimensoes_viajam_com_criterios():
    questions = build_questions()
    for key in _CHOICE_DIMENSIONS:
        question = questions[key]
        assert question["type"] == "choice"
        assert question["criteria"], key
        # A pergunta completa vai em `instructions`; o id não chega ao modelo.
        assert question["instructions"] != key
        assert "pergunta" in question["instructions"]
        assert question["instructions"].endswith("?")
        assert all(isinstance(text, str) and text for text in question["criteria"].values())
    for key in _NOUL_DIMENSIONS:
        assert questions[key]["type"] == "noul"
        assert questions[key]["instructions"].endswith("?")
        assert "`pergunta`" in questions[key]["instructions"]


def test_dimensoes_derivadas_e_sem_consumidor_nao_sao_perguntadas():
    """
    `needs_multi_hop` duplica a opção `multi_hop` de `query_type` e `technical_terms`
    não tem consumidor: perguntar as duas só produzia hesitação que prendia o mínimo
    da confiança (5 e 3 vezes em 15 perguntas medidas).
    """
    questions = build_questions()

    assert "needs_multi_hop" not in questions
    assert "technical_terms" not in questions
    # As duas dimensões de espectro seguem como Choice: como Score, a mesma
    # pergunta virava resposta graduada e menos decisiva ({0:0.66,...} contra 0.90 no nível).
    assert questions["retrieval_need"]["type"] == "choice"
    assert questions["complexity"]["type"] == "choice"


def test_estado_carrega_pergunta_e_contexto_do_corpus():
    state = build_state("qual o PIB de 2023?")
    assert state["pergunta"] == "qual o PIB de 2023?"
    assert "São Paulo" in state["corpus"]["tema"] or "São Paulo" in state["corpus"]["fonte"]
    assert state["fora_do_escopo"]


# ── Confiança, validação e fallback ──────────────────────────────────────────

def test_confidence_e_o_menor_grau_de_decisao():
    firme = JevQueryAnalyzer(client=_FakeJevClient([_payload()])).analyze("q")
    assert firme["confidence"] == pytest.approx(_FIRM_CONFIDENCE)

    hesitante = JevQueryAnalyzer(
        client=_FakeJevClient(
            [_payload(priority={"type": "choice", "choice": "precisao", "confidence": 0.52})]
        )
    ).analyze("q")
    # Em `Choice`, o grau de decisão é a própria confiança da resposta.
    assert hesitante["confidence"] == pytest.approx(0.52)


def test_dimensao_sem_consumidor_nao_derruba_a_confianca():
    """Só as dimensões que mudam a rota entram no mínimo (a hesitação aqui é inócua)."""
    result = JevQueryAnalyzer(
        client=_FakeJevClient(
            [_payload(expected_answer={"type": "choice", "choice": "serie", "confidence": 0.02})]
        )
    ).analyze("q")

    assert result["expected_answer"] == "serie"
    assert result["confidence"] == pytest.approx(_FIRM_CONFIDENCE)


def test_needs_multi_hop_vem_do_query_type():
    assert JevQueryAnalyzer(client=_FakeJevClient([_payload()])).analyze("q")["needs_multi_hop"] is False

    multi_hop = _payload(query_type={"type": "choice", "choice": "multi_hop", "confidence": 0.9})
    result = JevQueryAnalyzer(client=_FakeJevClient([multi_hop])).analyze("q")

    assert result["query_type"] == "multi_hop"
    assert result["needs_multi_hop"] is True


def test_opcao_fora_do_contrato_cai_no_default():
    result = JevQueryAnalyzer(
        client=_FakeJevClient([_payload(query_type={"type": "choice", "choice": "banana"})])
    ).analyze("q")

    assert result["query_type"] == "pontual"
    assert result["intent"] == "consulta_dado"


def test_resposta_parcial_completa_as_dimensoes_ausentes():
    partial = _payload()
    for key in ("expected_answer", "retrieval_need"):
        partial["answers"].pop(key)

    result = JevQueryAnalyzer(client=_FakeJevClient([partial])).analyze("q")

    assert result["expected_answer"] == _DEFAULTS["expected_answer"]
    assert result["retrieval_need"] == _DEFAULTS["retrieval_need"]
    assert result["query_type"] == "temporal"


def test_falha_da_jev_delega_para_o_analisador_reserva():
    fallback = _StubFallback()
    result = JevQueryAnalyzer(
        client=_FakeJevClient([None]), fallback=fallback
    ).analyze("q")

    assert fallback.calls == 1
    assert result["reasoning"] == "stub-llm"


def test_cliente_que_levanta_excecao_nao_quebra_a_analise():
    class _Explosive:
        def system_one(self, state, questions):
            raise RuntimeError("boom")

    fallback = _StubFallback()
    result = JevQueryAnalyzer(client=_Explosive(), fallback=fallback).analyze("q")

    assert fallback.calls == 1
    assert set(result) == set(_DEFAULTS)


def test_sem_chave_usa_o_caminho_llm(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("RAG_JEV_ENABLED", raising=False)
    fallback = _StubFallback()

    assert JevQueryAnalyzer(fallback=fallback).analyze("q")["reasoning"] == "stub-llm"
    assert fallback.calls == 1


def test_fallback_pode_ser_funcao(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "chave")
    chamadas = []

    def _reserva(question: str) -> dict:
        chamadas.append(question)
        return dict(_DEFAULTS, reasoning="funcao-reserva")

    result = JevQueryAnalyzer(client=_FakeJevClient([None]), fallback=_reserva).analyze("q")

    assert chamadas == ["q"]
    assert result["reasoning"] == "funcao-reserva"


# ── Fábrica e integração com o roteador ──────────────────────────────────────

def test_fabrica_escolhe_pelo_ambiente(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert isinstance(make_analyzer(), QueryAnalyzer)

    monkeypatch.setenv("TYPESAFE_API_KEY", "chave")
    assert isinstance(make_analyzer(), JevQueryAnalyzer)
    assert isinstance(make_analyzer(use_jev=False), QueryAnalyzer)


def test_roteador_recusa_escopo_apenas_com_confianca_alta():
    claro = JevQueryAnalyzer(
        client=_FakeJevClient([_payload(in_scope={"type": "noul", "noul": 0.02})])
    ).analyze("qual a taxa Selic hoje?")
    assert claro["in_scope"] is False
    assert claro["confidence"] >= 0.75
    decision = route(claro)
    assert decision.mode == "refuse" and decision.refused is True

    duvidoso = JevQueryAnalyzer(
        client=_FakeJevClient([_payload(in_scope={"type": "noul", "noul": 0.45})])
    ).analyze("qual a taxa Selic hoje?")
    assert duvidoso["in_scope"] is False
    assert route(duvidoso).mode != "refuse"


def test_modelo_relatado_vem_da_resposta(monkeypatch):
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-latest")
    analyzer = JevQueryAnalyzer(client=_FakeJevClient([_payload()]))
    assert analyzer.model == "jev-latest"

    analyzer.analyze("q")

    assert analyzer.model == "jev-1.13.0", "depois da chamada, vale o modelo servido"


def test_roteamento_normal_continua_funcionando():
    result = JevQueryAnalyzer(client=_FakeJevClient([_payload()])).analyze(
        "evolução do emprego em 2023"
    )
    decision = route(result)

    assert decision.mode == "single_best"
    assert decision.engines
