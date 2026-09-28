"""
Testes da seleção de fontes do rag_principal via Jev.

Dois níveis: o seletor (`JevSourceSelector`) sozinho, com um cliente Jev de
mentira, e o `interpret_query` completo — onde a decisão da Jev precisa vencer a
lista que o LLM devolveu, sem que a reescrita da consulta mude e sem que nenhuma
falha da Jev chegue ao chamador.
"""
from types import SimpleNamespace

import pytest

from rag_core.jev import JevResponse
from rag_principal.src.jev_interpreter import (
    AVAILABLE_SOURCES,
    JevSourceSelector,
    build_questions,
    build_state,
)
from rag_principal.src.query_interpreter import interpret_query


class _FakeJevClient:
    """Cliente Jev de mentira: devolve o roteiro e registra as chamadas."""

    def __init__(self, responses, error=None):
        self._responses = list(responses)
        self._error = error
        self.calls: list[dict] = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self._error is not None:
            raise self._error
        if not self._responses:
            return None
        item = self._responses.pop(0)
        return None if item is None else JevResponse.from_payload(item)


class _FakeLLM:
    """LLM de mentira que devolve um texto fixo (ou nada, ou uma exceção)."""

    def __init__(self, text=None, error=None):
        self._text = text
        self._error = error
        self.prompts: list[str] = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class _BrokenSelector:
    def select(self, question):
        raise RuntimeError("Jev caiu")


def _payload(**nouls) -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {key: {"type": "noul", "noul": value} for key, value in nouls.items()},
    }


def _llm_json(sources, rewritten="pergunta reescrita") -> _FakeLLM:
    return _FakeLLM(text='{"sources": %s, "rewritten_query": "%s"}' % (sources, rewritten))


@pytest.fixture(autouse=True)
def _sem_chave(monkeypatch):
    """Nenhum teste deve depender da chave real do ambiente."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("RAG_JEV_ENABLED", raising=False)
    monkeypatch.delenv("RAG_JEV_THRESHOLD", raising=False)


# ── JevSourceSelector ─────────────────────────────────────────────────────────

def test_fontes_acima_do_limiar_entram_e_text_sempre():
    client = _FakeJevClient([_payload(tables=0.71, timeseries=0.93, graph=0.12)])
    decision = JevSourceSelector(client=client).select("Qual o PIB do 1º trimestre de 2024?")

    assert decision["sources"] == ["text", "tables", "timeseries"]
    assert decision["probabilities"] == {"tables": 0.71, "timeseries": 0.93, "graph": 0.12}
    # Confiança = pior caso entre as três (0.71 está a 0.21 de 0.5).
    assert decision["confidence"] == pytest.approx(0.42, abs=0.001)
    assert "tables=0.71 (sim)" in decision["reasoning"]
    assert "graph=0.12 (não)" in decision["reasoning"]


def test_limiar_configuravel_muda_o_corte(monkeypatch):
    monkeypatch.setenv("RAG_JEV_THRESHOLD", "0.8")
    client = _FakeJevClient([_payload(tables=0.71, timeseries=0.93, graph=0.30)])
    decision = JevSourceSelector(client=client).select("Evolução do PIB")

    assert decision["sources"] == ["text", "timeseries"]


def test_uma_requisicao_para_todas_as_fontes():
    client = _FakeJevClient([_payload(tables=0.9, timeseries=0.9, graph=0.9)])
    JevSourceSelector(client=client).select("Pergunta")

    assert len(client.calls) == 1
    assert sorted(client.calls[0]["questions"]) == sorted(AVAILABLE_SOURCES)


def test_resposta_incompleta_nao_decide():
    client = _FakeJevClient([_payload(tables=0.9, timeseries=0.9)])
    assert JevSourceSelector(client=client).select("Pergunta") is None


def test_falha_do_cliente_nao_escapa():
    client = _FakeJevClient([], error=RuntimeError("timeout"))
    assert JevSourceSelector(client=client).select("Pergunta") is None


def test_sem_chave_nem_cria_cliente():
    selector = JevSourceSelector()
    assert selector.select("Pergunta") is None
    assert selector._client is None


def test_perguntas_e_estado_referenciam_a_pergunta():
    questions = build_questions()
    assert sorted(questions) == sorted(AVAILABLE_SOURCES)
    for question in questions.values():
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}
        assert "pergunta" in question["instructions"]

    state = build_state("Evolução do PIB paulista")
    assert state["pergunta"] == "Evolução do PIB paulista"


# ── interpret_query ───────────────────────────────────────────────────────────

def test_fontes_da_jev_vencem_a_lista_do_llm():
    selector = JevSourceSelector(
        client=_FakeJevClient([_payload(tables=0.91, timeseries=0.20, graph=0.80)])
    )
    llm = _llm_json('["graph", "tables"]')

    result = interpret_query("Qual o PIB do 1º trimestre?", llm, selector=selector)

    assert result["sources"] == ["text", "tables", "graph"]
    assert result["rewritten_query"] == "pergunta reescrita"
    assert llm.prompts  # a reescrita continua vindo do LLM


def test_sem_jev_o_caminho_antigo_permanece():
    llm = _llm_json('["graph", "tables"]')

    result = interpret_query("Comparação entre setores", llm)

    assert result["sources"] == ["text", "graph", "tables"]
    assert result["rewritten_query"] == "pergunta reescrita"


def test_json_invalido_com_jev_recupera_as_fontes():
    selector = JevSourceSelector(
        client=_FakeJevClient([_payload(tables=0.05, timeseries=0.88, graph=0.10)])
    )
    llm = _FakeLLM(text="Desculpe, não consegui entender a pergunta.")

    result = interpret_query("Evolução da desocupação em 2024", llm, selector=selector)

    assert result["sources"] == ["text", "timeseries"]
    assert result["rewritten_query"] == "Evolução da desocupação em 2024"


def test_json_invalido_sem_jev_degrada_para_text():
    llm = _FakeLLM(text=None)  # AttributeError no .strip()

    result = interpret_query("Pergunta qualquer", llm)

    assert result["sources"] == ["text"]
    assert result["rewritten_query"] == "Pergunta qualquer"


def test_jev_negando_todas_as_fontes_mantem_text():
    selector = JevSourceSelector(
        client=_FakeJevClient([_payload(tables=0.02, timeseries=0.03, graph=0.01)])
    )
    result = interpret_query("O que é conjuntura?", _llm_json('["tables"]'), selector=selector)

    assert result["sources"] == ["text"]


def test_falha_do_seletor_nao_derruba_a_consulta():
    result = interpret_query("Pergunta", _llm_json('["tables"]'), selector=_BrokenSelector())

    assert result["sources"] == ["text", "tables"]


def test_fontes_fora_do_contrato_do_llm_sao_descartadas():
    llm = _FakeLLM(text='{"sources": ["text", "grafo", "tables"], "rewritten_query": "x"}')

    result = interpret_query("Pergunta", llm)

    assert result["sources"] == ["text", "tables"]


def test_is_labor_market_segue_deterministico():
    selector = JevSourceSelector(
        client=_FakeJevClient([_payload(tables=0.90, timeseries=0.90, graph=0.10)])
    )

    result = interpret_query("Qual a taxa de desemprego em 2024?", _llm_json("[]"), selector=selector)

    assert result["is_labor_market"] is True
