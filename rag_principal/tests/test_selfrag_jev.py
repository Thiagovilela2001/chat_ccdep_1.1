"""
Testes dos julgamentos tipados (Jev) no Self-RAG.

Verificam que RETRIEVE?, ISREL e ISSUP usam o caminho Jev quando configurado —
incluindo o fan-out de uma pergunta Noul por trecho em uma única requisição — e
que o caminho LLM original continua intacto sem Jev ou quando ela falha.
"""
import asyncio
from types import SimpleNamespace

import pytest

from rag_core.answer_policy import REFUSAL_TEXT
from rag_core.jev import JevResponse
from rag_selfrag.src.self_rag_engine import SelfRAGEngine

_RESPOSTA = "A taxa de desocupação ficou em 7,1% em 2024."


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


def _payload(**answers) -> dict:
    return {"model": "jev-1.13.0", "answers": answers}


def _retrieve(probability: float) -> dict:
    return _payload(precisa_busca={"type": "noul", "noul": probability})


def _isrel(*probabilities: float) -> dict:
    return _payload(
        **{
            f"trecho_{index}": {"type": "noul", "noul": probability}
            for index, probability in enumerate(probabilities)
        }
    )


def _issup(choice_value: str, confidence: float = 0.9) -> dict:
    return _payload(
        suporte={
            "type": "choice",
            "choice": choice_value,
            "confidence": confidence,
            "probabilities": {choice_value: confidence},
        }
    )


def _engine(jev_client, monkeypatch) -> SelfRAGEngine:
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")  # chave fictícia, sem rede
    return SelfRAGEngine(
        text_retriever=object(),
        tables_retriever=object(),
        timeseries_retriever=object(),
        llm=SimpleNamespace(model="modelo-de-teste"),
        jev_client=jev_client,
    )


def _patch_generation(engine, captured: list) -> None:
    async def _generate(question, context, skill_block):
        captured.append(context)
        return _RESPOSTA

    engine._generate = _generate


def _patch_fetch(engine, passages: list) -> None:
    def _fetch(query, source_nodes):
        source_nodes.append(object())
        return list(passages)

    engine._fetch_all = _fetch


def _patch_json_call(engine, result: dict, calls: list) -> None:
    async def _json_call(prompt, model=None):
        calls.append(prompt)
        return result

    engine._json_call = _json_call


# ── RETRIEVE? ────────────────────────────────────────────────────────────────

def test_retrieve_desnecessario_recusa_sem_buscar(monkeypatch):
    client = _FakeJevClient([_retrieve(0.02)])
    engine = _engine(client, monkeypatch)
    buscando = []
    _patch_fetch(engine, buscando)

    answer, nodes = asyncio.run(engine.answer("qual a sua opinião?", [], "qual a sua opinião?"))

    assert answer == REFUSAL_TEXT
    assert nodes == []
    assert len(client.calls) == 1
    assert "precisa_busca" in client.calls[0]["questions"]


def test_falha_da_jev_em_retrieve_usa_o_llm(monkeypatch):
    client = _FakeJevClient([None])
    engine = _engine(client, monkeypatch)
    prompts = []
    _patch_json_call(engine, {"retrieve": False}, prompts)

    answer, _ = asyncio.run(engine.answer("qual a sua opinião?", [], "qual a sua opinião?"))

    assert answer == REFUSAL_TEXT
    assert prompts, "o caminho LLM deveria ter sido acionado"


# ── ISREL ────────────────────────────────────────────────────────────────────

def test_isrel_fan_out_descarta_trechos_irrelevantes(monkeypatch):
    client = _FakeJevClient(
        [_retrieve(0.93), _isrel(0.91, 0.04, 0.88), _issup("full")]
    )
    engine = _engine(client, monkeypatch)
    _patch_fetch(engine, ["trecho A sobre emprego", "trecho B sobre clima", "trecho C sobre renda"])
    captured = []
    _patch_generation(engine, captured)

    answer, nodes = asyncio.run(engine.answer("emprego em 2024", [], "emprego 2024"))

    assert answer == _RESPOSTA
    assert captured, "a geração deveria ter recebido contexto"
    assert "trecho A" in captured[0] and "trecho C" in captured[0]
    assert "trecho B" not in captured[0]
    assert nodes, "os nós usados devem ser devolvidos"
    # Uma única requisição com uma pergunta por trecho, sem contexto cruzado.
    isrel_call = client.calls[1]
    assert set(isrel_call["questions"]) == {"trecho_0", "trecho_1", "trecho_2"}
    assert [item["indice"] for item in isrel_call["state"]["trechos"]] == [0, 1, 2]
    assert isrel_call["state"]["pergunta"] == "emprego em 2024"


def test_isrel_incompleto_cai_para_o_llm(monkeypatch):
    client = _FakeJevClient([_retrieve(0.93), _isrel(0.9, 0.9), _issup("full")])
    engine = _engine(client, monkeypatch)
    _patch_fetch(engine, ["trecho A", "trecho B", "trecho C"])
    captured = []
    _patch_generation(engine, captured)
    prompts = []
    _patch_json_call(engine, {"relevant": [0, 1, 2]}, prompts)

    asyncio.run(engine.answer("emprego em 2024", [], "emprego 2024"))

    assert prompts, "ISREL incompleto deveria delegar ao LLM"
    assert "trecho B" in captured[0]


def test_isrel_sem_trechos_relevantes_mantem_comportamento_anterior(monkeypatch):
    client = _FakeJevClient([_retrieve(0.93), _isrel(0.01, 0.02), _issup("partial")])
    engine = _engine(client, monkeypatch)
    _patch_fetch(engine, ["trecho A", "trecho B"])
    captured = []
    _patch_generation(engine, captured)

    asyncio.run(engine.answer("emprego em 2024", [], "emprego 2024"))

    # Sem nenhum trecho relevante, o motor segue com todos (regra pré-existente).
    assert "trecho A" in captured[0] and "trecho B" in captured[0]


# ── ISSUP ────────────────────────────────────────────────────────────────────

def test_suporte_insuficiente_dispara_retry_e_nova_avaliacao(monkeypatch):
    client = _FakeJevClient(
        [
            _retrieve(0.93),
            _isrel(0.9),
            _issup("none", confidence=0.95),
            _isrel(0.95),
            _issup("full"),
        ]
    )
    engine = _engine(client, monkeypatch)
    _patch_fetch(engine, ["trecho A"])
    captured = []
    _patch_generation(engine, captured)
    prompts = []
    _patch_json_call(engine, {"query": "query refinada"}, prompts)

    answer, _ = asyncio.run(engine.answer("emprego em 2024", [], "emprego 2024"))

    assert answer == _RESPOSTA
    assert len(client.calls) == 5, "retrieve + isrel + issup + retry(isrel/issup)"
    assert prompts, "o refinamento da query continua sendo LLM"
    assert client.calls[2]["questions"]["suporte"]["type"] == "choice"
    assert set(client.calls[2]["questions"]["suporte"]["criteria"]) == {"full", "partial", "none"}
    assert len(captured) == 2, "geração inicial + geração após o retry"


def test_choice_de_suporte_respeita_o_contrato(monkeypatch):
    client = _FakeJevClient(
        [_payload(suporte={"type": "choice", "choice": "inventado", "confidence": 0.9})]
    )
    engine = _engine(client, monkeypatch)
    # O caminho LLM de fallback é stubbado de propósito: a afirmação do teste é que
    # a opção fora do contrato não é aceita e a decisão volta para o LLM — sem
    # depender de o provedor estar alcançável (o default "full" de `_check_support`
    # só aparece quando a chamada falha; com o `.env` carregado por outro teste, o
    # cliente é real e responde de verdade).
    prompts = []
    _patch_json_call(engine, {"support": "full"}, prompts)

    assert asyncio.run(engine._check_support("resposta", "contexto")) == "full"
    assert len(client.calls) == 1, "a Jev foi consultada e devolveu opção fora do contrato"
    assert prompts, "a decisão deveria ter voltado para o LLM"


# ── Caminho LLM preservado ───────────────────────────────────────────────────

def test_sem_jev_o_motor_usa_os_prompts_originais(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    engine = _engine(None, monkeypatch)
    prompts = []
    _patch_json_call(engine, {"retrieve": False}, prompts)

    assert engine._jev is None
    answer, _ = asyncio.run(engine.answer("qual a sua opinião?", [], "qual a sua opinião?"))

    assert answer == REFUSAL_TEXT
    assert len(prompts) == 1 and "busca em documentos" in prompts[0]


def test_isrel_sem_jev_usa_o_caminho_llm(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    engine = _engine(None, monkeypatch)
    prompts = []
    _patch_json_call(engine, {"relevant": [1]}, prompts)

    relevant = asyncio.run(engine._filter_relevant("pergunta", ["A", "B"]))

    assert relevant == ["B"]
    assert prompts and "relevância" in prompts[0]


@pytest.mark.parametrize("probability,expected", [(0.9, True), (0.1, False)])
def test_limiar_do_retrieve(monkeypatch, probability, expected):
    client = _FakeJevClient([_retrieve(probability)])
    engine = _engine(client, monkeypatch)

    assert asyncio.run(engine._decide_retrieve("pergunta")) is expected
