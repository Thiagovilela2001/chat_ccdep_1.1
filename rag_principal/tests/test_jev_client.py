"""
Testes do cliente TypeSafe Jev (rag_core/jev.py).

Sobe um servidor HTTP local que fala o contrato do endpoint System One
(POST /v1/systemone) e exercita o caminho real: montagem do pedido, cabeçalhos,
parse tipado, métricas de uso, repetição em falha recuperável e desistência
silenciosa em qualquer indisponibilidade.
"""
import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag_core import jev
from rag_core.jev import JevClient, JevResponse, choice, noul, score
from rag_core.metrics import snapshot

# Resposta de exemplo do quickstart oficial (docs.typesafe.ai).
_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.78,
            "probabilities": {"technical": 0.85, "sales": 0.0, "billing": 0.15},
        },
        "frustration": {
            "type": "score",
            "score": 1.0,
            "confidence": 1.0,
            "legend": {"0": "Calmo", "1": "Frustrado, mas civilizado"},
            "probabilities": {"0": 0.0, "1": 1.0},
        },
        "is_urgent": {"type": "noul", "noul": 1.0},
    },
    "usage": {"input_tokens": 392, "output_tokens": 65},
}


class _FakeTypeSafe:
    """Servidor System One de mentira: registra pedidos e segue um roteiro."""

    def __init__(self, script):
        self.script = list(script)  # [(status, payload|str)] por requisição
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                fake.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "content_type": self.headers.get("Content-Type"),
                        "json": json.loads(raw or b"{}"),
                    }
                )
                status, payload = (
                    fake.script.pop(0) if fake.script else (500, {"error": "roteiro vazio"})
                )
                body = payload if isinstance(payload, str) else json.dumps(payload)
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silencia o log do http.server
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _client(server, **kwargs) -> JevClient:
    return JevClient(api_key="chave-de-teste", base_url=server.base_url, **kwargs)


def _closed_port_base_url() -> str:
    """Endereço que recusa conexão: porta reservada e devolvida ao SO."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _questions() -> dict:
    return {
        "department": choice(
            "Qual equipe deve tratar isto",
            {"billing": "Cobrança", "technical": "Integração", "sales": "Comercial"},
        ),
        "frustration": score("Quão frustrado está o cliente", ["Calmo", "Frustrado"]),
        "is_urgent": noul("A mensagem transmite urgência?"),
    }


# ── Contrato do pedido e da resposta ─────────────────────────────────────────

def test_pedido_segue_o_contrato_do_endpoint():
    with _FakeTypeSafe([(200, _RESPONSE)]) as server:
        client = _client(server)
        client.system_one({"ticket": {"id": "A-104"}}, _questions())

        assert len(server.requests) == 1
        request = server.requests[0]
        assert request["path"] == "/v1/systemone"
        assert request["authorization"] == "Bearer chave-de-teste"
        assert request["content_type"] == "application/json"
        assert request["json"]["model"] == "jev-latest"
        assert request["json"]["state"] == {"ticket": {"id": "A-104"}}
        assert set(request["json"]["questions"]) == {"department", "frustration", "is_urgent"}
        assert request["json"]["questions"]["department"]["criteria"] == {
            "billing": "Cobrança",
            "technical": "Integração",
            "sales": "Comercial",
        }
        assert request["json"]["questions"]["frustration"]["criteria"] == ["Calmo", "Frustrado"]


def test_resposta_tipada_e_lida_com_probabilidades_e_uso():
    with _FakeTypeSafe([(200, _RESPONSE)]) as server:
        response = _client(server).system_one("texto", _questions())

        assert isinstance(response, JevResponse)
        assert response.model == "jev-1.13.0"
        assert response.choice("department") == "technical"
        assert response.confidence("department") == pytest.approx(0.78)
        assert response.answer("department").probability("technical") == pytest.approx(0.85)
        assert response.score("frustration") == pytest.approx(1.0)
        assert response.answer("frustration").legend["1"] == "Frustrado, mas civilizado"
        assert response.noul("is_urgent") == pytest.approx(1.0)
        assert response.noul_decision("is_urgent") is True
        # Noul não traz `confidence`: a decisão vem da própria probabilidade.
        assert response.answer("is_urgent").decisiveness() == pytest.approx(1.0)
        assert snapshot()["usage"]["jev"]["reported_input_tokens"] >= 392
        assert snapshot()["usage"]["jev"]["reported_output_tokens"] >= 65


def test_opcao_fora_das_permitidas_e_descartada():
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "marketing",
                "confidence": 0.9,
                "probabilities": {"marketing": 0.9},
            }
        },
    }
    with _FakeTypeSafe([(200, payload)]) as server:
        response = _client(server).system_one("texto", _questions())

        assert response.choice("department", allowed={"technical", "billing"}) is None
        assert response.choice("department") == "marketing"


# ── Resiliência: nada escapa ao chamador ─────────────────────────────────────

def test_falha_recuperavel_e_repetida_e_depois_aceita():
    with _FakeTypeSafe([(503, {"error": "indisponível"}), (200, _RESPONSE)]) as server:
        response = _client(server, max_attempts=2, timeout=5.0).system_one("x", _questions())

        assert response is not None
        assert len(server.requests) == 2


def test_falha_persistente_devolve_none_sem_excecao():
    with _FakeTypeSafe([(503, {"error": "indisponível"})] * 2) as server:
        response = _client(server, max_attempts=2, timeout=5.0).system_one("x", _questions())

        assert response is None
        assert len(server.requests) == 2


def test_erro_de_contrato_nao_e_repetido():
    with _FakeTypeSafe([(401, {"error": "chave inválida"})] * 3) as server:
        response = _client(server, max_attempts=3, timeout=5.0).system_one("x", _questions())

        assert response is None
        assert len(server.requests) == 1


def test_corpo_invalido_e_tratado_como_indisponibilidade():
    with _FakeTypeSafe([(200, "isto não é json")]) as server:
        response = _client(server, max_attempts=1, timeout=5.0).system_one("x", _questions())

        assert response is None


def test_resposta_sem_respostas_utilizaveis_e_indisponibilidade():
    payload = {"model": "jev-1.13.0", "answers": {"department": {"type": "desconhecido"}}}
    with _FakeTypeSafe([(200, payload)]) as server:
        response = _client(server, max_attempts=1, timeout=5.0).system_one("x", _questions())

        assert response is None


def test_servidor_fora_do_ar_devolve_none():
    client = JevClient(
        api_key="chave", base_url=_closed_port_base_url(), timeout=2.0, max_attempts=1
    )
    assert client.system_one("x", _questions()) is None


def test_sem_chave_nao_faz_requisicao():
    with _FakeTypeSafe([(200, _RESPONSE)]) as server:
        client = JevClient(api_key="", base_url=server.base_url)
        assert client.system_one("x", _questions()) is None
        assert server.requests == []


def test_versao_assincrona_usa_o_mesmo_contrato():
    with _FakeTypeSafe([(200, _RESPONSE)]) as server:
        response = asyncio.run(
            _client(server).asystem_one("x", _questions())
        )

        assert response is not None
        assert response.choice("department") == "technical"
        assert server.requests[0]["path"] == "/v1/systemone"


def test_versao_assincrona_nao_levanta_com_servidor_fora_do_ar():
    client = JevClient(
        api_key="chave", base_url=_closed_port_base_url(), timeout=2.0, max_attempts=1
    )
    assert asyncio.run(client.asystem_one("x", _questions())) is None


# ── Configuração e derivados ─────────────────────────────────────────────────

def test_jev_enabled_exige_chave_e_flag(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "chave")
    monkeypatch.delenv("RAG_JEV_ENABLED", raising=False)
    assert jev.jev_enabled() is True

    monkeypatch.setenv("RAG_JEV_ENABLED", "0")
    assert jev.jev_enabled() is False

    monkeypatch.delenv("RAG_JEV_ENABLED", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    assert jev.jev_enabled() is False


def test_alias_jev_api_key_e_aceito_e_o_nome_oficial_prevalece(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("JEV_API_KEY", "chave-do-alias")
    assert jev.jev_enabled() is True

    with _FakeTypeSafe([(200, _RESPONSE)]) as server:
        client = JevClient(base_url=server.base_url)
        assert client.system_one("x", _questions()) is not None
        assert server.requests[0]["authorization"] == "Bearer chave-do-alias"

    monkeypatch.setenv("TYPESAFE_API_KEY", "chave-oficial")
    assert JevClient(base_url="http://127.0.0.1:1")._api_key == "chave-oficial"  # noqa: SLF001


def test_limiar_do_noul_converte_probabilidade_em_decisao(monkeypatch):
    response = JevResponse.from_payload(
        {"answers": {"x": {"type": "noul", "noul": 0.62}}}
    )
    monkeypatch.delenv("RAG_JEV_THRESHOLD", raising=False)
    assert response.noul_decision("x") is True

    monkeypatch.setenv("RAG_JEV_THRESHOLD", "0.8")
    assert response.noul_decision("x") is False


def test_pergunta_invalida_e_rejeitada_na_construcao():
    with pytest.raises(ValueError):
        choice("sem opções", {})
    with pytest.raises(ValueError):
        score("escala curta demais", ["único nível"])


def test_weakest_decisiveness_usa_o_pior_caso():
    response = JevResponse.from_payload(
        {
            "answers": {
                "a": {"type": "choice", "choice": "x", "confidence": 0.9, "probabilities": {}},
                "b": {"type": "noul", "noul": 0.55},
            }
        }
    )
    assert response.weakest_decisiveness(["a", "b"]) == pytest.approx(0.1, abs=1e-6)
    assert response.weakest_decisiveness(["a"]) == pytest.approx(0.9)
    assert response.weakest_decisiveness([]) is None


def test_descricao_nula_de_opcao_e_preservada():
    question = choice("Qual o tom?", {"calmo": None, "irritado": "Usa linguagem forte"})
    assert question["criteria"] == {"calmo": None, "irritado": "Usa linguagem forte"}


def test_criteria_do_noul_exige_as_chaves_true_e_false():
    assert noul("Há urgência?", {"true": "prazos", "false": "sem prazo"})["criteria"] == {
        "true": "prazos",
        "false": "sem prazo",
    }
    assert "criteria" not in noul("Há urgência?")
    with pytest.raises(ValueError):
        noul("Há urgência?", {"sim": "prazos"})


def test_sobrecarga_529_tambem_e_repetida():
    with _FakeTypeSafe([(529, {"error": "overloaded"}), (200, _RESPONSE)]) as server:
        response = _client(server, max_attempts=2, timeout=5.0).system_one("x", _questions())

        assert response is not None
        assert len(server.requests) == 2
