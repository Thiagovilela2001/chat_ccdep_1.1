import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag_core import llm


def test_maritaca_is_the_default_provider(monkeypatch):
    for name in (
        "RAG_LLM_PROVIDER",
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "MARITACA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert llm.provider_name() == "maritaca"
    assert llm.main_model() == "sabia-4"
    assert llm.interp_model() == "sabia-4"
    assert llm._cfg()["base_url"] == "https://chat.maritaca.ai/api"
    assert llm._cfg()["context_window"] == 128000


def test_llm_environment_overrides_are_preserved(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "maritaca")
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-principal")
    monkeypatch.setenv("RAG_INTERP_MODEL", "modelo-interprete")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.invalid/api")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")

    config = llm._cfg()

    assert config["main_model"] == "modelo-principal"
    assert config["interp_model"] == "modelo-interprete"
    assert config["base_url"] == "https://example.invalid/api"
    assert config["api_key"] == "test-key"


def test_ollama_uses_local_defaults_without_api_key(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    for name in (
        "RAG_LLM_MODEL",
        "RAG_INTERP_MODEL",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "OLLAMA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    config = llm._cfg()

    assert config["base_url"] == "http://127.0.0.1:11434/v1"
    assert config["main_model"] == "qwen3:4b-instruct"
    assert config["interp_model"] == "qwen3:4b-instruct"
    assert config["api_key"] == "ollama"
    llm.require_api_key()


def test_ollama_limits_local_concurrency(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("RAG_LLM_CONCURRENCY", raising=False)
    assert llm.llm_concurrency(default=8) == 1

    monkeypatch.setenv("RAG_LLM_CONCURRENCY", "2")
    assert llm.llm_concurrency(default=8) == 2


# ── Modelo da extração estruturada (RAG_EXTRACT_MODEL) ────────────────────────


def test_extract_model_acompanha_o_modelo_de_sintese(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-principal")
    monkeypatch.delenv("RAG_EXTRACT_MODEL", raising=False)

    assert llm.extract_model() == "modelo-principal"
    assert llm._cfg()["extract_model"] == "modelo-principal"


def test_extract_model_aceita_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-principal")
    monkeypatch.setenv("RAG_EXTRACT_MODEL", "modelo-extrair")

    assert llm.extract_model() == "modelo-extrair"


def test_make_llm_roteia_sintese_interpretacao_e_extracao(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-principal")
    monkeypatch.setenv("RAG_INTERP_MODEL", "modelo-interprete")
    monkeypatch.setenv("RAG_EXTRACT_MODEL", "modelo-extrair")

    assert llm.make_llm().model == "modelo-principal"
    assert llm.make_llm(interp=True).model == "modelo-interprete"
    assert llm.make_llm(extract=True).model == "modelo-extrair"
    # model= explícito continua vencendo os atalhos
    assert llm.make_llm(extract=True, model="modelo-explicito").model == "modelo-explicito"


# ── Esforço de raciocínio (RAG_LLM_REASONING_EFFORT) ──────────────────────────
#
# O que importa provar aqui é o caminho até o fio: um provedor compatível com a
# API OpenAI precisa receber `reasoning_effort` no corpo do POST. Por isso o
# servidor de mentira abaixo, no mesmo estilo de test_jev_client.py.


class _FakeOpenAI:
    """Servidor de chat completions de mentira que registra os corpos recebidos."""

    def __init__(self):
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                fake.requests.append({"path": self.path, "json": json.loads(raw or b"{}")})
                payload = {
                    "id": "chatcmpl-teste",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "modelo-de-teste",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
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
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _provedor_local(monkeypatch, server, **env):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_LLM_BASE_URL", server.base_url)
    monkeypatch.setenv("RAG_LLM_API_KEY", "chave-de-teste")
    monkeypatch.setenv("RAG_LLM_MODEL", "modelo-de-teste")
    for name in ("RAG_INTERP_MODEL", "RAG_EXTRACT_MODEL", "RAG_POPUP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_make_llm_rejeita_papeis_conflitantes():
    import pytest
    with pytest.raises(ValueError, match="mutuamente exclusivos"):
        llm.make_llm(interp=True, extract=True)


def test_reasoning_effort_independente_por_papel(monkeypatch):
    with _FakeOpenAI() as server:
        _provedor_local(
            monkeypatch,
            server,
            RAG_LLM_REASONING_EFFORT="high",
            RAG_INTERP_REASONING_EFFORT="none",
            RAG_EXTRACT_REASONING_EFFORT="low",
        )

        llm.make_llm().complete("sintese")
        llm.make_llm(interp=True).complete("interp")
        llm.make_llm(extract=True).complete("extract")
        # Também suporta role explícito
        llm.make_llm(role="interp").complete("interp-role")

    assert [request["json"].get("reasoning_effort") for request in server.requests] == [
        "high",
        "none",
        "low",
        "none",
    ]


def test_reasoning_effort_ausente_nao_vai_no_corpo(monkeypatch):
    with _FakeOpenAI() as server:
        _provedor_local(monkeypatch, server)
        for name in (
            "RAG_LLM_REASONING_EFFORT",
            "RAG_INTERP_REASONING_EFFORT",
            "RAG_EXTRACT_REASONING_EFFORT",
        ):
            monkeypatch.delenv(name, raising=False)

        llm.make_llm().complete("sintese")
        llm.make_llm(interp=True).complete("interp")
        llm.make_llm(extract=True).complete("extract")

    assert server.requests, "o provedor deveria ter recebido as chamadas"
    for req in server.requests:
        assert "reasoning_effort" not in req["json"]


def test_reasoning_effort_invalido_e_ignorado(monkeypatch, caplog):
    with _FakeOpenAI() as server:
        _provedor_local(monkeypatch, server, RAG_INTERP_REASONING_EFFORT="turbo")

        with caplog.at_level("WARNING"):
            llm.make_llm(interp=True).complete("oi")

    assert "reasoning_effort" not in server.requests[-1]["json"]
    assert "RAG_INTERP_REASONING_EFFORT='turbo' ignorado" in caplog.text


def test_reasoning_effort_auto_mantem_o_padrao_do_provedor(monkeypatch):
    for var in (
        "RAG_LLM_REASONING_EFFORT",
        "RAG_INTERP_REASONING_EFFORT",
        "RAG_EXTRACT_REASONING_EFFORT",
    ):
        for value in ("", "auto", "default"):
            monkeypatch.setenv(var, value)
            assert llm._parse_reasoning_effort(var) is None
