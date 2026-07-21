from types import SimpleNamespace

import pytest

from meta_rag_ui.services.api import ApiError, RagClient


class _Response:
    status_code = 200
    elapsed = SimpleNamespace(total_seconds=lambda: 0.01)
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_query_rejeita_contrato_nao_objeto(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *_args, **_kwargs: _Response([]))
    with pytest.raises(ApiError, match="contrato inválido"):
        RagClient("http://local").query("pergunta")


def test_query_aceita_objeto_e_anexa_latencia(monkeypatch):
    monkeypatch.setattr(
        "requests.post", lambda *_args, **_kwargs: _Response({"answer": "ok"})
    )
    result = RagClient("http://local").query("pergunta")
    assert result["answer"] == "ok"
    assert result["_client_roundtrip_ms"] == 10.0
