"""
Testes de api_security: autenticação por API key e rate limiting.

Exercita as dependências como funções puras (sem HTTP), evitando o TestClient —
que depende de uma combinação específica de starlette/httpx.

Rode a partir de rag_principal/:
    cd rag_principal && python -m pytest tests/test_api_security.py -q
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import rag_core.api_security as sec


def _fake_request(host: str = "1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    sec._hits.clear()
    for var in ("RAG_API_KEY", "RAG_RATE_LIMIT", "RAG_RATE_WINDOW"):
        monkeypatch.delenv(var, raising=False)
    yield
    sec._hits.clear()


# ── API key ───────────────────────────────────────────────────────────────────

def test_sem_api_key_configurada_e_noop():
    # Não deve levantar quando RAG_API_KEY não está definido.
    assert sec.require_api_key(None) is None


def test_api_key_ausente_ou_errada_401(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "segredo")
    with pytest.raises(HTTPException) as e1:
        sec.require_api_key(None)
    assert e1.value.status_code == 401
    with pytest.raises(HTTPException):
        sec.require_api_key("errado")


def test_api_key_correta_passa(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "segredo")
    assert sec.require_api_key("segredo") is None


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_rate_limit_excedido_429(monkeypatch):
    monkeypatch.setenv("RAG_RATE_LIMIT", "2")
    monkeypatch.setenv("RAG_RATE_WINDOW", "60")
    req = _fake_request()
    sec.enforce_rate_limit(req)
    sec.enforce_rate_limit(req)
    with pytest.raises(HTTPException) as exc:
        sec.enforce_rate_limit(req)
    assert exc.value.status_code == 429


def test_rate_limit_por_ip_isolado(monkeypatch):
    monkeypatch.setenv("RAG_RATE_LIMIT", "1")
    sec.enforce_rate_limit(_fake_request("10.0.0.1"))
    # IP diferente não é afetado pelo limite do primeiro.
    sec.enforce_rate_limit(_fake_request("10.0.0.2"))
    with pytest.raises(HTTPException):
        sec.enforce_rate_limit(_fake_request("10.0.0.1"))


def test_rate_limit_zero_desliga(monkeypatch):
    monkeypatch.setenv("RAG_RATE_LIMIT", "0")
    req = _fake_request()
    for _ in range(10):
        sec.enforce_rate_limit(req)  # nunca levanta
