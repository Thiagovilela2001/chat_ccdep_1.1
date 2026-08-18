import asyncio

import pytest
import requests

from rag_ccdep.orchestrator.registry import EngineClient, StrategyProfile


def _profile():
    return StrategyProfile(
        key="test", label="Test", base_url="http://test",
        description="", strengths=(), limitations=(), good_for=(),
        priority=(), retrieval=(),
    )


def test_circuit_breaker_abre_apos_falhas(monkeypatch):
    monkeypatch.setenv("RAG_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("RAG_CIRCUIT_RECOVERY_SECONDS", "60")
    calls = {"count": 0}

    def fail(*_args, **_kwargs):
        calls["count"] += 1
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "post", fail)
    client = EngineClient(_profile(), timeout=1)
    for _ in range(2):
        with pytest.raises(requests.ConnectionError):
            asyncio.run(client.query("teste"))
    with pytest.raises(RuntimeError, match="Circuit breaker aberto"):
        asyncio.run(client.query("teste"))
    assert calls["count"] == 2


def test_health_negativo_contabiliza_falha(monkeypatch):
    monkeypatch.setenv("RAG_CIRCUIT_FAILURE_THRESHOLD", "1")

    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail)
    client = EngineClient(_profile(), timeout=1)
    assert asyncio.run(client.health()) is None
    assert client._circuit_is_open()
