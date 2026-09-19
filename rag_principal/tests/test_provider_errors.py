"""Falha do provedor de LLM precisa chegar ao cliente como indisponibilidade.

Antes, ``openai.APITimeoutError`` (que não herda de ``TimeoutError``) caía no
handler genérico e virava HTTP 500, depois de esgotar o prazo da consulta.
"""
import httpx
import openai
import pytest

from rag_core import api_security
from rag_core.provider_errors import (
    PROVIDER_TIMEOUT_DETAIL,
    PROVIDER_UNAVAILABLE_DETAIL,
    provider_http_error,
)


def _request():
    return httpx.Request("POST", "https://chat.maritaca.ai/api/chat/completions")


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    api_security._hits.clear()
    yield
    api_security._hits.clear()


# ── unidade: classificação ────────────────────────────────────────────────────

def test_timeout_do_provedor_vira_504():
    assert provider_http_error(openai.APITimeoutError(request=_request())) == (
        504,
        PROVIDER_TIMEOUT_DETAIL,
    )


def test_queda_de_conexao_vira_503():
    assert provider_http_error(openai.APIConnectionError(request=_request())) == (
        503,
        PROVIDER_UNAVAILABLE_DETAIL,
    )


def test_timeout_nativo_tambem_vira_504():
    assert provider_http_error(TimeoutError("prazo esgotado"))[0] == 504


def test_erro_comum_nao_e_classificado_como_provedor():
    assert provider_http_error(ValueError("bug de codigo")) is None


def test_sobrecarga_do_provedor_nao_e_mascarada_como_indisponivel():
    # 429 é limite de uso, não indisponibilidade: segue como erro interno.
    over_quota = openai.RateLimitError(
        "rate limit", response=httpx.Response(429, request=_request()), body=None
    )
    assert provider_http_error(over_quota) is None


# ── ponta a ponta: o endpoint devolve o status certo ──────────────────────────
#
# O TestClient do FastAPI não funciona neste ambiente (starlette 0.27 com httpx
# 0.28), então a corrotina do endpoint é chamada direto: mesmo contrato, sem HTTP.

def _post_query(monkeypatch, exc):
    import asyncio

    from fastapi import HTTPException

    from rag_core.api_models import QueryRequest
    from rag_principal.src import api

    async def explode(**_kwargs):
        raise exc

    monkeypatch.setattr(api, "_engine", object())
    monkeypatch.setattr(api, "_interp_llm", object())
    monkeypatch.setattr(api, "execute_engine_query", explode)

    request = QueryRequest(question="Qual foi a taxa de desocupacao?")
    try:
        asyncio.run(api.query(request, _rl=None, _auth=None))
    except HTTPException as http_error:
        return http_error
    raise AssertionError("o endpoint deveria ter levantado HTTPException")


@pytest.mark.parametrize(
    "exc, esperado",
    [
        (openai.APITimeoutError(request=_request()), 504),
        (openai.APIConnectionError(request=_request()), 503),
    ],
)
def test_endpoint_traduz_falha_do_provedor(monkeypatch, exc, esperado):
    erro = _post_query(monkeypatch, exc)

    assert erro.status_code == esperado
    assert erro.detail


def test_endpoint_mantem_500_para_erro_interno(monkeypatch):
    erro = _post_query(monkeypatch, ValueError("bug de codigo"))

    assert erro.status_code == 500
