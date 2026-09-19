"""Traduz falhas do provedor de LLM em respostas HTTP claras.

Uma indisponibilidade do provedor chegava ao cliente como erro interno (500)
depois de esgotar todo o prazo da consulta: ``openai.APITimeoutError`` não é
subclasse de ``TimeoutError``, então não caía no mapeamento 504 que já existia.
"""
from __future__ import annotations

PROVIDER_TIMEOUT_DETAIL = "O provedor de LLM não respondeu a tempo. Tente novamente."
PROVIDER_UNAVAILABLE_DETAIL = "Provedor de LLM indisponível no momento. Tente novamente."


def _provider_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """Tipos de timeout e de indisponibilidade do cliente do provedor."""
    try:
        import openai
    except ImportError:  # pragma: no cover - dependência opcional
        return (), ()
    # APITimeoutError herda de APIConnectionError: a ordem de checagem importa.
    return (openai.APITimeoutError,), (openai.APIConnectionError,)


def provider_http_error(exc: BaseException) -> tuple[int, str] | None:
    """Devolve ``(status, mensagem)`` quando a falha vem do provedor de LLM."""
    timeouts, unavailable = _provider_types()
    if timeouts and isinstance(exc, timeouts):
        return 504, PROVIDER_TIMEOUT_DETAIL
    if isinstance(exc, TimeoutError):
        return 504, PROVIDER_TIMEOUT_DETAIL
    if unavailable and isinstance(exc, unavailable):
        return 503, PROVIDER_UNAVAILABLE_DETAIL
    if isinstance(exc, ConnectionError):
        return 503, PROVIDER_UNAVAILABLE_DETAIL
    return None
