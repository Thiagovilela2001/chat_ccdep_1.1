"""
api_security.py — autenticação por API key e rate limiting, sem dependências extras.

Ambos são **opt-in** via variáveis de ambiente, preservando compatibilidade:

    RAG_API_KEY      Se definido, exige o header 'x-api-key' igual em cada
                     requisição a /query. Se ausente/vazio, a autenticação fica
                     desligada (comportamento atual).
    RAG_RATE_LIMIT   Máximo de requisições por IP dentro da janela (default 30).
                     0 ou negativo desliga o rate limiting.
    RAG_RATE_WINDOW  Tamanho da janela deslizante em segundos (default 60).

Ambas as funções são dependências do FastAPI — plugue-as no endpoint com
``Depends(...)``. O rate limiter usa uma janela deslizante em memória por IP;
para múltiplas réplicas do processo, troque por um backend compartilhado (Redis).
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _rate_limit_config() -> tuple[int, float]:
    try:
        limit = int(os.getenv("RAG_RATE_LIMIT", "30"))
    except ValueError:
        limit = 30
    try:
        window = float(os.getenv("RAG_RATE_WINDOW", "60"))
    except ValueError:
        window = 60.0
    return limit, window


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Exige 'x-api-key' se RAG_API_KEY estiver definido; caso contrário, no-op."""
    expected = os.getenv("RAG_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key inválida ou ausente.")


def enforce_rate_limit(request: Request) -> None:
    """Janela deslizante por IP; 429 ao exceder RAG_RATE_LIMIT requisições."""
    limit, window = _rate_limit_config()
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        dq = _hits[client]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Limite de requisições excedido. Tente novamente em instantes.",
            )
        dq.append(now)
