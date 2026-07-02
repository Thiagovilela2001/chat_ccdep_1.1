"""
services/api.py — Cliente HTTP para o backend Meta RAG.

Único ponto de contato do frontend com o backend. O frontend não decide rota,
não interpreta consultas e não acessa índices: apenas envia a pergunta e recebe
o resultado já produzido pela engine selecionada pelo Router.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class ApiError(Exception):
    """Erro de comunicação/HTTP com o backend Meta RAG."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _extract_detail(resp: requests.Response) -> str:
    """Extrai a mensagem de erro do corpo da resposta, com fallback."""
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except ValueError:
        pass
    return resp.text or f"HTTP {resp.status_code}"


@dataclass
class MetaRagClient:
    """Cliente do orquestrador Meta RAG (`/query`, `/route`, `/health`)."""

    base_url: str
    timeout: int = 180

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def health(self) -> dict[str, Any] | None:
        """Estado do orquestrador e dos backends. `None` se inacessível."""
        try:
            resp = requests.get(self._url("/health"), timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def query(self, question: str) -> dict[str, Any]:
        """
        Envia a pergunta a `POST /query` e devolve o resultado completo
        (resposta + rota + telemetria). Lança `ApiError` em falha.
        """
        try:
            resp = requests.post(
                self._url("/query"), json={"question": question}, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ApiError(f"Falha de conexão com {self.base_url}: {exc}") from exc

        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status=resp.status_code)

        data = resp.json()
        # Round-trip medido no cliente — fallback caso o backend não informe.
        data.setdefault("_client_roundtrip_ms", round(resp.elapsed.total_seconds() * 1000, 1))
        return data

    def route(self, question: str) -> dict[str, Any]:
        """Apenas a decisão de roteamento (`POST /route`), sem executar engine."""
        try:
            resp = requests.post(
                self._url("/route"), json={"question": question}, timeout=30
            )
        except requests.RequestException as exc:
            raise ApiError(f"Falha de conexão com {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status=resp.status_code)
        return resp.json()
