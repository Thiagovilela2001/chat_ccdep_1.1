"""
services/api.py — Cliente HTTP para os backends RAG do projeto.

Único ponto de contato do frontend com os backends. Funciona tanto com o
orquestrador (Meta RAG) quanto com qualquer engine direta (principal, agentic,
raptor, selfrag): todos expõem `POST /query` e `GET /health` com o mesmo
contrato básico. `POST /route` existe apenas no orquestrador.

Se o backend estiver protegido (env `RAG_API_KEY` no servidor), configure a
mesma chave aqui — ela é enviada no header `x-api-key`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class ApiError(Exception):
    """Erro de comunicação/HTTP com o backend RAG."""

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
class RagClient:
    """Cliente de um backend RAG (`/query`, `/health` e, no Meta, `/route`)."""

    base_url: str
    api_key: str = ""
    timeout: int = 180

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def health(self) -> dict[str, Any] | None:
        """Estado do backend (`/health` é sempre aberto). `None` se inacessível."""
        try:
            resp = requests.get(self._url("/health"), timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def query(self, question: str) -> dict[str, Any]:
        """
        Envia a pergunta a `POST /query` e devolve o resultado completo.
        Meta RAG: resposta + rota + telemetria. Engine direta: resposta +
        fontes + validação numérica. Lança `ApiError` em falha.
        """
        try:
            resp = requests.post(
                self._url("/query"),
                json={"question": question},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Falha de conexão com {self.base_url}: {exc}") from exc

        if resp.status_code == 401:
            raise ApiError(
                "Backend exige API key (401). Informe-a na barra lateral.",
                status=401,
            )
        if resp.status_code == 429:
            raise ApiError(
                "Limite de requisições excedido (429). Aguarde alguns instantes.",
                status=429,
            )
        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status=resp.status_code)

        data = resp.json()
        # Round-trip medido no cliente — fallback caso o backend não informe.
        data.setdefault("_client_roundtrip_ms", round(resp.elapsed.total_seconds() * 1000, 1))
        return data

    def route(self, question: str) -> dict[str, Any]:
        """Apenas a decisão de roteamento (`POST /route`, só no Meta RAG)."""
        try:
            resp = requests.post(
                self._url("/route"),
                json={"question": question},
                headers=self._headers(),
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Falha de conexão com {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise ApiError(_extract_detail(resp), status=resp.status_code)
        return resp.json()


# Compatibilidade com código anterior à unificação do frontend.
MetaRagClient = RagClient
