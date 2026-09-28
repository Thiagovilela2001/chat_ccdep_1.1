"""
jev.py — Julgamentos semânticos tipados via TypeSafe Jev (System One).

Jev não gera texto: recebe um **estado** e perguntas fechadas — `Choice`,
`Score` e `Noul` — e devolve valores tipados, distribuição de probabilidade e
`confidence` (exceto Noul, cuja própria probabilidade descreve a incerteza).
Este módulo é a única fronteira HTTP com a API da TypeSafe; o restante do
projeto consome apenas `JevResponse`.

Operação:
- `jev_enabled()` é False sem `TYPESAFE_API_KEY` ou com `RAG_JEV_ENABLED=0`;
  nesse caso os chamadores seguem o caminho LLM original, sem mudança.
- Nenhuma exceção de rede/HTTP escapa de `system_one`/`asystem_one`: falha
  devolve `None` e o chamador decide o fallback. A Jev nunca derruba o RAG.
- Todas as perguntas de um pedido são avaliadas em paralelo e de forma
  independente, contra o mesmo estado: peça tudo o que precisar em uma chamada.
- Calibração não é correção: probabilidade alta não garante acerto, então quem
  consome decide limiar (`RAG_JEV_THRESHOLD`) e escalonamento.

Contrato da API: https://docs.typesafe.ai/api (POST /v1/systemone).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from .logger import get_logger
from .metrics import record_reported_usage
from .runtime import bounded_float, bounded_int

log = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
ENDPOINT_PATH = "/v1/systemone"

_FALSY = {"0", "false", "no", "off", ""}
# 429/5xx e afins: vale repetir a chamada; 4xx de contrato não (ver docs/api).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class JevError(RuntimeError):
    """Falha de transporte ou de contrato na chamada à API da TypeSafe."""


# ── Configuração ──────────────────────────────────────────────────────────────

def _api_key() -> str:
    """
    Chave da TypeSafe.

    `TYPESAFE_API_KEY` é o nome oficial (lido também pelo SDK da TypeSafe);
    `JEV_API_KEY` é aceito como alias do projeto. Se ambos existirem, o oficial
    prevalece.
    """
    return (os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY") or "").strip()


def jev_enabled() -> bool:
    """Jev só entra em operação com chave presente e recurso habilitado."""
    if os.getenv("RAG_JEV_ENABLED", "1").strip().lower() in _FALSY:
        return False
    return bool(_api_key())


def decision_threshold() -> float:
    """Limiar que converte probabilidade de Noul em decisão booleana."""
    return bounded_float("RAG_JEV_THRESHOLD", 0.5, 0.0, 1.0)


def _call_timeout() -> float:
    return bounded_float("RAG_JEV_TIMEOUT", 20.0, 2.0, 120.0)


def _max_attempts() -> int:
    return bounded_int("RAG_JEV_MAX_ATTEMPTS", 2, 1, 4)


def model_name() -> str:
    """Modelo System One configurado (`TYPESAFE_MODEL`, padrão `jev-latest`)."""
    return os.getenv("TYPESAFE_MODEL") or DEFAULT_MODEL


# ── Perguntas (payload tipado) ────────────────────────────────────────────────

def choice(instructions: str, criteria: Mapping[str, str | None]) -> dict:
    """
    Escolha entre opções conhecidas (sem ordem entre elas).

    O valor pode ser `None` quando o nome da opção já se explica — a API aceita
    descrição nula por opção.
    """
    if not criteria:
        raise ValueError("Choice exige ao menos uma opção em `criteria`.")
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {
            str(key): (None if value is None else str(value)) for key, value in criteria.items()
        },
    }


def score(instructions: str, criteria: Sequence[str]) -> dict:
    """Posição em uma escala ordenada e descrita em palavras."""
    levels = [str(level) for level in criteria]
    if not 2 <= len(levels) <= 10:
        raise ValueError("Score aceita de 2 a 10 níveis.")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict:
    """
    Probabilidade de uma proposição sim/não ser verdadeira.

    `criteria` é opcional e, quando usado, descreve o que conta como sim e o que
    conta como não: mapa com as chaves `true` e `false`.
    """
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        if set(criteria) != {"true", "false"}:
            raise ValueError("Noul aceita `criteria` com as chaves 'true' e 'false'.")
        question["criteria"] = {key: str(value) for key, value in criteria.items()}
    return question


# ── Respostas tipadas ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Answer:
    """Resposta tipada de uma pergunta, com a distribuição que a sustenta."""

    id: str
    type: str
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    legend: dict[str, str] = field(default_factory=dict)

    def probability(self, option: str) -> float:
        return float(self.probabilities.get(str(option), 0.0))

    def decisiveness(self) -> float:
        """
        Quão decidida é a resposta, em 0..1. Noul não traz `confidence`, então a
        distância de 0.5 é a medida equivalente (0.5 = empate, 0/1 = certeza).
        """
        if self.type == "noul" and self.noul is not None:
            return min(abs(self.noul - 0.5) * 2.0, 1.0)
        if self.confidence is not None:
            return min(max(float(self.confidence), 0.0), 1.0)
        return 0.0


@dataclass(frozen=True)
class JevResponse:
    """Resposta completa de um pedido System One (todas as perguntas)."""

    model: str = ""
    answers: dict[str, Answer] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JevResponse":
        answers: dict[str, Answer] = {}
        raw_answers = payload.get("answers") if isinstance(payload, Mapping) else None
        if isinstance(raw_answers, Mapping):
            for key, value in raw_answers.items():
                answer = _parse_answer(str(key), value)
                if answer is not None:
                    answers[str(key)] = answer
        usage_raw = payload.get("usage") if isinstance(payload, Mapping) else None
        usage = {}
        if isinstance(usage_raw, Mapping):
            for name in ("input_tokens", "output_tokens"):
                number = _as_float(usage_raw.get(name))
                if number is not None:
                    usage[name] = int(number)
        return cls(
            model=str(payload.get("model") or ""),
            answers=answers,
            usage=usage,
        )

    # ── Leitura tipada ────────────────────────────────────────────────────────

    def answer(self, question_id: str) -> Answer | None:
        return self.answers.get(question_id)

    def choice(
        self,
        question_id: str,
        allowed: Iterable[str] | None = None,
    ) -> str | None:
        """Opção vencedora, validada contra as opções que foram enviadas."""
        answer = self.answers.get(question_id)
        if answer is None or answer.type != "choice" or answer.choice is None:
            return None
        value = str(answer.choice)
        if allowed is not None and value not in set(allowed):
            log.warning("Jev: '%s' devolveu opção fora do contrato: %r", question_id, value)
            return None
        return value

    def noul(self, question_id: str) -> float | None:
        answer = self.answers.get(question_id)
        if answer is None or answer.type != "noul":
            return None
        return answer.noul

    def noul_decision(self, question_id: str) -> bool | None:
        """Converte a probabilidade do Noul em decisão usando `RAG_JEV_THRESHOLD`."""
        probability = self.noul(question_id)
        if probability is None:
            return None
        return probability >= decision_threshold()

    def score(self, question_id: str) -> float | None:
        answer = self.answers.get(question_id)
        if answer is None or answer.type != "score":
            return None
        return answer.score

    def confidence(self, question_id: str) -> float | None:
        answer = self.answers.get(question_id)
        return None if answer is None else answer.confidence

    def decisiveness(self, question_id: str) -> float:
        answer = self.answers.get(question_id)
        return 0.0 if answer is None else answer.decisiveness()

    def weakest_decisiveness(self, question_ids: Iterable[str]) -> float | None:
        """Menor grau de decisão entre as dimensões informadas (None se vazio)."""
        values = [self.decisiveness(key) for key in question_ids if key in self.answers]
        return min(values) if values else None


def _parse_answer(question_id: str, value: Any) -> Answer | None:
    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("type") or "").lower()
    if kind not in {"choice", "score", "noul"}:
        return None
    probabilities: dict[str, float] = {}
    raw_probabilities = value.get("probabilities")
    if isinstance(raw_probabilities, Mapping):
        for key, probability in raw_probabilities.items():
            number = _as_float(probability)
            if number is not None:
                probabilities[str(key)] = min(max(number, 0.0), 1.0)
    legend: dict[str, str] = {}
    raw_legend = value.get("legend")
    if isinstance(raw_legend, Mapping):
        legend = {str(key): str(text) for key, text in raw_legend.items()}
    confidence = _as_float(value.get("confidence"))
    return Answer(
        id=question_id,
        type=kind,
        choice=None if value.get("choice") is None else str(value.get("choice")),
        score=_as_float(value.get("score")),
        noul=_as_float(value.get("noul")),
        confidence=None if confidence is None else min(max(confidence, 0.0), 1.0),
        probabilities=probabilities,
        legend=legend,
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Cliente ───────────────────────────────────────────────────────────────────

class JevClient:
    """
    Cliente síncrono/assíncrono do endpoint System One.

    `transport` existe para testes (ex.: `httpx.MockTransport`); em produção o
    cliente usa o transporte HTTP padrão.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_attempts: int | None = None,
        transport: Any = None,
    ):
        self._api_key = api_key if api_key is not None else _api_key()
        self._base_url = (
            base_url or os.getenv("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = model or model_name()
        self._timeout = _call_timeout() if timeout is None else timeout
        self._max_attempts = _max_attempts() if max_attempts is None else max_attempts
        self._transport = transport
        self._sync_client: Any = None
        self._async_client: Any = None

    # ── Transporte ────────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Modelo System One usado nos pedidos."""
        return self._model

    @property
    def endpoint(self) -> str:
        """URL completa do endpoint System One."""
        return f"{self._base_url}{ENDPOINT_PATH}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, state: Any, questions: Mapping[str, Any]) -> dict:
        if not questions:
            raise ValueError("O pedido System One exige ao menos uma pergunta.")
        return {"state": state, "model": self._model, "questions": dict(questions)}

    def _sync(self):
        if self._sync_client is None:
            import httpx  # import tardio: módulo importável sem httpx instalado

            self._sync_client = httpx.Client(timeout=self._timeout, transport=self._transport)
        return self._sync_client

    def _async(self):
        if self._async_client is None:
            import httpx

            self._async_client = httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            )
        return self._async_client

    # ── Chamada ───────────────────────────────────────────────────────────────

    def system_one(self, state: Any, questions: Mapping[str, Any]) -> JevResponse | None:
        """Executa um pedido; devolve `None` (e loga) quando a Jev não responde."""
        if not self._api_key:
            log.warning("Jev: TYPESAFE_API_KEY ausente — chamada ignorada.")
            return None
        payload = self._payload(state, questions)

        failure: Any = None
        for attempt in range(1, self._max_attempts + 1):
            retryable = True
            try:
                response = self._sync().post(self.endpoint, json=payload, headers=self._headers())
                if response.status_code == 200:
                    parsed = _parse_response(response)
                    if parsed is not None:
                        return parsed
                    failure = JevError("HTTP 200 sem respostas utilizáveis")
                else:
                    failure = JevError(f"HTTP {response.status_code}: {response.text[:200]}")
                    retryable = response.status_code in _RETRYABLE_STATUS
            except Exception as exc:  # rede, timeout, DNS, corpo inválido
                failure = exc
            if not retryable or attempt >= self._max_attempts:
                break
            time.sleep(min(0.5 * attempt, 2.0))

        log.warning("Jev indisponível: %s", failure)
        return None

    async def asystem_one(self, state: Any, questions: Mapping[str, Any]) -> JevResponse | None:
        """Versão assíncrona de `system_one`, mesma política de falha."""
        if not self._api_key:
            log.warning("Jev: TYPESAFE_API_KEY ausente — chamada ignorada.")
            return None
        import asyncio

        payload = self._payload(state, questions)
        failure: Any = None
        for attempt in range(1, self._max_attempts + 1):
            retryable = True
            try:
                response = await self._async().post(
                    self.endpoint, json=payload, headers=self._headers()
                )
                if response.status_code == 200:
                    parsed = _parse_response(response)
                    if parsed is not None:
                        return parsed
                    failure = JevError("HTTP 200 sem respostas utilizáveis")
                else:
                    failure = JevError(f"HTTP {response.status_code}: {response.text[:200]}")
                    retryable = response.status_code in _RETRYABLE_STATUS
            except Exception as exc:
                failure = exc
            if not retryable or attempt >= self._max_attempts:
                break
            await asyncio.sleep(min(0.5 * attempt, 2.0))

        log.warning("Jev indisponível: %s", failure)
        return None

def _parse_response(response: Any) -> JevResponse | None:
    """Converte o corpo HTTP em `JevResponse`, tolerando corpo inválido."""
    try:
        payload = response.json()
    except Exception as exc:
        log.warning("Jev: corpo da resposta não é JSON válido (%s)", exc)
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        parsed = JevResponse.from_payload(payload)
    except Exception as exc:  # contrato inesperado: trata como indisponibilidade
        log.warning("Jev: resposta inválida (%s)", exc)
        return None
    if parsed.usage:
        record_reported_usage(
            "jev",
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=parsed.usage.get("input_tokens", 0),
                    output_tokens=parsed.usage.get("output_tokens", 0),
                )
            ),
        )
    if not parsed.answers:
        log.warning("Jev: resposta sem respostas utilizáveis.")
        return None
    return parsed
