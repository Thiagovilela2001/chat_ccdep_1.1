"""
registry.py — Registro declarativo das estratégias de recuperação.

Cada engine RAG existente é tratada como um **backend intercambiável**, descrito
por um `StrategyProfile` (características, pontos fortes, limitações e tipos de
consulta em que rende melhor). O orquestrador nunca altera a lógica das engines:
apenas as seleciona e encaminha a consulta para o `/query` da escolhida.

Adicionar uma nova estratégia (princípio 4 — extensibilidade) exige **apenas**
acrescentar uma entrada em `STRATEGIES`. Nenhuma outra parte do sistema muda.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field

import requests
from rag_ccdep.core.llm import main_model
from rag_ccdep.core.runtime import bounded_float, bounded_int

# ── Endereços dos backends locais ─────────────────────────────────────────────
# Sobrescrevíveis por variável de ambiente RAG_<CHAVE>_URL.

_DEFAULT_URLS = {
    "principal": "http://localhost:8000",
    "agentic": "http://localhost:8001",
    "raptor": "http://localhost:8002",
    "selfrag": "http://localhost:8003",
}


def _url(key: str) -> str:
    return os.getenv(f"RAG_{key.upper()}_URL", _DEFAULT_URLS[key]).rstrip("/")


# ── Perfil declarativo de uma estratégia ──────────────────────────────────────

@dataclass(frozen=True)
class StrategyProfile:
    key: str
    label: str
    base_url: str
    description: str                 # característica geral
    strengths: tuple[str, ...]       # pontos fortes
    limitations: tuple[str, ...]     # limitações (princípio 3)
    good_for: tuple[str, ...]        # query_type(s) em que rende melhor
    priority: tuple[str, ...]        # "precisao" | "abrangencia"
    retrieval: tuple[str, ...]       # "lexical" | "semantica" | "hibrida"
    complexity: tuple[str, ...] = () # níveis de complexidade que suporta bem
    prototypes: tuple[str, ...] = () # frases-protótipo p/ rota rápida por embeddings
    llm_model: str = field(default_factory=main_model)  # LLM de geração da engine
    embed_model: str = "BAAI/bge-m3"         # modelo de embeddings da engine


# ── Registro das estratégias (adicionar estratégia = 1 entrada aqui) ──────────

STRATEGIES: dict[str, StrategyProfile] = {
    "principal": StrategyProfile(
        key="principal",
        label="RAG Principal (híbrido + grafo opcional)",
        base_url=_url("principal"),
        description=(
            "Híbrido Vector+BM25 + retrievers de tabela/série; "
            "grafo de conhecimento quando RAG_USE_GRAPH=1."
        ),
        strengths=("fatos pontuais", "números", "tabelas", "séries temporais", "relações entre entidades"),
        limitations=("síntese ampla de muitos trechos",),
        good_for=("pontual", "tabular", "temporal", "relacional"),
        priority=("precisao",),
        retrieval=("lexical", "semantica", "hibrida"),
        complexity=("baixa", "media"),
        prototypes=(
            "Qual foi a taxa de desocupação no 3º trimestre de 2023?",
            "Quantos empregos formais foram criados na indústria em 2024?",
            "Mostre a série do PIB paulista entre 2020 e 2024.",
        ),
    ),
    "raptor": StrategyProfile(
        key="raptor",
        label="RAG RAPTOR (hierárquico)",
        base_url=_url("raptor"),
        description="Índice hierárquico com folhas e resumos automáticos multinível.",
        strengths=("perguntas amplas", "comparações de período", "panoramas temáticos"),
        limitations=("detalhe numérico fino",),
        good_for=("ampla", "comparativo"),
        priority=("abrangencia",),
        retrieval=("semantica",),
        complexity=("baixa", "media"),
        prototypes=(
            "Faça um panorama da economia paulista entre 2020 e 2024.",
            "Resuma a evolução da indústria de transformação no período.",
        ),
    ),
    "agentic": StrategyProfile(
        key="agentic",
        label="RAG Agentic (function calling)",
        base_url=_url("agentic"),
        description="Loop iterativo com function calling; decompõe a pergunta em buscas encadeadas.",
        strengths=("multi-hop", "decomposição de perguntas encadeadas"),
        limitations=("latência/custo altos", "excessivo para lookup simples"),
        good_for=("multi_hop",),
        priority=("precisao",),
        retrieval=("hibrida",),
        complexity=("alta",),
        prototypes=(
            "Como a variação do PIB da indústria se relacionou com o emprego formal no setor?",
        ),
    ),
    "selfrag": StrategyProfile(
        key="selfrag",
        label="RAG Self-RAG (self-reflective)",
        base_url=_url("selfrag"),
        description="Self-reflective (RETRIEVE?→ISREL→GENERATE→ISSUP→RETRY); verifica suporte.",
        strengths=("alta fidelidade", "redução de alucinação", "verificação de suporte"),
        limitations=("latência (críticas via LLM)",),
        good_for=("verificacao",),
        priority=("precisao",),
        retrieval=("semantica",),
        complexity=("media", "alta"),
        prototypes=(
            "O desemprego caiu ou subiu no último boletim?",
        ),
    ),
    # ── Nova estratégia entra aqui, e só aqui. ──
    # "longcontext": StrategyProfile(key="longcontext", ...),
}


def get_profiles() -> dict[str, StrategyProfile]:
    """Retorna o registro de estratégias (fonte única de verdade do roteador)."""
    return STRATEGIES


def profile_dict(key: str) -> dict | None:
    """Serializa o EngineProfile de uma estratégia (consumido pelo frontend)."""
    p = STRATEGIES.get(key)
    if p is None:
        return None
    return {
        "key": p.key,
        "label": p.label,
        "description": p.description,
        "strengths": list(p.strengths),
        "limitations": list(p.limitations),
        "good_for": list(p.good_for),
        "priority": list(p.priority),
        "retrieval": list(p.retrieval),
        "llm_model": p.llm_model,
        "embed_model": p.embed_model,
    }


def health_is_ready(payload: dict | None) -> bool:
    """Interpreta contratos de health atuais sem exigir novos campos."""
    if not isinstance(payload, dict):
        return False
    if payload.get("status", "ok") != "ok":
        return False
    return payload.get("engine_ready", True) is not False


# ── Cliente HTTP para uma engine (não importa transformers/HuggingFace) ───────

class EngineClient:
    """
    Encaminha consultas ao `/query` de uma engine já no ar. Usa `requests` em
    thread para não bloquear o event loop do orquestrador.
    """

    def __init__(self, profile: StrategyProfile, timeout: int = 180):
        self.profile = profile
        self.timeout = timeout
        self._failures = 0
        self._opened_at: float | None = None
        self._circuit_lock = threading.Lock()

    def _circuit_is_open(self) -> bool:
        threshold = bounded_int("RAG_CIRCUIT_FAILURE_THRESHOLD", 3, 1, 20)
        recovery = bounded_float("RAG_CIRCUIT_RECOVERY_SECONDS", 30.0, 1.0, 300.0)
        with self._circuit_lock:
            if self._failures < threshold or self._opened_at is None:
                return False
            return time.monotonic() - self._opened_at < recovery

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._failures = 0
            self._opened_at = None

    def _record_failure(self) -> None:
        threshold = bounded_int("RAG_CIRCUIT_FAILURE_THRESHOLD", 3, 1, 20)
        with self._circuit_lock:
            self._failures += 1
            if self._failures >= threshold:
                self._opened_at = time.monotonic()

    @staticmethod
    def _headers() -> dict[str, str]:
        """Credencial serviço-a-serviço; usa a chave pública como fallback."""
        key = os.getenv("RAG_BACKEND_API_KEY") or os.getenv("RAG_API_KEY")
        return {"x-api-key": key} if key else {}

    async def health(self) -> dict | None:
        if self._circuit_is_open():
            return None

        def _do():
            try:
                r = requests.get(
                    f"{self.profile.base_url}/health",
                    headers=self._headers(),
                    timeout=5,
                )
                r.raise_for_status()
                payload = r.json()
                if not isinstance(payload, dict):
                    raise ValueError("Resposta de health não é um objeto JSON.")
                self._record_success()
                return payload
            except (requests.RequestException, ValueError):
                self._record_failure()
                return None
        return await asyncio.to_thread(_do)

    async def query(self, question: str) -> dict:
        if self._circuit_is_open():
            raise RuntimeError(f"Circuit breaker aberto para '{self.profile.key}'.")

        def _do():
            try:
                r = requests.post(
                    f"{self.profile.base_url}/query",
                    json={"question": question},
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                r.raise_for_status()
                payload = r.json()
                if not isinstance(payload, dict):
                    raise ValueError("Resposta da engine não é um objeto JSON.")
                self._record_success()
                return payload
            except Exception:
                self._record_failure()
                raise
        return await asyncio.to_thread(_do)


_clients: dict[tuple[str, int], EngineClient] = {}


def get_client(key: str, timeout: int = 180) -> EngineClient:
    """Cliente HTTP (cacheado) para a engine `key`."""
    if key not in STRATEGIES:
        raise KeyError(f"Estratégia desconhecida: {key!r}")
    cache_key = (key, timeout)
    if cache_key not in _clients:
        _clients[cache_key] = EngineClient(STRATEGIES[key], timeout=timeout)
    return _clients[cache_key]
