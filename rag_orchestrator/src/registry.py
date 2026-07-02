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
from dataclasses import dataclass, field

import requests

# ── Endereços dos backends (mesmas portas do docker-compose.yml) ──────────────
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
    llm_model: str = "gpt-5-chat-latest"     # LLM de geração usado pela engine
    embed_model: str = "BAAI/bge-m3"         # modelo de embeddings da engine


# ── Registro das estratégias (adicionar estratégia = 1 entrada aqui) ──────────

STRATEGIES: dict[str, StrategyProfile] = {
    "principal": StrategyProfile(
        key="principal",
        label="RAG Principal (híbrido + grafo)",
        base_url=_url("principal"),
        description="Híbrido Vector+BM25 + retrievers de tabela/série + grafo de conhecimento.",
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


# ── Cliente HTTP para uma engine (não importa transformers/HuggingFace) ───────

class EngineClient:
    """
    Encaminha consultas ao `/query` de uma engine já no ar. Usa `requests` em
    thread para não bloquear o event loop do orquestrador.
    """

    def __init__(self, profile: StrategyProfile, timeout: int = 180):
        self.profile = profile
        self.timeout = timeout

    async def health(self) -> dict | None:
        def _do():
            try:
                r = requests.get(f"{self.profile.base_url}/health", timeout=5)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                return None
        return await asyncio.to_thread(_do)

    async def query(self, question: str) -> dict:
        def _do():
            r = requests.post(
                f"{self.profile.base_url}/query",
                json={"question": question},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        return await asyncio.to_thread(_do)


_clients: dict[str, EngineClient] = {}


def get_client(key: str, timeout: int = 180) -> EngineClient:
    """Cliente HTTP (cacheado) para a engine `key`."""
    if key not in STRATEGIES:
        raise KeyError(f"Estratégia desconhecida: {key!r}")
    if key not in _clients:
        _clients[key] = EngineClient(STRATEGIES[key], timeout=timeout)
    return _clients[key]
