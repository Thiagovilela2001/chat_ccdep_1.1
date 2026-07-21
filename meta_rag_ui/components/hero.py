"""
components/hero.py — Cabeçalho (hero) com ícone, título, descrição e badges.

Os badges mostram métricas REAIS do sistema, com fallback "—" quando ainda não
há dados (nenhuma pergunta feita / backend offline) — nunca inventa valores:
- Documentos   → nº de fontes da última resposta
- Metodologia  → rótulo do backend ativo (health) ou engine da última rota
- Modelo LLM   → modelo de geração informado na última resposta
- Tempo médio  → média de `total_ms` das respostas da sessão
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from utils.formatting import format_ms, safe_get
from utils.icons import icon
from utils.session import get_messages
from utils.ui import badge

NA = "—"


def _avg_latency() -> str:
    times = []
    for msg in get_messages():
        if msg.get("role") != "assistant":
            continue
        meta = msg.get("meta") or {}
        t = safe_get(meta, "timings", "total_ms", default=meta.get("_client_roundtrip_ms"))
        if isinstance(t, (int, float)):
            times.append(t)
    return format_ms(sum(times) / len(times)) if times else NA


def _last_meta() -> dict[str, Any]:
    for msg in reversed(get_messages()):
        if msg.get("role") == "assistant" and msg.get("meta"):
            return msg["meta"]
    return {}


def render(health: dict[str, Any] | None) -> None:
    """Renderiza o hero + a linha de badges."""
    meta = _last_meta()
    route = meta.get("route") or {}

    metodologia = (
        route.get("engine_label")
        or (health or {}).get("rag_label")
        or meta.get("rag_label")
        or NA
    )
    modelo = safe_get(meta, "models", "generation", default=NA) or NA
    docs = meta.get("sources")
    n_docs = str(len(docs)) if isinstance(docs, list) and docs else NA
    tempo = _avg_latency()

    badges = "".join([
        badge("database", "Documentos", n_docs),
        badge("layers", "Metodologia", str(metodologia)),
        badge("cpu", "Modelo LLM", str(modelo)),
        badge("clock", "Tempo médio", tempo),
    ])

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-badge"><span class="lu">{icon("sparkles", 28)}</span></div>
          <h1>RAG Estatístico SP</h1>
          <p>Perguntas e respostas sobre os Boletins de Conjuntura Paulista da
             Fundação Seade. Recuperação aumentada com seleção automática da
             melhor metodologia para cada pergunta.</p>
          <div class="badge-row">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
