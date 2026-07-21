"""
components/status.py — Aba Serviços: cartões de disponibilidade dos serviços RAG.

Cada serviço conhecido (mesmas portas do docker-compose.yml) vira um cartão com
nome, indicador de status, tipo, modelo, documentos e latência. Latência é
medida no probe; campos não expostos pelo `/health` aparecem como "—" (o
frontend nunca inventa valores). Cada cartão tem um botão de teste.
"""
from __future__ import annotations

from html import escape
from time import perf_counter
from typing import Any

import streamlit as st

from components.sidebar import PRESETS
from services.api import RagClient
from utils.icons import icon
from utils.ui import section_header, status_pill

NA = "—"


def _probe(url: str) -> tuple[dict[str, Any] | None, float]:
    """Consulta `/health` e mede a latência (ms) do round-trip."""
    t0 = perf_counter()
    health = RagClient(url).health()
    return health, (perf_counter() - t0) * 1000


def _row(icon_name: str, label: str, value: str) -> str:
    return (
        f'<div class="svc-row"><span class="k"><span class="lu">{icon(icon_name, 14)}</span>'
        f'{escape(label)}</span><span class="v">{escape(value)}</span></div>'
    )


def _card_html(nome: str, health: dict[str, Any] | None, latency_ms: float) -> str:
    """HTML de um cartão de serviço."""
    online = health is not None
    state = "online" if online else "offline"
    tipo = (health or {}).get("rag_type", NA) if online else NA
    lat = f"{latency_ms:.0f} ms" if online else NA

    return f"""
    <div class="svc">
      <div class="svc-top">
        <div class="svc-name"><span class="lu">{icon("layers", 18)}</span>{escape(nome)}</div>
        {status_pill(state)}
      </div>
      <div class="svc-rows">
        {_row("compass", "Tipo", str(tipo))}
        {_row("cpu", "Modelo", NA)}
        {_row("database", "Documentos", NA)}
        {_row("zap", "Latência", lat)}
      </div>
    </div>
    """


def render() -> None:
    """Grade de cartões de todos os serviços conhecidos."""
    st.markdown(section_header("activity", "Serviços"), unsafe_allow_html=True)
    st.caption(
        "Portas padrão do `docker-compose.yml`. Suba tudo com `docker compose up` "
        "ou individualmente com `python main.py --port <porta>`."
    )

    servicos = [(nome, url) for nome, url in PRESETS.items() if url]
    cols = st.columns(2, gap="medium")
    for i, (nome, url) in enumerate(servicos):
        health, latency = _probe(url)
        with cols[i % 2]:
            st.markdown(_card_html(nome, health, latency), unsafe_allow_html=True)
            st.button("Testar conexão", key=f"test_{i}", use_container_width=True)
