"""
components/sidebar.py — Barra lateral: escolha do backend, API key,
informações do sistema, engines registradas e métricas da última resposta.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from components import metrics
from utils.session import clear_messages

# Mesmas portas do docker-compose.yml.
PRESETS: dict[str, str] = {
    "🧠 Meta RAG (orquestrador)": "http://localhost:8010",
    "RAG Principal (híbrido + grafo)": "http://localhost:8000",
    "RAG Agentic (function calling)": "http://localhost:8001",
    "RAG RAPTOR (hierárquico)": "http://localhost:8002",
    "RAG Self-RAG (self-reflective)": "http://localhost:8003",
    "Outro (URL customizada)": "",
}


def render_connection() -> str:
    """Controles de conexão. Retorna a base_url escolhida."""
    st.subheader("🔌 Backend")
    nome = st.selectbox("Backend", list(PRESETS.keys()), label_visibility="collapsed")
    url = PRESETS[nome]
    if not url:
        url = st.text_input("URL do backend", value="http://localhost:8010")
    return url.rstrip("/")


def render_api_key() -> str:
    """Campo opcional de API key (enviada como header `x-api-key`)."""
    key = st.text_input(
        "API key (se o backend exigir)",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="Necessária apenas se o servidor foi iniciado com RAG_API_KEY.",
    )
    st.session_state.api_key = key
    return key


def render_dev_toggle() -> bool:
    """Botão do Modo Desenvolvedor. Persiste em session_state."""
    dev = st.toggle("🛠️ Modo desenvolvedor", value=st.session_state.get("dev_mode", False))
    st.session_state.dev_mode = dev
    return dev


def render_system_info(health: dict[str, Any] | None, base_url: str) -> None:
    """Estado do backend selecionado (orquestrador ou engine direta)."""
    st.subheader("ℹ️ Sistema")
    if health is None:
        st.error(f"🔴 Sem conexão com {base_url}")
        st.caption(
            "Suba o serviço correspondente — ex.: "
            "`cd rag_orchestrator && python main.py --port 8010` ou "
            "`docker compose up`."
        )
        return
    st.success(f"🟢 {health.get('rag_label', health.get('rag_type', 'RAG'))}")
    pronto = health.get("orchestrator_ready", health.get("engine_ready"))
    st.caption(f"Tipo: `{health.get('rag_type', '?')}`  ·  Pronto: {pronto}")


def render_engines(health: dict[str, Any] | None) -> None:
    """Engines registradas no orquestrador (visível só quando o backend é o Meta)."""
    backends = (health or {}).get("backends") or {}
    if not backends:
        return
    st.divider()
    st.subheader("🧩 Engines registradas")
    for key, info in backends.items():
        up = bool(info.get("up"))
        icon = "🟢" if up else "⚪"
        estado = "no ar" if up else "offline"
        st.markdown(f"{icon} **{key}** — {estado}")


def render(health: dict[str, Any] | None, base_url: str, last_meta: dict[str, Any] | None) -> None:
    """Monta a seção informativa da sidebar (após a conexão já resolvida)."""
    render_system_info(health, base_url)
    render_engines(health)
    st.divider()
    metrics.render_sidebar(last_meta)
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        clear_messages()
        st.rerun()
