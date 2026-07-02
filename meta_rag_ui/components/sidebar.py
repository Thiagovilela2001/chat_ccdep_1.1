"""
components/sidebar.py — Barra lateral: conexão, informações do sistema,
engines registradas e métricas da última resposta.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from components import metrics
from utils.session import clear_messages

_PRESETS = {
    "Meta RAG (local :8010)": "http://localhost:8010",
    "Outro (URL customizada)": "",
}


def render_connection() -> str:
    """Controles de conexão. Retorna a base_url escolhida."""
    st.subheader("🔌 Conexão")
    nome = st.selectbox("Backend", list(_PRESETS.keys()), label_visibility="collapsed")
    url = _PRESETS[nome]
    if not url:
        url = st.text_input("URL do backend", value="http://localhost:8010")
    return url.rstrip("/")


def render_dev_toggle() -> bool:
    """Botão do Modo Desenvolvedor. Persiste em session_state."""
    dev = st.toggle("🛠️ Modo desenvolvedor", value=st.session_state.get("dev_mode", False))
    st.session_state.dev_mode = dev
    return dev


def render_system_info(health: dict[str, Any] | None, base_url: str) -> None:
    """Estado do orquestrador e conexão."""
    st.subheader("ℹ️ Sistema")
    if health is None:
        st.error(f"🔴 Sem conexão com {base_url}")
        st.caption("Suba o orquestrador: `cd rag_orchestrator && python main.py --port 8010`")
        return
    st.success(f"🟢 {health.get('rag_label', 'Meta RAG')}")
    st.caption(f"Tipo: `{health.get('rag_type', '?')}`  ·  Pronto: {health.get('orchestrator_ready')}")


def render_engines(health: dict[str, Any] | None) -> None:
    """Lista das engines registradas e sua disponibilidade."""
    st.subheader("🧩 Engines registradas")
    backends = (health or {}).get("backends") or {}
    if not backends:
        st.caption("Nenhuma engine reportada pelo backend.")
        return
    for key, info in backends.items():
        up = bool(info.get("up"))
        icon = "🟢" if up else "⚪"
        estado = "no ar" if up else "offline"
        st.markdown(f"{icon} **{key}** — {estado}")


def render(health: dict[str, Any] | None, base_url: str, last_meta: dict[str, Any] | None) -> None:
    """Monta a seção informativa da sidebar (após a conexão já resolvida)."""
    render_system_info(health, base_url)
    st.divider()
    render_engines(health)
    st.divider()
    metrics.render_sidebar(last_meta)
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        clear_messages()
        st.rerun()
