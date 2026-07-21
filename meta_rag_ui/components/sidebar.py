"""
components/sidebar.py — Barra lateral em seções (cartões): Metodologia,
Configuração, Status e Debug. Cada seção vive dentro de um `st.container`
com borda; nada de widgets soltos. O estado de conexão é um indicador
elegante (● Online / Offline), não um alerta vermelho.
"""
from __future__ import annotations

import os
from html import escape
from typing import Any

import streamlit as st

from utils.session import clear_messages
from utils.ui import section_header, status_pill

# Portas padrão (docker-compose.yml). Cada URL é sobrescrevível por env para
# funcionar dentro da rede Docker (nomes de serviço) sem alterar o código.
PRESETS: dict[str, str] = {
    "Meta RAG — orquestrador": os.getenv("META_URL", "http://localhost:8010"),
    "RAG Principal — híbrido + grafo": os.getenv("PRINCIPAL_URL", "http://localhost:8000"),
    "RAG Agentic — function calling": os.getenv("AGENTIC_URL", "http://localhost:8001"),
    "RAG RAPTOR — hierárquico": os.getenv("RAPTOR_URL", "http://localhost:8002"),
    "RAG Self-RAG — self-reflective": os.getenv("SELFRAG_URL", "http://localhost:8003"),
}

_ALLOW_CUSTOM_URL = os.getenv("META_RAG_ALLOW_CUSTOM_URL", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
if _ALLOW_CUSTOM_URL:
    PRESETS["Outro — URL customizada"] = ""


def render_connection() -> str:
    """Seção Metodologia: seleção do backend. Retorna a base_url escolhida."""
    with st.container(border=True):
        st.markdown(section_header("layers", "Metodologia"), unsafe_allow_html=True)
        nome = st.selectbox(
            "Metodologia", list(PRESETS.keys()), label_visibility="collapsed"
        )
        url = PRESETS[nome]
        if not url and _ALLOW_CUSTOM_URL:
            url = st.text_input(
                "URL", value="http://localhost:8010",
                label_visibility="collapsed", placeholder="http://localhost:8010",
            )
    return url.rstrip("/")


def render_config() -> bool:
    """Seção Configuração: API key + modo desenvolvedor. Retorna dev_mode."""
    with st.container(border=True):
        st.markdown(section_header("settings", "Configuração"), unsafe_allow_html=True)
        key = st.text_input(
            "API key", value=st.session_state.get("api_key", ""),
            type="password", label_visibility="collapsed",
            placeholder="API key (se o servidor exigir)",
        )
        st.session_state.api_key = key
        dev = st.toggle(
            "Modo desenvolvedor", value=st.session_state.get("dev_mode", False)
        )
        st.session_state.dev_mode = dev
    return dev


def _engines_html(backends: dict[str, Any]) -> str:
    """Mini-lista das engines registradas, cada uma com seu ponto de status."""
    rows = []
    for key, info in backends.items():
        up = bool(info.get("up"))
        color = "var(--success)" if up else "var(--fg-faint)"
        estado = "no ar" if up else "offline"
        rows.append(
            f'<div class="svc-row"><span class="k">'
            f'<span class="status-dot" style="background:{color}"></span>'
            f'{escape(key)}</span><span class="v">{estado}</span></div>'
        )
    return f'<div class="svc-rows" style="margin-top:.7rem">{"".join(rows)}</div>'


def render_status(health: dict[str, Any] | None, base_url: str) -> None:
    """Seção Status: indicador de conexão + engines registradas."""
    with st.container(border=True):
        st.markdown(section_header("activity", "Status"), unsafe_allow_html=True)
        if health is None:
            st.markdown(
                status_pill("offline", sub=f"Sem conexão com {base_url}"),
                unsafe_allow_html=True,
            )
            st.caption("Suba o serviço: `docker compose up` ou `python main.py`.")
            return

        pronto = health.get("orchestrator_ready", health.get("engine_ready"))
        label = health.get("rag_label", health.get("rag_type", "Conectado"))
        sub = f"{health.get('rag_type', '?')} · pronto: {pronto}"
        st.markdown(status_pill("online", label, sub=sub), unsafe_allow_html=True)

        backends = health.get("backends") or {}
        if backends:
            st.markdown(_engines_html(backends), unsafe_allow_html=True)


def render_debug(dev_mode: bool) -> None:
    """Seção Debug: atalho para inspeção + limpar conversa."""
    with st.container(border=True):
        st.markdown(section_header("bug", "Debug"), unsafe_allow_html=True)
        if dev_mode:
            st.caption("Modo desenvolvedor ativo — inspecione o pipeline na aba Desenvolvedor.")
        else:
            st.caption("Ative o Modo desenvolvedor (Configuração) para inspecionar o pipeline.")
        if st.button("Limpar conversa", key="btn_clear", use_container_width=True):
            clear_messages()
            st.rerun()
