"""
utils/session.py — Estado da conversa via st.session_state.

Mantém o histórico de mensagens e as preferências da sessão (ex.: modo
desenvolvedor). Cada mensagem é um dict: {role, content, meta?}.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


def init_session() -> None:
    """Inicializa as chaves de sessão (idempotente)."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("dev_mode", False)


def add_message(role: str, content: str, meta: dict[str, Any] | None = None) -> None:
    """Acrescenta uma mensagem ao histórico."""
    st.session_state.messages.append({"role": role, "content": content, "meta": meta})


def get_messages() -> list[dict[str, Any]]:
    """Retorna o histórico completo da conversa."""
    return st.session_state.messages


def clear_messages() -> None:
    """Limpa o histórico da conversa."""
    st.session_state.messages = []


def last_assistant_meta() -> dict[str, Any] | None:
    """Metadados (rota/telemetria) da última resposta do assistente, se houver."""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("meta"):
            return msg["meta"]
    return None
