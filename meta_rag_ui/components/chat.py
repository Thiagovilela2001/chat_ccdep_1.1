"""
components/chat.py — Área de conversa: entrada da pergunta e histórico.

A entrada usa um `st.form` (envio por botão **ou** Enter). O processamento
delega ao backend via `RagClient`; erros viram mensagens no histórico, sem
interromper a aplicação.
"""
from __future__ import annotations

import streamlit as st

from components import metrics
from services.api import ApiError, RagClient
from utils.session import add_message, get_messages


def render_input(client: RagClient) -> None:
    """Campo de pergunta em destaque + botão 'Perguntar' (Enter também envia)."""
    with st.form("form_pergunta", clear_on_submit=True):
        pergunta = st.text_input(
            "Pergunta",
            placeholder="Pergunte sobre emprego, PIB, indústria, preços…",
            label_visibility="collapsed",
        )
        enviado = st.form_submit_button("🔍 Perguntar", use_container_width=True)

    if enviado and pergunta.strip():
        _processar(client, pergunta.strip())


def _processar(client: RagClient, pergunta: str) -> None:
    """Envia a pergunta ao backend e registra a resposta (ou o erro) no histórico."""
    add_message("user", pergunta)
    try:
        with st.spinner("Recuperando evidências e sintetizando…"):
            data = client.query(pergunta)
    except ApiError as exc:
        add_message("assistant", f"⚠️ {exc.message}", meta={"error": exc.message})
    except Exception as exc:  # rede/inesperado — nunca derruba a aplicação
        add_message("assistant", f"⚠️ Erro inesperado: {exc}", meta={"error": str(exc)})
    else:
        add_message("assistant", data.get("answer", "(resposta vazia)"), meta=data)
    st.rerun()


def render_history() -> None:
    """Histórico da conversa: resposta em Markdown + métricas + fontes/validação."""
    for msg in get_messages():
        with st.chat_message(msg["role"]):
            meta = msg.get("meta") or {}
            if meta.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and meta and not meta.get("error"):
                metrics.render_inline(meta)
                metrics.render_details(meta)
