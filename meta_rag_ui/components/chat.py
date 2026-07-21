"""
components/chat.py — Área de conversa: entrada da pergunta e histórico.

Entrada tipo ChatGPT: campo grande + botão de envio compacto (envio por botão
ou Enter). O processamento delega ao backend via `RagClient`; erros viram
mensagens no histórico, sem interromper a aplicação.
"""
from __future__ import annotations

import streamlit as st

from components import metrics
from services.api import ApiError, RagClient
from utils.session import add_message, get_messages

_USER_AVATAR = ":material/person:"
_AI_AVATAR = ":material/auto_awesome:"


def render_input(client: RagClient) -> None:
    """Campo de pergunta grande + botão de envio compacto (Enter também envia)."""
    with st.form("form_pergunta", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
        with col_in:
            pergunta = st.text_input(
                "Pergunta",
                placeholder="Pergunte sobre emprego, PIB, indústria, preços…",
                label_visibility="collapsed",
            )
        with col_btn:
            enviado = st.form_submit_button("Enviar", use_container_width=True)

    if enviado and pergunta.strip():
        _processar(client, pergunta.strip())


def _processar(client: RagClient, pergunta: str) -> None:
    """Envia a pergunta ao backend e registra a resposta (ou o erro) no histórico."""
    add_message("user", pergunta)
    try:
        with st.spinner("Analisando as fontes e sintetizando a resposta…"):
            data = client.query(pergunta)
    except ApiError as exc:
        add_message("assistant", exc.message, meta={"error": exc.message})
    except Exception as exc:  # rede/inesperado — nunca derruba a aplicação
        add_message("assistant", f"Erro inesperado: {exc}", meta={"error": str(exc)})
    else:
        add_message("assistant", data.get("answer", "(resposta vazia)"), meta=data)
    st.rerun()


def render_history() -> None:
    """Histórico: resposta em Markdown + métricas + fontes/validação."""
    for msg in get_messages():
        role = msg["role"]
        avatar = _USER_AVATAR if role == "user" else _AI_AVATAR
        with st.chat_message(role, avatar=avatar):
            meta = msg.get("meta") or {}
            if meta.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if role == "assistant" and meta and not meta.get("error"):
                metrics.render_inline(meta)
                metrics.render_details(meta)
