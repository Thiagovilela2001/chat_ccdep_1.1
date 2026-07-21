"""
app.py — Frontend do RAG Estatístico SP (Streamlit).

Interface única para todos os backends do projeto: o orquestrador Meta RAG
(recomendado) ou qualquer engine direta (principal, agentic, raptor, selfrag).
Apenas consome a API REST via `RagClient`: envia a pergunta e apresenta o
resultado. Nenhuma lógica de roteamento, classificação ou recuperação vive aqui.

Executar:  streamlit run meta_rag_ui/app.py
"""
from __future__ import annotations

import streamlit as st

from components import chat, developer_panel, hero, sidebar, status
from services.api import RagClient
from utils import theme
from utils.session import init_session, last_assistant_meta

st.set_page_config(page_title="RAG Estatístico SP", page_icon="◆", layout="wide")
theme.inject()
init_session()

# ── Barra lateral: metodologia + configuração + status + debug ────────────────
with st.sidebar:
    base_url = sidebar.render_connection()
    dev_mode = sidebar.render_config()

client = RagClient(base_url, api_key=st.session_state.get("api_key", ""))
health = client.health()

with st.sidebar:
    sidebar.render_status(health, base_url)
    sidebar.render_debug(dev_mode)

# ── Área principal ────────────────────────────────────────────────────────────
hero.render(health)

abas = ["Conversa", "Serviços"]
if dev_mode:
    abas.append("Desenvolvedor")
tabs = st.tabs(abas)

with tabs[0]:
    chat.render_input(client)
    chat.render_history()

with tabs[1]:
    status.render()

if dev_mode:
    with tabs[2]:
        developer_panel.render(last_assistant_meta())
