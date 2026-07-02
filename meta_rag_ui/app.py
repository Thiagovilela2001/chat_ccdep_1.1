"""
app.py — Frontend Meta RAG (Streamlit).

Ponto de entrada da interface. Apenas consome o backend (orquestrador) via
`MetaRagClient`: envia a pergunta e apresenta o resultado. Nenhuma lógica de
roteamento, classificação ou recuperação vive aqui.

Executar:  streamlit run meta_rag_ui/app.py
"""
from __future__ import annotations

import streamlit as st

from components import chat, developer_panel, sidebar
from services.api import MetaRagClient
from utils.session import init_session, last_assistant_meta

st.set_page_config(page_title="Meta RAG", page_icon="🧠", layout="wide")
init_session()

# ── Barra lateral: conexão + modo desenvolvedor (resolvidos primeiro) ─────────
with st.sidebar:
    st.title("🧠 Meta RAG")
    base_url = sidebar.render_connection()
    dev_mode = sidebar.render_dev_toggle()
    st.divider()

client = MetaRagClient(base_url)
health = client.health()

with st.sidebar:
    sidebar.render(health, base_url, last_assistant_meta())

# ── Área principal ────────────────────────────────────────────────────────────
st.title("🧠 Meta RAG — Conjuntura Paulista")
st.caption(
    "Faça uma pergunta: o orquestrador analisa a consulta, escolhe a melhor "
    "estratégia de recuperação e responde com base nos Boletins de Conjuntura."
)

if dev_mode:
    aba_chat, aba_dev = st.tabs(["💬 Conversa", "🛠️ Desenvolvedor"])
    with aba_chat:
        chat.render_input(client)
        chat.render_history()
    with aba_dev:
        developer_panel.render(last_assistant_meta())
else:
    chat.render_input(client)
    chat.render_history()
