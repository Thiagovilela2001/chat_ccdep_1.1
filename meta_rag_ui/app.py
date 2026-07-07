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

from components import chat, developer_panel, sidebar, status
from services.api import RagClient
from utils.session import init_session, last_assistant_meta

st.set_page_config(page_title="RAG Estatístico SP", page_icon="📊", layout="wide")
init_session()

# ── Barra lateral: conexão + API key + modo desenvolvedor ─────────────────────
with st.sidebar:
    st.title("📊 RAG Estatístico SP")
    base_url = sidebar.render_connection()
    api_key = sidebar.render_api_key()
    dev_mode = sidebar.render_dev_toggle()
    st.divider()

client = RagClient(base_url, api_key=api_key)
health = client.health()

with st.sidebar:
    sidebar.render(health, base_url, last_assistant_meta())

# ── Área principal ────────────────────────────────────────────────────────────
titulo = (health or {}).get("rag_label") or "RAG Estatístico SP"
st.title(f"📊 {titulo}")
st.caption(
    "Perguntas e respostas sobre os Boletins de Conjuntura Paulista (Seade). "
    "Escolha o backend na barra lateral — o Meta RAG seleciona automaticamente "
    "a melhor estratégia de recuperação para cada pergunta."
)

abas = ["💬 Conversa", "🩺 Serviços"]
if dev_mode:
    abas.append("🛠️ Desenvolvedor")
tabs = st.tabs(abas)

with tabs[0]:
    chat.render_input(client)
    chat.render_history()

with tabs[1]:
    status.render()

if dev_mode:
    with tabs[2]:
        developer_panel.render(last_assistant_meta())
