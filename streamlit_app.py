"""
Chatbox Streamlit para o RAG Estatístico SP.

Cliente HTTP leve para a API REST de qualquer variante RAG (POST /query).
Não importa modelos nem `transformers` — apenas conversa com o backend já no ar.

Uso:
    # 1. Suba uma variante (em outro terminal):
    #    cd rag_principal && python main.py --port 8000
    # 2. Rode o chat:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import requests
import streamlit as st

# ── Configuração das variantes (mesmas portas do docker-compose.yml) ──────────

VARIANTES: dict[str, str] = {
    "🧠 Meta RAG (auto-router)": "http://localhost:8010",
    "RAG Principal (híbrido + grafo)": "http://localhost:8000",
    "RAG Agentic (function calling)": "http://localhost:8001",
    "RAG RAPTOR (hierárquico)": "http://localhost:8002",
    "RAG Self-RAG (self-reflective)": "http://localhost:8003",
    "Outro (URL customizada)": "",
}

REQUEST_TIMEOUT = 180  # síntese LLM pode demorar


# ── Helpers de rede ───────────────────────────────────────────────────────────

def checar_saude(base_url: str) -> dict | None:
    """Retorna o JSON de /health, ou None se o backend não responder."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def perguntar(base_url: str, pergunta: str) -> dict:
    """Envia a pergunta para POST /query e devolve o JSON da resposta."""
    resp = requests.post(
        f"{base_url}/query",
        json={"question": pergunta},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ── Estado ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Estatístico SP — Chat", page_icon="📊", layout="centered")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []  # lista de {"role", "content", "meta"?}


# ── Sidebar: seleção de variante + conexão ────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuração")

    nome_variante = st.selectbox("Variante RAG", list(VARIANTES.keys()))
    base_url = VARIANTES[nome_variante]
    if not base_url:
        base_url = st.text_input("URL da API", value="http://localhost:8000")
    base_url = base_url.rstrip("/")

    saude = checar_saude(base_url)
    if saude is None:
        st.error(f"🔴 Sem conexão com {base_url}")
        st.caption("Suba a variante: `cd rag_principal && python main.py --port 8000`")
    elif saude.get("engine_ready"):
        st.success(f"🟢 Conectado — {saude.get('rag_label', '?')}")
    else:
        st.warning("🟡 Backend no ar, mas ainda inicializando o índice…")

    st.divider()
    mostrar_fontes = st.toggle("Mostrar fontes", value=True)
    mostrar_validacao = st.toggle("Mostrar validação numérica", value=True)

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()


# ── Cabeçalho ─────────────────────────────────────────────────────────────────

st.title("📊 RAG Estatístico SP")
st.caption("Perguntas e respostas sobre os Boletins de Conjuntura Paulista (Seade, 2020–2025).")


# ── Renderização de metadados de uma resposta ─────────────────────────────────

def render_meta(meta: dict) -> None:
    rota = meta.get("route")
    if rota and rota.get("engine"):
        conf = rota.get("confidence")
        conf_txt = f" · confiança {conf:.2f}" if isinstance(conf, (int, float)) else ""
        modo = "múltiplas engines" if rota.get("mode") == "multi" else rota.get("engine")
        st.info(f"🧠 Roteador → **{rota.get('engine_label') or modo}** "
                f"({rota.get('query_type')}{conf_txt})")

    reescrita = meta.get("rewritten_query")
    if reescrita:
        st.caption(f"🔎 Query interpretada: *{reescrita}*  ·  fontes: {', '.join(meta.get('sources_used', [])) or '—'}")

    if mostrar_fontes and meta.get("sources"):
        with st.expander(f"📎 Fontes ({len(meta['sources'])})"):
            for s in meta["sources"]:
                st.markdown(f"- `{s['file']}` — relevância {s['score']:.2f}")

    if mostrar_validacao and meta.get("validation"):
        v = meta["validation"]
        total = v.get("total", 0)
        verif = v.get("verified", 0)
        if total:
            ok = verif == total
            icone = "✅" if ok else "⚠️"
            st.caption(f"{icone} Validação numérica: {verif}/{total} valores conferidos nas fontes")
            if v.get("unverified"):
                st.caption("Não verificados: " + ", ".join(f"`{x}`" for x in v["unverified"]))


# ── Histórico ─────────────────────────────────────────────────────────────────

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_meta(msg["meta"])


# ── Entrada do chat ───────────────────────────────────────────────────────────

if pergunta := st.chat_input("Pergunte sobre emprego, PIB, inflação, indústria…"):
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        if checar_saude(base_url) is None:
            erro = f"Não consegui falar com a API em `{base_url}`. Verifique se a variante está no ar."
            st.error(erro)
            st.session_state.mensagens.append({"role": "assistant", "content": erro})
        else:
            with st.spinner("Buscando e analisando…"):
                try:
                    dados = perguntar(base_url, pergunta)
                    resposta = dados.get("answer", "(resposta vazia)")
                    meta = {
                        "route": dados.get("route"),
                        "rewritten_query": dados.get("rewritten_query"),
                        "sources_used": dados.get("sources_used", []),
                        "sources": dados.get("sources", []),
                        "validation": dados.get("validation", {}),
                    }
                    st.markdown(resposta)
                    render_meta(meta)
                    st.session_state.mensagens.append(
                        {"role": "assistant", "content": resposta, "meta": meta}
                    )
                except requests.HTTPError as exc:
                    detalhe = ""
                    try:
                        detalhe = exc.response.json().get("detail", "")
                    except Exception:
                        detalhe = exc.response.text if exc.response is not None else str(exc)
                    erro = f"Erro {exc.response.status_code if exc.response is not None else ''}: {detalhe}"
                    st.error(erro)
                    st.session_state.mensagens.append({"role": "assistant", "content": erro})
                except requests.RequestException as exc:
                    erro = f"Falha de rede: {exc}"
                    st.error(erro)
                    st.session_state.mensagens.append({"role": "assistant", "content": erro})
