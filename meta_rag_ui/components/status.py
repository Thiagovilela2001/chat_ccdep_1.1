"""
components/status.py — Painel de status dos serviços RAG do projeto.

Consulta `GET /health` de cada serviço conhecido (mesmas portas do
docker-compose.yml) e apresenta a disponibilidade em tabela. Quando o
orquestrador está no ar, mostra também a visão dele sobre as engines.
"""
from __future__ import annotations

import streamlit as st

from components.sidebar import PRESETS
from services.api import RagClient


def _linha(nome: str, url: str) -> dict[str, str]:
    health = RagClient(url).health()
    if health is None:
        return {"Serviço": nome, "URL": url, "Estado": "🔴 offline", "Detalhe": "—"}
    pronto = health.get("orchestrator_ready", health.get("engine_ready"))
    detalhe = f"pronto: {pronto}" if pronto is not None else "—"
    return {
        "Serviço": health.get("rag_label", nome),
        "URL": url,
        "Estado": "🟢 no ar",
        "Detalhe": detalhe,
    }


def render() -> None:
    """Tabela de disponibilidade de todos os serviços conhecidos."""
    st.subheader("🩺 Status dos serviços")
    st.caption(
        "Portas padrão do `docker-compose.yml`. Suba tudo com `docker compose up` "
        "ou individualmente com `cd <serviço> && python main.py --port <porta>`."
    )

    if st.button("🔄 Atualizar", key="refresh_status"):
        st.rerun()

    linhas = [
        _linha(nome, url)
        for nome, url in PRESETS.items()
        if url  # ignora o preset "URL customizada"
    ]
    st.table(linhas)

    # Visão do orquestrador sobre as engines (se estiver no ar)
    meta_url = next(iter(PRESETS.values()))
    health = RagClient(meta_url).health()
    backends = (health or {}).get("backends") or {}
    if backends:
        st.markdown("**Visão do orquestrador (Meta RAG) sobre as engines:**")
        st.table([
            {
                "Engine": k,
                "Estado": "🟢 no ar" if info.get("up") else "⚪ offline",
            }
            for k, info in backends.items()
        ])
