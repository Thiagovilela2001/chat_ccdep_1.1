"""
components/metrics.py — Exibição de métricas de rota e desempenho.

`render_inline`  → resumo compacto sob cada resposta.
`render_sidebar` → painel detalhado (tempos, docs, modelos) na barra lateral.
Toda a informação vem do backend; o frontend apenas apresenta.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from utils.formatting import NA, format_confidence, format_ms, n_docs, safe_get


def render_inline(meta: dict[str, Any]) -> None:
    """Badge compacto sob uma resposta: engine escolhida, tipo, confiança, tempo."""
    route = meta.get("route") or {}
    engine = route.get("engine_label") or route.get("engine")
    if not engine:
        return
    partes = [f"🧠 **{engine}**"]
    if route.get("query_type"):
        partes.append(str(route["query_type"]))
    if isinstance(route.get("confidence"), (int, float)):
        partes.append(f"conf {format_confidence(route['confidence'])}")
    total = safe_get(meta, "timings", "total_ms", default=meta.get("_client_roundtrip_ms"))
    partes.append(format_ms(total))
    st.caption("  ·  ".join(partes))


def render_sidebar(meta: dict[str, Any] | None) -> None:
    """Painel de métricas da última resposta, na barra lateral."""
    st.subheader("📊 Última resposta")
    if not meta:
        st.caption("Faça uma pergunta para ver as métricas.")
        return

    route = meta.get("route") or {}
    st.markdown(f"**Engine selecionada:** {route.get('engine_label') or route.get('engine') or NA}")
    conf = route.get("confidence")
    st.markdown(f"**Confiança do Router:** {format_confidence(conf)}")
    if route.get("mode"):
        st.caption(f"Modo: {route['mode']}")

    timings = meta.get("timings") or {}
    col1, col2 = st.columns(2)
    col1.metric("⏱️ Total", format_ms(timings.get("total_ms")))
    col2.metric("📄 Documentos", n_docs(meta))
    col1.metric("🔎 Analyzer", format_ms(timings.get("analyzer_ms")))
    col2.metric("🧭 Router", format_ms(timings.get("router_ms")))
    col1.metric("⚙️ Engine", format_ms(timings.get("engine_ms")))

    models = meta.get("models") or {}
    st.markdown("**Modelos**")
    st.caption(f"LLM (geração): `{models.get('generation') or NA}`")
    st.caption(f"Analyzer: `{models.get('analyzer') or NA}`")
    st.caption(f"Embeddings: `{models.get('embeddings') or NA}`")
