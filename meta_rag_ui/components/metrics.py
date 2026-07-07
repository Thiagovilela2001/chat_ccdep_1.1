"""
components/metrics.py — Exibição de métricas, fontes e validação numérica.

`render_inline`   → resumo compacto sob cada resposta.
`render_details`  → expander com fontes recuperadas + validação numérica.
`render_sidebar`  → painel detalhado da última resposta na barra lateral.

Funciona com as duas formas de resposta do projeto:
- Meta RAG: inclui `route`/`timings`/`models` (+ os campos da engine escolhida).
- Engine direta: `rag_label`, `sources_used`, `sources`, `validation`.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from utils.formatting import NA, format_confidence, format_ms, n_docs, safe_get


def _validation_summary(meta: dict[str, Any]) -> str | None:
    """'✓ 3/4 números verificados' a partir de `validation`, se presente."""
    val = meta.get("validation") or {}
    total = val.get("total")
    if not isinstance(total, int) or total == 0:
        return None
    verified = val.get("verified", 0)
    icon = "✅" if verified == total else "⚠️"
    return f"{icon} {verified}/{total} números verificados"


def render_inline(meta: dict[str, Any]) -> None:
    """Badge compacto sob uma resposta: engine, tipo, confiança, validação, tempo."""
    route = meta.get("route") or {}
    partes: list[str] = []

    engine = route.get("engine_label") or route.get("engine") or meta.get("rag_label")
    if engine:
        partes.append(f"🧠 **{engine}**")
    if route.get("query_type"):
        partes.append(str(route["query_type"]))
    if isinstance(route.get("confidence"), (int, float)):
        partes.append(f"conf {format_confidence(route['confidence'])}")
    if meta.get("sources_used"):
        partes.append("fontes: " + ", ".join(meta["sources_used"]))
    val = _validation_summary(meta)
    if val:
        partes.append(val)
    total = safe_get(meta, "timings", "total_ms", default=meta.get("_client_roundtrip_ms"))
    if total is not None:
        partes.append(format_ms(total))

    if partes:
        st.caption("  ·  ".join(partes))


def render_details(meta: dict[str, Any]) -> None:
    """Expander com fontes recuperadas e validação numérica da resposta."""
    sources = meta.get("sources") or []
    val = meta.get("validation") or {}
    unverified = val.get("unverified") or []
    if not sources and not unverified:
        return

    with st.expander(f"📄 Fontes e validação ({len(sources)} documento(s))"):
        if meta.get("rewritten_query"):
            st.caption(f"Query reescrita: _{meta['rewritten_query']}_")
        if sources:
            st.table(sources)
        if unverified:
            st.warning(
                "Números na resposta **não localizados** nas fontes: "
                + ", ".join(str(u) for u in unverified)
            )
        elif isinstance(val.get("total"), int) and val["total"] > 0:
            st.success(f"Todos os {val['total']} números da resposta conferem com as fontes.")


def render_sidebar(meta: dict[str, Any] | None) -> None:
    """Painel de métricas da última resposta, na barra lateral."""
    st.subheader("📊 Última resposta")
    if not meta:
        st.caption("Faça uma pergunta para ver as métricas.")
        return

    route = meta.get("route") or {}
    engine = route.get("engine_label") or route.get("engine") or meta.get("rag_label")
    st.markdown(f"**Engine:** {engine or NA}")
    if route:
        st.markdown(f"**Confiança do Router:** {format_confidence(route.get('confidence'))}")
        if route.get("mode"):
            st.caption(f"Modo: {route['mode']}")

    timings = meta.get("timings") or {}
    col1, col2 = st.columns(2)
    total = timings.get("total_ms", meta.get("_client_roundtrip_ms"))
    col1.metric("⏱️ Total", format_ms(total))
    col2.metric("📄 Documentos", n_docs(meta))
    if timings:
        col1.metric("🔎 Analyzer", format_ms(timings.get("analyzer_ms")))
        col2.metric("🧭 Router", format_ms(timings.get("router_ms")))
        col1.metric("⚙️ Engine", format_ms(timings.get("engine_ms")))

    val = _validation_summary(meta)
    if val:
        st.markdown(f"**Validação numérica:** {val}")

    models = meta.get("models") or {}
    if models:
        st.markdown("**Modelos**")
        st.caption(f"LLM (geração): `{models.get('generation') or NA}`")
        st.caption(f"Analyzer: `{models.get('analyzer') or NA}`")
        st.caption(f"Embeddings: `{models.get('embeddings') or NA}`")
