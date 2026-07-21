"""
components/metrics.py — Métricas, fontes e validação numérica sob cada resposta.

`render_inline`   → chips compactos (engine, tipo, confiança, validação, tempo).
`render_details`  → expander com as citações (cartões) + validação numérica.

Funciona com as duas formas de resposta do projeto:
- Meta RAG: inclui `route`/`timings`/`models` (+ campos da engine escolhida).
- Engine direta: `rag_label`, `sources_used`, `sources`, `validation`.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from utils.formatting import format_confidence, format_ms, safe_get
from utils.ui import chip


def _validation_summary(meta: dict[str, Any]) -> tuple[str, str] | None:
    """('3/4 números verificados', variant) a partir de `validation`."""
    val = meta.get("validation") or {}
    total = val.get("total")
    if not isinstance(total, int) or total == 0:
        return None
    verified = val.get("verified", 0)
    variant = "chip-ok" if verified == total else "chip-warn"
    return f"{verified}/{total} números verificados", variant


def _citation_summary(meta: dict[str, Any]) -> tuple[str, str] | None:
    val = meta.get("citation_validation") or {}
    total = val.get("total")
    if not isinstance(total, int) or total == 0:
        return None
    verified = val.get("verified", 0)
    variant = "chip-ok" if verified == total else "chip-warn"
    return f"{verified}/{total} citações verificadas", variant


def render_inline(meta: dict[str, Any]) -> None:
    """Linha de chips sob uma resposta."""
    route = meta.get("route") or {}
    chips: list[str] = []

    engine = route.get("engine_label") or route.get("engine") or meta.get("rag_label")
    if engine:
        chips.append(chip(str(engine), "layers", "chip-accent"))
    if route.get("query_type"):
        chips.append(chip(str(route["query_type"]), "compass"))
    if isinstance(route.get("confidence"), (int, float)):
        chips.append(chip(f"conf {format_confidence(route['confidence'])}", "activity"))
    if meta.get("sources_used"):
        chips.append(chip(", ".join(meta["sources_used"]), "database"))
    val = _validation_summary(meta)
    if val:
        chips.append(chip(val[0], "check-circle" if val[1] == "chip-ok" else "alert-triangle", val[1]))
    citations = _citation_summary(meta)
    if citations:
        chips.append(chip(
            citations[0],
            "check-circle" if citations[1] == "chip-ok" else "alert-triangle",
            citations[1],
        ))
    total = safe_get(meta, "timings", "total_ms", default=meta.get("_client_roundtrip_ms"))
    if total is not None:
        chips.append(chip(format_ms(total), "clock"))

    if chips:
        st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)


def _cite_card(source: dict[str, Any]) -> str:
    """Cartão de citação a partir de um dict de fonte (chaves arbitrárias)."""
    rows = "".join(
        f'<div><span class="cite-k">{escape(str(k))}</span> '
        f'<span class="cite-v">{escape(str(v))}</span></div>'
        for k, v in source.items()
    )
    return f'<div class="cite">{rows}</div>'


def render_details(meta: dict[str, Any]) -> None:
    """Expander com as citações (cartões) e a validação numérica."""
    sources = meta.get("sources") or []
    val = meta.get("validation") or {}
    unverified = val.get("unverified") or []
    citation_val = meta.get("citation_validation") or {}
    unverified_citations = citation_val.get("unverified") or []
    if not sources and not unverified and not unverified_citations:
        return

    with st.expander(f"Fontes e validação · {len(sources)} documento(s)"):
        if meta.get("rewritten_query"):
            st.caption(f"Query reescrita: _{meta['rewritten_query']}_")
        if sources:
            cards = "".join(
                _cite_card(s) if isinstance(s, dict) else f'<div class="cite">{escape(str(s))}</div>'
                for s in sources
            )
            st.markdown(cards, unsafe_allow_html=True)
        if unverified:
            st.warning(
                "Números na resposta não localizados nas fontes: "
                + ", ".join(str(u) for u in unverified)
            )
        elif isinstance(val.get("total"), int) and val["total"] > 0:
            st.success(f"Todos os {val['total']} números da resposta conferem com as fontes.")
        if unverified_citations:
            st.warning(
                "Citações sem correspondência nas fontes: "
                + ", ".join(str(value) for value in unverified_citations)
            )
        elif isinstance(citation_val.get("total"), int) and citation_val["total"] > 0:
            st.success(
                f"Todas as {citation_val['total']} citações conferem com arquivo e página."
            )
