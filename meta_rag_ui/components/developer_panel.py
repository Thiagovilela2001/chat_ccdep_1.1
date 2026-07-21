"""
components/developer_panel.py — Painel de depuração (Modo Desenvolvedor).

Exibe os artefatos de decisão produzidos pelo backend (QueryProfile,
EngineProfile, scores, motivo, tempos, documentos). Campos que vivem dentro das
engines e não são expostos pela API (prompt final, logs internos) são
sinalizados como indisponíveis — o frontend nunca os inventa.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from utils.formatting import format_ms, pretty_json, safe_get

_NOT_EXPOSED = "_Não exposto pelo backend nesta versão da API._"


def render(meta: dict[str, Any] | None) -> None:
    """Renderiza o painel de depuração da última resposta."""
    st.subheader("Painel do Desenvolvedor")
    if not meta:
        st.info("Faça uma pergunta para inspecionar o pipeline.")
        return
    if meta.get("error"):
        st.error(f"Última chamada falhou: {meta['error']}")
        return

    route = meta.get("route") or {}

    with st.expander("Decisão de roteamento", expanded=True):
        st.markdown(f"**Engine escolhida:** `{route.get('engine') or '—'}` "
                    f"({route.get('mode') or '—'})")
        st.markdown(f"**Motivo:** {route.get('reasoning') or '—'}")
        st.markdown(f"**Tipo de consulta:** `{route.get('query_type') or '—'}`")
        st.markdown("**Score por engine avaliada:**")
        scores = route.get("scores") or {}
        if scores:
            rows = [{"engine": k, "score": v} for k, v in
                    sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
            st.table(rows)
        else:
            st.caption("—")

    with st.expander("QueryProfile (Analyzer)"):
        st.json(meta.get("analysis") or {})

    with st.expander("EngineProfile escolhido"):
        prof = meta.get("engine_profile")
        if prof:
            st.markdown(f"**{prof.get('label')}** — {prof.get('description')}")
            st.markdown(f"**Estratégia de recuperação:** {', '.join(prof.get('retrieval', [])) or '—'}")
            st.markdown(f"**Forte em:** {', '.join(prof.get('good_for', [])) or '—'}")
            st.markdown(f"**Limitações:** {', '.join(prof.get('limitations', [])) or '—'}")
            st.json(prof)
        else:
            st.caption("—")

    with st.expander("Tempo detalhado do pipeline"):
        timings = meta.get("timings") or {}
        st.markdown(f"- **Analyzer:** {format_ms(timings.get('analyzer_ms'))}")
        st.markdown(f"- **Router:** {format_ms(timings.get('router_ms'))}")
        st.markdown(f"- **Engine:** {format_ms(timings.get('engine_ms'))}")
        st.markdown(f"- **Total:** {format_ms(timings.get('total_ms'))}")

    with st.expander("Documentos recuperados + metadados"):
        sources = meta.get("sources") or []
        if sources:
            st.table(sources)
            st.caption(f"{len(sources)} documento(s). "
                       f"Validação numérica: {safe_get(meta, 'quality', 'numbers_verified', default='—')}")
        else:
            st.caption("Nenhum documento retornado.")

    with st.expander("Prompt final enviado à LLM"):
        st.markdown(meta.get("prompt") or _NOT_EXPOSED)

    with st.expander("Logs do backend"):
        logs = meta.get("logs")
        if logs:
            st.code(logs if isinstance(logs, str) else pretty_json(logs))
        else:
            st.markdown(_NOT_EXPOSED)
