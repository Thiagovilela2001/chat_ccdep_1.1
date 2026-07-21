"""
utils/theme.py — Injeção da folha de estilo institucional.

Lê `assets/style.css` (ao lado do pacote) e injeta no app via `st.markdown`.
A paleta base vem de `.streamlit/config.toml`; este módulo cuida do refino
visual (tipografia, cartões, botões, chat) sem alterar o layout.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


@lru_cache(maxsize=1)
def _load_css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def inject() -> None:
    """Aplica o tema institucional. Chamar uma vez, após `set_page_config`."""
    css = _load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
