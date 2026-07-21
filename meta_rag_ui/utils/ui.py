"""
utils/ui.py — Construtores de HTML para os elementos visuais customizados.

Funções puras: recebem dados e devolvem strings HTML (usando ícones Lucide de
`utils.icons`). Os componentes injetam via `st.markdown(..., unsafe_allow_html=True)`.
Nenhuma lógica de negócio aqui — só apresentação.
"""
from __future__ import annotations

from html import escape

from utils.icons import icon

# Estados de conexão → (classe CSS, rótulo).
_STATUS = {
    "online": ("status-online", "Online"),
    "offline": ("status-offline", "Offline"),
    "connecting": ("status-connecting", "Conectando"),
}


def status_pill(state: str, label: str | None = None, sub: str | None = None) -> str:
    """Indicador ● Online / Offline / Conectando."""
    cls, default = _STATUS.get(state, _STATUS["offline"])
    text = escape(label or default)
    html = (
        f'<div class="status-pill {cls}"><span class="status-dot"></span>{text}</div>'
    )
    if sub:
        html += f'<div class="status-sub">{escape(sub)}</div>'
    return html


def section_header(icon_name: str, title: str) -> str:
    """Cabeçalho de seção da sidebar: ícone + rótulo em caixa-alta."""
    return (
        f'<div class="side-head"><span class="lu">{icon(icon_name, 14)}</span>'
        f'{escape(title)}</div>'
    )


def badge(icon_name: str, label: str, value: str) -> str:
    """Pequeno cartão horizontal (métrica do hero)."""
    return (
        f'<div class="badge"><span class="lu">{icon(icon_name, 18)}</span>'
        f'<span class="badge-txt"><span class="badge-label">{escape(label)}</span>'
        f'<span class="badge-value">{escape(value)}</span></span></div>'
    )


def chip(text: str, icon_name: str | None = None, variant: str = "") -> str:
    """Chip inline (métrica sob uma resposta)."""
    ic = f'<span class="lu">{icon(icon_name, 13)}</span>' if icon_name else ""
    cls = f"chip {variant}".strip()
    return f'<span class="{cls}">{ic}{escape(text)}</span>'
