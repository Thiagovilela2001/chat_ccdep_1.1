"""
utils/formatting.py — Funções puras de formatação e acesso seguro a dados.

Sem dependência de Streamlit: apenas transformam valores para exibição.
"""
from __future__ import annotations

import json
from typing import Any

NA = "—"


def safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """Acesso aninhado tolerante: safe_get(d, 'a', 'b') == d['a']['b'] ou default."""
    cur = data
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def format_ms(ms: float | int | None) -> str:
    """Milissegundos → '820 ms' ou '1.34 s'. `None` → '—'."""
    if ms is None:
        return NA
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return NA
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def format_confidence(conf: float | int | None) -> str:
    """Confiança 0–1 → percentual. `None`/inválido → '—'."""
    if not isinstance(conf, (int, float)):
        return NA
    return f"{conf * 100:.0f}%"


def n_docs(meta: dict[str, Any]) -> int:
    """Quantidade de documentos recuperados (tamanho de `sources`)."""
    sources = meta.get("sources")
    return len(sources) if isinstance(sources, list) else 0


def pretty_json(obj: Any) -> str:
    """Serializa um objeto para exibição legível (JSON identado, UTF-8)."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)
