"""
Validação numérica de respostas RAG.

Pipeline:
    resposta do LLM → extrai números → compara com chunks de origem → relatório
"""
import re
from dataclasses import dataclass
from typing import Optional

# ── Regex para números em PT-BR ───────────────────────────────────────────────
# Captura: 1.234,56 | 1.234 | 4,2 | 42 | com % opcional
# Exclui inteiros de 1 dígito (muito comuns, sem valor estatístico)
_NUM_RE = re.compile(
    r"(?<!\w)"
    r"([-−]?"
    r"(?:"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"   # 1.234 ou 1.234,56
    r"|\d+,\d+"                         # 4,2
    r"|\d{2,}"                          # inteiro ≥ 2 dígitos
    r")"
    r"(?:\s*%)?)"
    r"(?!\w)"
)

# ── Dataclass de resultado ────────────────────────────────────────────────────

@dataclass
class NumberCheck:
    value: str                          # como aparece na resposta
    verified: bool
    source_snippet: Optional[str] = None  # trecho onde foi encontrado


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Converte número PT-BR para string numérica comparável."""
    s = s.strip().rstrip("%").strip()
    if "," in s and "." in s:
        # 1.234,56 → 1234.56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # 4,2 → 4.2
        s = s.replace(",", ".")
    return s


def _find_snippet(needle: str, haystack: str, ctx: int = 60) -> str:
    idx = haystack.find(needle)
    if idx == -1:
        return ""
    start = max(0, idx - ctx)
    end = min(len(haystack), idx + len(needle) + ctx)
    return "…" + haystack[start:end].strip() + "…"


# ── Validação principal ───────────────────────────────────────────────────────

def validate_numbers(response_text: str, source_nodes) -> list[NumberCheck]:
    """
    Extrai todos os números da resposta e verifica se existem nos chunks de origem.
    Tenta primeiro match verbatim, depois match normalizado (diferenças de formatação).
    """
    source_texts = [n.get_content() for n in source_nodes]
    combined = " ".join(source_texts)

    seen: set[str] = set()
    results: list[NumberCheck] = []

    for m in _NUM_RE.finditer(response_text):
        raw = m.group(1).strip()
        if raw in seen:
            continue
        seen.add(raw)

        # 1. Match verbatim
        if raw in combined:
            results.append(NumberCheck(
                value=raw,
                verified=True,
                source_snippet=_find_snippet(raw, combined),
            ))
            continue

        # 2. Match normalizado (trata diferenças de formatação PT-BR vs EN)
        norm = _normalize(raw)
        found = False
        for node_text in source_texts:
            for cm in _NUM_RE.finditer(node_text):
                if _normalize(cm.group(1).strip()) == norm:
                    results.append(NumberCheck(
                        value=raw,
                        verified=True,
                        source_snippet=_find_snippet(cm.group(1), node_text),
                    ))
                    found = True
                    break
            if found:
                break

        if not found:
            results.append(NumberCheck(value=raw, verified=False))

    return results


def format_validation_report(checks: list[NumberCheck]) -> str:
    """Formata o relatório de validação para exibição no console."""
    if not checks:
        return "  Nenhum número encontrado na resposta."

    verified = [c for c in checks if c.verified]
    unverified = [c for c in checks if not c.verified]

    lines = [f"  Verificados nos documentos: {len(verified)}/{len(checks)}"]

    if unverified:
        lines.append("  ⚠️  Não encontrados nos documentos originais:")
        for c in unverified:
            lines.append(f"    • {c.value}")
    else:
        lines.append("  ✅ Todos os números foram verificados nos documentos.")

    return "\n".join(lines)
