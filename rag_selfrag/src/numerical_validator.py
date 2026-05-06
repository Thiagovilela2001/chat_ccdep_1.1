"""
Validação numérica de respostas RAG.

Pipeline:
    resposta do LLM → extrai números → compara com chunks de origem → relatório
"""
import re
from dataclasses import dataclass
from typing import Optional

_NUM_RE = re.compile(
    r"(?<!\w)"
    r"([-−]?"
    r"(?:"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d+,\d+"
    r"|\d{2,}"
    r")"
    r"(?:\s*%)?)"
    r"(?!\w)"
)


@dataclass
class NumberCheck:
    value: str
    verified: bool
    source_snippet: Optional[str] = None
    response_snippet: Optional[str] = None


def _normalize(s: str) -> str:
    s = s.strip().rstrip("%").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    return s


def _find_snippet(needle: str, haystack: str, ctx: int = 60) -> str:
    idx = haystack.find(needle)
    if idx == -1:
        return ""
    start = max(0, idx - ctx)
    end = min(len(haystack), idx + len(needle) + ctx)
    return "…" + haystack[start:end].strip() + "…"


def validate_numbers(response_text: str, source_nodes) -> list[NumberCheck]:
    source_texts = [n.get_content() for n in source_nodes]
    combined = " ".join(source_texts)

    seen: set[str] = set()
    results: list[NumberCheck] = []

    for m in _NUM_RE.finditer(response_text):
        raw = m.group(1).strip()
        if raw in seen:
            continue
        seen.add(raw)

        resp_snippet = _find_snippet(raw, response_text)

        if raw in combined:
            results.append(NumberCheck(
                value=raw,
                verified=True,
                source_snippet=_find_snippet(raw, combined),
                response_snippet=resp_snippet,
            ))
            continue

        norm = _normalize(raw)
        found = False
        for node_text in source_texts:
            for cm in _NUM_RE.finditer(node_text):
                if _normalize(cm.group(1).strip()) == norm:
                    results.append(NumberCheck(
                        value=raw,
                        verified=True,
                        source_snippet=_find_snippet(cm.group(1), node_text),
                        response_snippet=resp_snippet,
                    ))
                    found = True
                    break
            if found:
                break

        if not found:
            results.append(NumberCheck(value=raw, verified=False, response_snippet=resp_snippet))

    return results


def format_validation_report(checks: list[NumberCheck]) -> str:
    if not checks:
        return "  Nenhum número encontrado na resposta."

    verified = [c for c in checks if c.verified]
    unverified = [c for c in checks if not c.verified]

    lines = [f"  Verificados nos documentos: {len(verified)}/{len(checks)}"]

    if unverified:
        lines.append("  ⚠️  Não encontrados nos documentos originais:")
        for c in unverified:
            lines.append(f"    • {c.value}")
            if c.response_snippet:
                lines.append(f"      Contexto na resposta: {c.response_snippet}")
    else:
        lines.append("  ✅ Todos os números foram verificados nos documentos.")

    return "\n".join(lines)