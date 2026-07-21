"""Validação determinística das citações explícitas geradas pelas engines."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_CITATION_RE = re.compile(
    r"\(Fonte:\s*([^,\)\n]+?)"
    r"(?:,\s*(?:p(?:ágina)?\.?|p\./aba|aba)\s*:?[ ]*([^\)\n]+?))?\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationCheck:
    citation: str
    verified: bool


def _normal_file(value: str) -> str:
    return value.strip().replace("\\", "/").lower()


def _node_files(node) -> set[str]:
    metadata = getattr(node, "metadata", {}) or {}
    raw = metadata.get("source_files") or metadata.get("source_file") or metadata.get("file_name")
    if not raw:
        return set()
    files = {part.strip() for part in str(raw).split(",") if part.strip()}
    normalized = {_normal_file(value) for value in files}
    normalized.update(os.path.basename(value) for value in tuple(normalized))
    return normalized


def validate_citations(answer: str, source_nodes) -> list[CitationCheck]:
    allowed_files: set[str] = set()
    pages_by_file: dict[str, set[str]] = {}
    for node in source_nodes:
        files = _node_files(node)
        allowed_files.update(files)
        page = (getattr(node, "metadata", {}) or {}).get("page")
        if page is not None:
            for file in files:
                pages_by_file.setdefault(file, set()).add(str(page).strip().lower())

    checks = []
    for match in _CITATION_RE.finditer(answer or ""):
        cited_file = _normal_file(match.group(1))
        candidates = {cited_file, os.path.basename(cited_file)}
        matching_files = candidates & allowed_files
        verified = bool(matching_files)
        cited_page = (match.group(2) or "").strip().lower()
        if verified and cited_page:
            known_pages = set().union(*(pages_by_file.get(file, set()) for file in matching_files))
            if known_pages:
                verified = cited_page in known_pages
        checks.append(CitationCheck(citation=match.group(0), verified=verified))
    return checks
