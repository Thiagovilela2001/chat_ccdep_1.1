"""
Nota de fonte sob cada tabela da resposta (modo artigo).

A fonte não é pedida ao modelo: `sanitize_answer` apaga linhas "Fonte:", nomes de
arquivo e páginas, e uma atribuição escrita pelo LLM não seria rastreável. Aqui ela
vem dos números: primeiro os que o validador numérico verificou dentro da tabela
(cada um aponta o trecho de onde saiu); depois os trechos que contêm literalmente
os números da tabela, porque células costumam vir sem unidade. As posições das
citações numéricas depois de cada nota são deslocadas para apontar o mesmo número.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace

from .provenance import source_file, source_page

_TABLE_LINE_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_CELL_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
_MAX_LOOSE_SOURCES = 3


def _distinctive_numbers(table: str) -> set[str]:
    """Números da tabela que identificam um trecho: sem anos e sem dígito isolado."""
    numbers = set()
    for raw in _CELL_NUMBER_RE.findall(table):
        if len(raw) < 2 or (raw.isdigit() and 1990 <= int(raw) <= 2035):
            continue
        numbers.add(raw)
    return numbers


def _loose_source_indexes(table: str, source_nodes: list) -> list[int]:
    """Trechos que contêm literalmente mais números da tabela.

    Tabelas costumam trazer o número sem a unidade ("63,4" na coluna "R$ bilhões"),
    o que o validador estrito não confirma; para atribuir a fonte basta o número
    aparecer no trecho. Exige 2+ números em comum, salvo tabela de um número só.
    """
    numbers = _distinctive_numbers(table)
    if not numbers:
        return []
    hits = []
    for index, node in enumerate(source_nodes):
        content = node.get_content() if hasattr(node, "get_content") else str(node)
        found = sum(1 for n in numbers if re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)", content))
        if found:
            hits.append((found, index))
    needed = 1 if len(numbers) == 1 else 2
    ranked = sorted((h for h in hits if h[0] >= needed), key=lambda h: (-h[0], h[1]))
    return [index for _found, index in ranked[:_MAX_LOOSE_SOURCES]]


def _table_ends(answer: str) -> list[tuple[int, int]]:
    """(início, fim) de cada bloco de 2+ linhas de tabela markdown."""
    spans, start, end, rows, pos = [], None, 0, 0, 0
    for line in answer.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if _TABLE_LINE_RE.match(bare):
            if start is None:
                start, rows = pos, 0
            rows += 1
            end = pos + len(bare)
        elif start is not None:
            if rows >= 2:
                spans.append((start, end))
            start = None
        pos += len(line)
    if start is not None and rows >= 2:
        spans.append((start, end))
    return spans


def document_label(node) -> str:
    """'Fundação Seade, seade trabalho taxa informalidade ... 2026, p. 2'."""
    name = os.path.splitext(os.path.basename(source_file(node)))[0]
    name = re.sub(r"[-_]+", " ", name).strip() or "documento sem nome"
    page = source_page(node)
    return f"Fundação Seade, {name}" + (f", p. {page}" if page not in (None, "") else "")


def annotate_table_sources(answer: str, checks: list, source_nodes: list) -> tuple[str, list]:
    """Anexa '*Fonte: ...*' sob cada tabela com números verificados; desloca `checks`."""
    inserts: list[tuple[int, str]] = []
    for start, end in _table_ends(answer):
        labels: list[str] = []
        for check in sorted(checks, key=lambda c: c.response_start or 0):
            inside = check.response_start is not None and start <= check.response_start < end
            if not (inside and check.verified and check.source_index is not None):
                continue
            if 0 <= check.source_index < len(source_nodes):
                label = document_label(source_nodes[check.source_index])
                if label not in labels:
                    labels.append(label)
        for index in _loose_source_indexes(answer[start:end], source_nodes):
            label = document_label(source_nodes[index])
            if label not in labels:
                labels.append(label)
        if labels:
            inserts.append((end, f"\n\n*Fonte: {'; '.join(labels)}.*"))
    if not inserts:
        return answer, checks

    for pos, note in reversed(inserts):
        answer = answer[:pos] + note + answer[pos:]

    def shift(offset: int | None) -> int | None:
        if offset is None:
            return None
        return offset + sum(len(note) for pos, note in inserts if pos <= offset)

    shifted = [
        replace(c, response_start=shift(c.response_start), response_end=shift(c.response_end))
        for c in checks
    ]
    return answer, shifted
