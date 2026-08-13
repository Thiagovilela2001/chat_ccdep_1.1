#!/usr/bin/env python3
"""Gera inventário estável dos PDFs em data/seade_social."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import fitz
except ImportError as exc:  # pragma: no cover - depende do ambiente
    raise SystemExit("PyMuPDF ausente. Instale com: pip install pymupdf") from exc


def classify(collection: str, filename: str) -> str:
    name = filename.casefold()
    if collection == "painel":
        if "cadunico" in name or "inscrit" in name or "reducao-pessoas" in name:
            return "cadunico"
        if "bolsa-familia" in name or "beneficios-programa" in name:
            return "bolsa_familia"
        return "transferencia_renda"

    rules = (
        (r"desigualdad", "desigualdades"),
        (r"diferenc?a?s?-regionais|diferenca-regionais", "diferencas_regionais"),
        (r"informalidade", "informalidade"),
        (r"renda-domiciliar", "renda_domiciliar_do_trabalho"),
        (r"emprego-formal|empregos", "emprego_formal"),
        (r"ocupacao-rendimento|ocupados-e-rendimento|trabalho-anual", "ocupacao_rendimento"),
        (r"mensal", "sintese_mensal"),
    )
    for pattern, topic in rules:
        if re.search(pattern, name):
            return topic
    return "mercado_de_trabalho"


def pdf_metadata(path: Path) -> tuple[str, int]:
    with fitz.open(path) as document:
        title = (document.metadata.get("title") or "").strip()
        return title, document.page_count


def build_manifest(corpus: Path) -> dict:
    documents = []
    for path in sorted(corpus.rglob("*.pdf")):
        relative = path.relative_to(corpus).as_posix()
        collection = relative.split("/", 1)[0]
        title, pages = pdf_metadata(path)
        documents.append(
            {
                "path": relative,
                "collection": collection,
                "topic": classify(collection, path.name),
                "title": title,
                "pages": pages,
                "bytes": path.stat().st_size,
            }
        )

    by_collection = Counter(item["collection"] for item in documents)
    by_topic = Counter(item["topic"] for item in documents)
    return {
        "schema_version": 1,
        "corpus": "data/seade_social",
        "total_documents": len(documents),
        "by_collection": dict(sorted(by_collection.items())),
        "by_topic": dict(sorted(by_topic.items())),
        "documents": documents,
    }


def parse_args() -> argparse.Namespace:
    skill_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repo_root / "data" / "seade_social",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=skill_dir / "references" / "corpus-manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Corpus não encontrado: {corpus}", file=sys.stderr)
        return 1

    manifest = build_manifest(corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{manifest['total_documents']} PDFs inventariados em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
