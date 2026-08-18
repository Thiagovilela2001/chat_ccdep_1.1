"""Snapshot compartilhado dos documentos usados na indexação.

O snapshot usa caminhos relativos, tamanho e ``mtime_ns``. Assim, mudanças em
subdiretórios e arquivos removidos também invalidam o índice, sem ler o conteúdo
inteiro de todos os documentos em cada inicialização.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from rag_ccdep.paths import data_dir as project_data_dir

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".txt"}
_MANIFEST_NAME = "indexed_manifest.json"


def resolve_data_dir(base_dir: str, data_dir: str | None = None) -> str:
    """Resolve a base documental sem depender do diretório de execução.

    A base canônica fica em ``data/``. Caminhos próximos ao runtime continuam
    aceitos por compatibilidade. Um caminho explícito sempre tem precedência.
    """
    if data_dir:
        return str(Path(data_dir).expanduser().resolve())

    configured = os.getenv("RAG_DATA_DIR")
    if configured:
        return str(Path(configured).expanduser().resolve())

    canonical_dir = project_data_dir()
    engine_dir = Path(base_dir).resolve() / "data"
    shared_dir = Path(base_dir).resolve().parent / "data"
    for candidate in (engine_dir, shared_dir, canonical_dir):
        if data_snapshot(str(candidate)):
            return str(candidate)
    return str(canonical_dir)


def resolve_db_dir(base_dir: str, db_dir: str | None = None) -> str:
    """Resolve o ChromaDB, permitindo isolar corpus por variável de ambiente.

    Um caminho explícito tem precedência sobre ``RAG_DB_DIR``. Sem ambos,
    preserva o comportamento anterior em ``<engine>/chroma_db``.
    """
    configured = db_dir or os.getenv("RAG_DB_DIR")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path(base_dir).resolve() / "chroma_db")


def data_snapshot(data_dir: str) -> dict[str, dict[str, int]]:
    root = Path(data_dir)
    if not root.is_dir():
        return {}

    snapshot: dict[str, dict[str, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return snapshot


def load_manifest(db_path: str) -> dict:
    path = Path(db_path) / _MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        # Manifesto inválido deve provocar reconstrução, não impedir startup.
        return {}


def save_manifest(db_path: str, snapshot: dict) -> None:
    os.makedirs(db_path, exist_ok=True)
    path = Path(db_path) / _MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def detect_changes(snapshot: dict, manifest: dict) -> list[str]:
    """Retorna caminhos adicionados, modificados ou removidos."""
    return sorted(
        path
        for path in set(snapshot) | set(manifest)
        if snapshot.get(path) != manifest.get(path)
    )
