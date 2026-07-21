"""Snapshot compartilhado dos documentos usados na indexação.

O snapshot usa caminhos relativos, tamanho e ``mtime_ns``. Assim, mudanças em
subdiretórios e arquivos removidos também invalidam o índice, sem ler o conteúdo
inteiro de todos os documentos em cada inicialização.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".txt"}
_MANIFEST_NAME = "indexed_manifest.json"


def resolve_data_dir(base_dir: str, data_dir: str | None = None) -> str:
    """Resolve a base documental sem depender do diretório de execução.

    Em containers, os documentos ficam normalmente em ``<engine>/data``. Na
    execução local do monorepo, a base compartilhada fica em ``../data``. Um
    caminho explícito sempre tem precedência; entre os defaults, escolhemos o
    primeiro que realmente contém documentos suportados.
    """
    if data_dir:
        return str(Path(data_dir).expanduser().resolve())

    configured = os.getenv("RAG_DATA_DIR")
    if configured:
        return str(Path(configured).expanduser().resolve())

    engine_dir = Path(base_dir).resolve() / "data"
    shared_dir = Path(base_dir).resolve().parent / "data"
    for candidate in (engine_dir, shared_dir):
        if data_snapshot(str(candidate)):
            return str(candidate)
    return str(engine_dir)


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
