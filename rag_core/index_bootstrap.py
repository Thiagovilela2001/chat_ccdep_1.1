"""Bootstrap compartilhado do índice padrão distribuído por GitHub Release."""
from __future__ import annotations

import os
from pathlib import Path

from rag_core.index_manifest import data_snapshot, detect_changes, load_manifest
from scripts.index_artifact import (
    DEFAULT_RELEASE_ASSET,
    DEFAULT_RELEASE_REPO,
    DEFAULT_RELEASE_TAG,
    ensure_release_index,
)


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def ensure_portable_index(db_path: str, data_dir: str | None = None, log=None) -> None:
    """Baixa índice padrão e escolhe carga imutável ou sincronização local."""
    if not _env_enabled("RAG_INDEX_AUTO_DOWNLOAD"):
        return

    repo = os.getenv("RAG_INDEX_REPO", DEFAULT_RELEASE_REPO)
    tag = os.getenv("RAG_INDEX_TAG", DEFAULT_RELEASE_TAG)
    asset = os.getenv("RAG_INDEX_ASSET", DEFAULT_RELEASE_ASSET)
    token = os.getenv("GITHUB_TOKEN") or None
    try:
        timeout = float(os.getenv("RAG_INDEX_DOWNLOAD_TIMEOUT", "600"))
    except ValueError as exc:
        raise RuntimeError("RAG_INDEX_DOWNLOAD_TIMEOUT deve ser numérico.") from exc

    if log:
        log.info("[0] Verificando índice vetorial portátil")
    count, downloaded = ensure_release_index(
        target=Path(db_path),
        repo=repo,
        tag=tag,
        asset=asset,
        token=token,
        timeout=timeout,
    )
    if log:
        if downloaded:
            log.info("[0] Índice baixado e validado (%d vetores)", count)
        else:
            log.info("[0] Índice local válido (%d vetores); download dispensado", count)

    if "RAG_INDEX_READ_ONLY" in os.environ:
        return

    snapshot = data_snapshot(data_dir) if data_dir else {}
    changed = detect_changes(snapshot, load_manifest(db_path)) if snapshot else []
    if changed:
        os.environ["RAG_INDEX_READ_ONLY"] = "0"
        if log:
            log.info(
                "[0] Corpus local contém %d diferença(s); sincronização habilitada",
                len(changed),
            )
    else:
        os.environ["RAG_INDEX_READ_ONLY"] = "1"
