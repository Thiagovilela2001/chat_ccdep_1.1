"""Resolução centralizada de caminhos do projeto e artefatos de execução."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.getenv("RAG_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def data_dir() -> Path:
    configured = os.getenv("RAG_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else project_root() / "data"


def runtime_dir(engine: str) -> Path:
    key = f"RAG_{engine.upper()}_RUNTIME_DIR"
    configured = os.getenv(key)
    if configured:
        return Path(configured).expanduser().resolve()

    base = os.getenv("RAG_RUNTIME_DIR")
    root = Path(base).expanduser().resolve() if base else project_root() / "var"
    return root / engine


def frontend_dist_dir() -> Path:
    return project_root() / "frontend" / "dist"
