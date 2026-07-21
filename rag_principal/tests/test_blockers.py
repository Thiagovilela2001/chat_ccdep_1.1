from pathlib import Path

import pandas as pd

from rag_core.index_manifest import (
    data_snapshot,
    detect_changes,
    load_manifest,
    resolve_data_dir,
    save_manifest,
)
from rag_core.indexing import load_nodes_cache, reset_nodes_cache, save_nodes_cache
from rag_core.ingestion import load_documents
from rag_core.provenance import relevance_score, source_file, source_page
from rag_core.tables_retriever import _is_static_table
from rag_core.timeseries_retriever import _df_is_valid_timeseries, _is_temporal_table
from rag_core.runtime import estimate_tokens, limit_context
from rag_raptor.src.raptor_engine import _raptor_level


class _Node:
    def __init__(self, metadata, score=None):
        self.metadata = metadata
        self.score = score


def test_manifest_detecta_subdiretorios_e_remocao(tmp_path):
    data_dir = tmp_path / "data"
    nested = data_dir / "setor"
    nested.mkdir(parents=True)
    document = nested / "boletim.txt"
    document.write_text("conteúdo", encoding="utf-8")

    first = data_snapshot(str(data_dir))
    assert list(first) == ["setor/boletim.txt"]

    document.unlink()
    second = data_snapshot(str(data_dir))
    assert detect_changes(second, first) == ["setor/boletim.txt"]


def test_diretorio_de_dados_local_compartilhado_e_resolvido(tmp_path):
    engine_dir = tmp_path / "rag_principal"
    (engine_dir / "data").mkdir(parents=True)
    shared_dir = tmp_path / "data"
    shared_dir.mkdir()
    (shared_dir / "boletim.pdf").write_bytes(b"pdf")

    assert resolve_data_dir(str(engine_dir)) == str(shared_dir.resolve())


def test_diretorio_da_engine_tem_precedencia_quando_contem_documentos(tmp_path):
    engine_dir = tmp_path / "rag_principal"
    engine_data = engine_dir / "data"
    engine_data.mkdir(parents=True)
    (engine_data / "container.pdf").write_bytes(b"pdf")
    shared_dir = tmp_path / "data"
    shared_dir.mkdir()
    (shared_dir / "local.pdf").write_bytes(b"pdf")

    assert resolve_data_dir(str(engine_dir)) == str(engine_data.resolve())


def test_manifest_corrompido_forca_reconstrucao(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    (db / "indexed_manifest.json").write_text("{", encoding="utf-8")
    assert load_manifest(str(db)) == {}

    snapshot = {"a.txt": {"mtime_ns": 1, "size": 2}}
    save_manifest(str(db), snapshot)
    assert load_manifest(str(db)) == snapshot


def test_reset_cache_remove_nos_bm25_obsoletos(tmp_path):
    save_nodes_cache(["obsoleto"], str(tmp_path))
    assert load_nodes_cache(str(tmp_path)) == ["obsoleto"]
    reset_nodes_cache(str(tmp_path))
    assert load_nodes_cache(str(tmp_path)) == []


def test_ingestao_preserva_caminho_relativo_da_fonte(tmp_path):
    data_dir = tmp_path / "data"
    nested = data_dir / "regional"
    nested.mkdir(parents=True)
    (nested / "boletim.txt").write_text("Informação econômica relevante.", encoding="utf-8")
    docs = load_documents(str(data_dir))
    assert docs[0].metadata["source_file"] == "regional/boletim.txt"


def test_serie_anual_numerica_e_valida():
    df = pd.DataFrame({"Ano": [2022, 2023, 2024], "Valor": [1.0, 2.0, 3.0]})
    assert _df_is_valid_timeseries(df)
    node = _Node({"type": "table", "table_granularidade": "anual"})
    assert _is_temporal_table(node)
    assert not _is_static_table(node)


def test_indice_numerico_generico_nao_e_serie_temporal():
    df = pd.DataFrame({"Índice": [0, 1, 2], "Valor": [1.0, 2.0, 3.0]})
    assert not _df_is_valid_timeseries(df)


def test_nivel_raptor_normaliza_indices_legados():
    assert _raptor_level("2") == 2
    assert _raptor_level(1) == 1
    assert _raptor_level("inválido") == 0
    assert _raptor_level(-1) == 0


def test_proveniencia_normaliza_origem_pagina_e_escore():
    leaf = _Node({"source_file": "regional/a.pdf", "page": 3}, score=0.82)
    summary = _Node({"source_files": "a.pdf, b.pdf"}, score=8.2)
    assert source_file(leaf) == "regional/a.pdf"
    assert source_page(leaf) == 3
    assert relevance_score(leaf) == 0.82
    assert source_file(summary) == "a.pdf, b.pdf"
    assert relevance_score(summary) == 0.82


def test_orcamento_de_contexto_e_deterministico():
    text = "a" * 10_000
    limited = limit_context(text, max_tokens=100)
    assert estimate_tokens(limited) <= 110  # inclui o marcador explícito de truncamento
    assert "Contexto truncado" in limited
