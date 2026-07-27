"""Sincronização incremental dos índices vetorial e lexical padrão."""
from __future__ import annotations

import chromadb

from rag_core.index_manifest import (
    data_snapshot,
    detect_changes,
    load_manifest,
    save_manifest,
)
from rag_core.indexing import (
    create_or_load_index,
    load_nodes_cache,
    reset_nodes_cache,
    update_index_incrementally,
)
from rag_core.ingestion import load_documents
from rag_core.processing import process_documents


def sync_standard_index(data_dir: str, db_path: str, log, collection_name="estatisticas"):
    """Atualiza somente fontes alteradas; reconstrói quando cache base falta."""
    snapshot = data_snapshot(data_dir)
    manifest = load_manifest(db_path)
    changed = detect_changes(snapshot, manifest)

    if not snapshot:
        raise RuntimeError(
            f"Nenhum documento encontrado em '{data_dir}'; índice anterior preservado."
        )

    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection(collection_name)
    already_indexed = collection.count() > 0

    if changed and already_indexed and load_nodes_cache(db_path):
        upserts = [source for source in changed if source in snapshot]
        removed = [source for source in changed if source not in snapshot]
        log.info(
            "[1] Atualização incremental: %d novo(s)/modificado(s), %d removido(s)",
            len(upserts),
            len(removed),
        )
        docs = (
            load_documents(data_dir, source_files=upserts, save_output=False)
            if upserts
            else []
        )
        nodes = process_documents(docs) if docs else []
        index = update_index_incrementally(
            nodes,
            changed,
            db_path=db_path,
            collection_name=collection_name,
        )
        save_manifest(db_path, snapshot)
        return index, changed

    if changed or not already_indexed:
        if changed and already_indexed:
            log.warning("[1] Cache BM25 ausente; reconstruindo índice completo")
        else:
            log.info("[1] Banco vazio; indexando corpus completo")

        docs = load_documents(data_dir)
        if not docs:
            raise RuntimeError(
                f"Nenhum documento encontrado em '{data_dir}'; índice anterior preservado."
            )
        nodes = process_documents(docs)
        if not nodes:
            raise RuntimeError("Processamento não produziu nós; índice anterior preservado.")

        reset_nodes_cache(db_path)
        if already_indexed:
            db.delete_collection(collection_name)
            db.get_or_create_collection(collection_name)
        index = create_or_load_index(
            nodes,
            db_path=db_path,
            collection_name=collection_name,
        )
        save_manifest(db_path, snapshot)
        return index, changed

    log.info("[1] Banco atualizado (%d vetores); sem mudanças", collection.count())
    return (
        create_or_load_index([], db_path=db_path, collection_name=collection_name),
        changed,
    )
