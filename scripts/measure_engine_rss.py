"""Mede o consumo do pipeline SEM o modelo de embedding local.

Mesmas bibliotecas da engine, cache BM25 carregado do disco, indice ligado —
sem carregar o bge-m3. A diferenca para os 1490 MB medidos com o modelo e
exatamente o que a migracao para embedding por API economiza.

psutil quando existir; senao tasklist com a pagina de codigo do console.
"""
import os
import pickle
import subprocess
import sys

sys.path.insert(0, ".")


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {os.getpid()}", "/FO", "CSV", "/NH"],
            capture_output=True,
            encoding="cp850",
            errors="replace",
        ).stdout.strip().splitlines()[-1]
        kb = out.split('","')[4].replace('"', "").replace(".", "").replace(" K", "")
        return float(kb) / 1024


print(f"python recem-iniciado              : {rss_mb():7.1f} MB")

print("importando as mesmas bibliotecas da engine (sem o modelo)...")
import chromadb  # noqa: E402
import pandas  # noqa: E402
import torch  # noqa: E402
from llama_index.core import StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402
import rank_bm25  # noqa: E402

print(f"depois de importar as bibliotecas  : {rss_mb():7.1f} MB")

with open("rag_principal/chroma_db/bm25_nodes.pkl", "rb") as fh:
    nodes = pickle.load(fh)

print(f"depois do cache BM25 carregado     : {rss_mb():7.1f} MB")

db = chromadb.PersistentClient(path="rag_principal/chroma_db")
collection = db.get_or_create_collection("estatisticas")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=MockEmbedding(embed_dim=1024),
)
retriever = index.as_retriever(similarity_top_k=5)

final = rss_mb()
print(f"com indice e retriever ligados     : {final:7.1f} MB")
print(f"\nnos BM25 carregados : {len(nodes)}")
print(f"vetores na colecao  : {collection.count()}")
print(f"SEM o modelo        : {final:.0f} MB  (com bge-m3: 1490 MB)")
print(f"economia da migracao: ~{1490 - final:.0f} MB por processo")
