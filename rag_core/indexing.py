import os
import pickle
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Cache BM25 ancorado ao db_path da engine — não ao CWD. O default "./chroma_db"
# preserva o comportamento de quem roda do diretório da engine (main.py, Docker).
_NODES_CACHE_NAME = "bm25_nodes.pkl"

def _nodes_cache_path(db_path: str) -> str:
    return os.path.join(db_path, _NODES_CACHE_NAME)

def save_nodes_cache(nodes, db_path="./chroma_db"):
    os.makedirs(db_path, exist_ok=True)
    path = _nodes_cache_path(db_path)
    temporary = path + ".tmp"
    with open(temporary, "wb") as f:
        pickle.dump(nodes, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def load_nodes_cache(db_path="./chroma_db"):
    path = _nodes_cache_path(db_path)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            return []
    return []

def reset_nodes_cache(db_path="./chroma_db"):
    """Remove o cache lexical quando o índice vetorial será reconstruído."""
    path = _nodes_cache_path(db_path)
    if os.path.exists(path):
        os.remove(path)


def merge_nodes_cache(cached_nodes, changed_sources, new_nodes):
    """Substitui no cache somente nós pertencentes às fontes alteradas."""
    changed = set(changed_sources)
    retained = [
        node for node in cached_nodes
        if node.metadata.get("source_file") not in changed
    ]
    return retained + list(new_nodes)


def _node_ids_for_sources(collection, source_files):
    """Captura IDs atuais antes da inserção dos nós substitutos."""
    ids: list[str] = []
    for source_file in source_files:
        result = collection.get(where={"source_file": source_file})
        ids.extend(result.get("ids") or [])
    return ids


def _delete_ids(collection, ids, batch_size=1000):
    for start in range(0, len(ids), batch_size):
        collection.delete(ids=ids[start:start + batch_size])

def setup_embeddings():
    """
    Configura Embeddings Locais (HuggingFace) para não enviar dados sensíveis p/ nuvem na vetorização.
    Utilizando modelo aberto super leve (BGE pequenos ou similar) adequado para buscas em PT-BR/EN.
    """
    print("Configurando modelo de embeddings local (BAAI/bge-m3) ...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.embed_model = embed_model
    return embed_model

def create_or_load_index(nodes, db_path="./chroma_db", collection_name="estatisticas"):
    """
    Cria um VectorStore Persistente via ChromaDB local, permitindo reutilizar o banco entre sessões.
    """
    setup_embeddings()
    
    print(f"Acessando/Criando ChromaDB no diretório: {db_path}")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if nodes and len(nodes) > 0:
        print(f"Indexando {len(nodes)} blocos processados no BD (isso fará download do modelo HF na 1ª vez)...")
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        save_nodes_cache(nodes, db_path)
        print(f"Cache BM25 salvo em {_nodes_cache_path(db_path)}")
    else:
        print("Carregando conhecimento a partir do banco de dados vetorial já populado...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

    return index


def update_index_incrementally(
    nodes,
    changed_sources,
    db_path="./chroma_db",
    collection_name="estatisticas",
):
    """Insere substitutos e só depois remove IDs antigos das fontes alteradas.

    A ordem preserva o índice anterior quando embedding/inserção falha. O
    manifesto só deve ser salvo pelo chamador após esta função concluir.
    """
    setup_embeddings()

    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection(collection_name)
    stale_ids = _node_ids_for_sources(collection, changed_sources)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    if nodes:
        print(f"Indexando incrementalmente {len(nodes)} bloco(s) novo(s)/alterado(s)...")
        index.insert_nodes(list(nodes))

    _delete_ids(collection, stale_ids)

    cached_nodes = load_nodes_cache(db_path)
    merged_nodes = merge_nodes_cache(cached_nodes, changed_sources, nodes)
    save_nodes_cache(merged_nodes, db_path)
    print(
        f"Atualização incremental concluída: {len(stale_ids)} bloco(s) antigo(s) "
        f"removido(s), {len(nodes)} inserido(s)."
    )
    return index
