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
