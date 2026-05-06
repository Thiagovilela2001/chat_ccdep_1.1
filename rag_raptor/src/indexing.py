import os
import pickle
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

NODES_CACHE_FILE = "chroma_db/bm25_nodes.pkl"

def save_nodes_cache(nodes):
    os.makedirs(os.path.dirname(NODES_CACHE_FILE), exist_ok=True)
    with open(NODES_CACHE_FILE, "wb") as f:
        pickle.dump(nodes, f)

def load_nodes_cache():
    if os.path.exists(NODES_CACHE_FILE):
        with open(NODES_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return []

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
        save_nodes_cache(nodes)
        print(f"Cache BM25 salvo em {NODES_CACHE_FILE}")
    else:
        print("Carregando conhecimento a partir do banco de dados vetorial já populado...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

    return index