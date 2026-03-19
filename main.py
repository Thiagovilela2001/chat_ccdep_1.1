import os
import sys
import json
from dotenv import load_dotenv

# Garante UTF-8 no console do Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import chromadb
from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index
from src.qa_chain import get_query_engine, answer_question

MANIFEST_FILE = "chroma_db/indexed_manifest.json"

def get_data_snapshot(data_dir):
    """Retorna um dicionário {nome_arquivo: ultima_modificacao} dos arquivos em /data."""
    snapshot = {}
    if not os.path.isdir(data_dir):
        return snapshot
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            snapshot[fname] = os.path.getmtime(fpath)
    return snapshot

def load_manifest():
    """Carrega o manifesto de arquivos já indexados."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(snapshot):
    """Salva o estado atual dos arquivos como manifesto."""
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

def detect_changes(current_snapshot, manifest):
    """Retorna lista de arquivos novos ou modificados."""
    changed = []
    for fname, mtime in current_snapshot.items():
        if fname not in manifest or manifest[fname] != mtime:
            changed.append(fname)
    return changed

def main():
    # 1. Carrega variáveis de ambiente (ex: OPENAI_API_KEY)
    load_dotenv()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_PATH = os.path.join(BASE_DIR, "chroma_db")

    # Validações Iniciais
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERRO] AVISO CRÍTICO: Chave da OpenAI não encontrada.")
        print("Por favor, crie um arquivo .env na pasta raiz e adicione: OPENAI_API_KEY=sua_chave\n")
        return

    print("==================================================")
    print(" INICIALIZANDO RAG ESTATÍSTICO (LlamaIndex + GPT)")
    print("==================================================")

    # 2. Detecta mudanças nos documentos
    current_snapshot = get_data_snapshot(DATA_DIR)
    manifest = load_manifest()
    changed_files = detect_changes(current_snapshot, manifest)

    db = chromadb.PersistentClient(path=DB_PATH)
    col = db.get_or_create_collection("estatisticas")
    already_indexed = col.count() > 0

    # 3. Ingestão e processamento apenas se necessário
    nodes = []
    if changed_files:
        print(f"\n1. Documentos novos/modificados detectados: {changed_files}")
        print("   Reindexando...")
        # Limpa o banco para reindexar tudo de forma consistente
        db.delete_collection("estatisticas")
        col = db.get_or_create_collection("estatisticas")

        print("\n2. Carregando e processando documentos de /data...")
        docs = load_documents(DATA_DIR)
        if docs:
            nodes = process_documents(docs)
    elif not already_indexed:
        print("\n1. Banco vazio. Carregando documentos de /data...")
        docs = load_documents(DATA_DIR)
        if docs:
            print("\n2. Processando/Normalizando os documentos extraídos...")
            nodes = process_documents(docs)
        else:
            print("\n2. Nenhum documento encontrado em /data.")
    else:
        print(f"\n1. Banco de dados atualizado ({col.count()} vetores). Nenhuma mudança detectada.")

    # 4. Indexação (Vector Store local ChromaDB + Embeddings)
    print("\n3. Indexando vetores e metadados no ChromaDB...")
    try:
        index = create_or_load_index(nodes, db_path=DB_PATH)
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar banco de dados ou criar embeddings. Detalhes: {e}")
        return

    # Salva manifesto após indexação bem-sucedida
    if changed_files or not already_indexed:
        save_manifest(current_snapshot)

    # 5. RAG Motor de Busca
    print("\n4. Inicializando Motor de Consulta...")
    query_engine = get_query_engine(index)

    print("\n" + "="*50)
    print(" RAG PRONTO PARA USO!")
    print(" Digite sua pergunta sobre os documentos.")
    print(" Digite 'sair' para encerrar o programa.")
    print("="*50 + "\n")

    while True:
        try:
            pergunta = input("▶️ Pergunta: ")
            if pergunta.lower().strip() in ["sair", "exit", "quit", "q"]:
                print("Encerrando...")
                break

            if not pergunta.strip():
                continue

            print("⏳ Buscando e analisando dados...\n")
            resposta = answer_question(query_engine, pergunta)

            # Resposta final em texto gerada pelo LLM
            print(f"✅ Resposta:\n{resposta.response}\n")

            # Validação via Fontes/Metadados Extrativos do Retrieval Original
            print("🔍 Referências e Fontes da Extração Original:")
            for i, node in enumerate(resposta.source_nodes):
                file_name = node.metadata.get('file_name', 'Nome Desconhecido')
                score = node.score if node.score is not None else 0.0
                print(f"  [{i+1}] 📄 Documento: {file_name} (Confiança/Score: {score:.3f})")

            print("\n" + "-"*40 + "\n")

        except KeyboardInterrupt:
            print("\nProcesso interrompido.")
            break
        except Exception as e:
            print(f"\n[ERRO] Ocorreu um problema durante a consulta: {e}\n")

if __name__ == "__main__":
    main()
