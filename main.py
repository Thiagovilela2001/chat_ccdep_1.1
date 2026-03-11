import os
from dotenv import load_dotenv

from src.ingestion import load_documents
from src.processing import process_documents
from src.indexing import create_or_load_index
from src.qa_chain import get_query_engine, answer_question

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
    
    # 2. Ingestão
    print("\n1. Verificando e carregando documentos brutos de /data...")
    docs = load_documents(DATA_DIR)
    
    # 3. Processamento/Normalização
    nodes = []
    if len(docs) > 0:
        print("\n2. Processando/Normalizando os documentos extraídos...")
        nodes = process_documents(docs)
    else:
        print("\n2. Nenhum documento novo detectado. Utilizaremos o banco de dados existente.")
        
    # 4. Indexação (Vector Store local ChromaDB + BGE Embeddings)
    print("\n3. Indexando vetores e metadados no ChromaDB...")
    try:
        index = create_or_load_index(nodes, db_path=DB_PATH)
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar banco de dados ou criar embeddings. Detalhes: {e}")
        return
    
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
