from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline

def get_processing_pipeline(chunk_size=1024, chunk_overlap=200):
    """
    Cria a pipeline de processamento e normalização que fragmenta (chunking) e 
    extrai metadados dos documentos para enriquecer a base do Vector Store.
    Isso é fundamental para a recuperação híbrida funcionar bem em textos e elementos tabulares brutos.
    """
    transformations = [
        # 1. Quebra de de texto mantendo frases juntas quando possível
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        
        # 2. Extração de títulos - descobre possíveis títulos baseado no início do documento
        TitleExtractor(nodes=5),
        
        # 3. Extração de palavras chaves do texto/tabelas para ajudar em buscas híbridas sintáticas
        KeywordExtractor(keywords=5),
        
        # Opcional futuramente: Sumarização de Nós/Tabelas para Vector Retrieval
    ]
    
    pipeline = IngestionPipeline(transformations=transformations)
    return pipeline

def process_documents(documents):
    """
    Aplica a pipeline inicial de processamento convertendo Documentos Brutos em Nós (Nodes).
    Nós são pedaços de textos prontos para serem vetorizados.
    """
    print(f"Iniciando normalização de dados e extração de metadados para {len(documents)} arquivos/partes.")
    pipeline = get_processing_pipeline()
    nodes = pipeline.run(documents=documents)
    print(f"Normalização concluída. Gerados {len(nodes)} Nós (chunks).")
    return nodes
