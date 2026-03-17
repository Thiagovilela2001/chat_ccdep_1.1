import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def setup_llm():
    """Configura o gerador OpenAI (GPT-4o) para elaborar as respostas com temperatura baixa (0.0)
    Evita alucinações ao buscar dados de documentos estatísticos focando em fatos e números exatos do texto base."""
    # Exige OPENAI_API_KEY no .env
    llm = OpenAI(model="gpt-5", temperature=0.0)
    Settings.llm = llm
    return llm

def get_query_engine(index):
    """
    Constrói a engine de perguntas e respostas (RAG).
    A engine é composta de um Retriever (Buscador Similar) e um Synthesizer (LLM formatando a resposta final).
    """
    setup_llm()
    
    # 1. Recupera o top-k blocos com maior semelhança (embeddings)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5, 
    )
    
    # 2. Configura como o LLM unirá as informações buscadas para responder (modo compact otimiza uso do contexto)
    response_synthesizer = get_response_synthesizer(
        response_mode="compact" 
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    
    return query_engine

def answer_question(query_engine, question: str):
    """Encapsula a função de consulta e devolve ao script principal"""
    response = query_engine.query(question)
    return response
