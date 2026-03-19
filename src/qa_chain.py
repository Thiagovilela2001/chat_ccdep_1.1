import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    "Você é um analista especialista em dados econômicos e estatísticos. "
    "Responda somente com base no conteúdo dos documentos fornecidos no contexto. "
    "É proibido usar conhecimento externo, preencher lacunas por inferência especulativa ou inventar qualquer informação.\n\n"
    "Regras obrigatórias:\n"
    "- Use apenas informações presentes nos documentos.\n"
    "- Não invente números, datas, nomes, períodos, classificações ou relações causais.\n"
    "- Se a resposta não estiver disponível, diga: 'A informação não consta nos documentos fornecidos.'\n"
    "- Se houver evidência parcial, responda apenas com o que está documentado e explicite a limitação.\n"
    "- Quando citar dados, mantenha o sentido original e preserve o recorte temporal, geográfico ou setorial.\n"
    "- Evite linguagem vaga. Seja específico.\n\n"
    "Estilo da resposta:\n"
    "- Linguagem clara, direta e acessível.\n"
    "- Tom analítico e profissional.\n"
    "- Destaque números e tendências relevantes.\n"
    "- Contextualize apenas com base nas fontes.\n"
    "- Aponte implicações práticas apenas quando sustentadas pelos dados.\n\n"
    "Documentos de referência:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Pergunta: {query_str}\n\n"
    "Resposta:"
)

def setup_llm():
    """Configura o gerador OpenAI (GPT-4.1) com temperatura 0 para respostas factuais e precisas."""
    llm = OpenAI(model="gpt-4.1", temperature=0.0)
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
        similarity_top_k=30,
    )

    # 2. tree_summarize sintetiza respostas hierarquicamente sobre múltiplos chunks/documentos
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT,
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
