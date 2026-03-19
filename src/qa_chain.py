import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.retrievers.bm25 import BM25Retriever

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

def get_query_engine(index, nodes=None):
    """
    Constrói a engine de perguntas e respostas com retrieval híbrido (Vector + BM25).
    - Vector: captura similaridade semântica
    - BM25: captura termos exatos (anos, indicadores, valores numéricos)
    - Fusão via Reciprocal Rank Fusion (RRF)
    """
    setup_llm()

    # 1. Retriever denso (embeddings / similaridade semântica)
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=30,
    )

    # 2. Retriever esparso (BM25 / termos exatos)
    if nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=30,
        )
        # 3. Fusão híbrida via Reciprocal Rank Fusion
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=30,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        print("  Modo: Retrieval Híbrido (Vector + BM25)")
    else:
        retriever = vector_retriever
        print("  Modo: Retrieval Vetorial (BM25 indisponível — nodes não encontrados)")

    # 4. tree_summarize sintetiza respostas hierarquicamente sobre múltiplos chunks/documentos
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
