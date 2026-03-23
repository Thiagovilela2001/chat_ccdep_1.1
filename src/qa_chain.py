import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
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
    "Citação de fontes (obrigatório):\n"
    "- Toda afirmação com dado numérico, estatística ou fato específico deve ser seguida de uma citação inline.\n"
    "- Formato: (Fonte: [nome_do_arquivo], p. [página])\n"
    "- Use exatamente o nome do arquivo e o número de página que aparecem nos metadados do contexto.\n"
    "- Se a página não estiver disponível, cite apenas o nome do arquivo.\n"
    "- Exemplo: 'O PIB cresceu 2,3% em 2022 (Fonte: Boletim_Conjuntura_3Trim.pdf, p. 14).'\n\n"
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
    - Reranker cross-encoder (bge-reranker-large): top 20 → top 5

    Retorna (query_engine, retriever, reranker) para reuso pelo CalculationEngine.
    """
    setup_llm()

    # 1. Retriever denso (embeddings / similaridade semântica)
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=20,
    )

    # 2. Retriever esparso (BM25 / termos exatos)
    if nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=20,
        )
        # 3. Fusão híbrida via Reciprocal Rank Fusion
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=20,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        print("  Modo: Retrieval Híbrido (Vector + BM25) → Reranking (bge-reranker-large) → top 5")
    else:
        retriever = vector_retriever
        print("  Modo: Retrieval Vetorial → Reranking (bge-reranker-large) → top 5")

    # 4. Reranker cross-encoder: seleciona os 5 chunks mais relevantes dos 20 recuperados
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=5,
    )

    # 5. tree_summarize sintetiza respostas hierarquicamente sobre múltiplos chunks/documentos
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )

    return query_engine, retriever, reranker

def answer_question(query_engine, question: str):
    """Encapsula a função de consulta e devolve ao script principal"""
    response = query_engine.query(question)
    return response
