from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import LLMRerank
from llama_index.retrievers.bm25 import BM25Retriever

REWRITE_PROMPT = (
    "Você é um especialista em recuperação de informações econômicas sobre o Estado de São Paulo.\n"
    "Reescreva a pergunta abaixo para maximizar a precisão na busca em documentos de conjuntura econômica.\n"
    "Regras:\n"
    "- Seja específico sobre períodos (trimestre, ano), indicadores e setores mencionados.\n"
    "- Expanda siglas relevantes (ex: PIB, PNAD, IPCA).\n"
    "- Mantenha o significado original.\n"
    "- Retorne APENAS a pergunta reescrita, sem explicações.\n\n"
    "Pergunta: {question}\n"
    "Reescrita:"
)

QA_PROMPT = PromptTemplate(
    "Você é um analista especialista em dados econômicos e estatísticos. "
    "Responda somente com base no conteúdo dos documentos fornecidos no contexto. "
    "É proibido usar conhecimento externo, preencher lacunas por inferência especulativa ou inventar qualquer informação.\n\n"
    "Regras obrigatórias:\n"
    "- Use apenas informações presentes nos documentos.\n"
    "- Não invente números, datas, nomes, períodos, classificações ou relações causais.\n"
    "- Se a resposta não estiver disponível, diga: 'A informação não consta nos documentos fornecidos.'\n"
    "- Se houver evidência parcial, responda apenas com o que está documentado e explicite a limitação.\n"
    "- Nunca substitua o tema da pergunta por um tema adjacente (ex: pergunta sobre emprego → não responda com dados de PIB setorial).\n"
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
    llm = OpenAI(model="gpt-5-chat-latest", temperature=0.0)
    Settings.llm = llm
    return llm


def rewrite_query(question: str, llm) -> str:
    response = llm.complete(REWRITE_PROMPT.format(question=question))
    return response.text.strip()


def get_query_engine(index, nodes=None):
    """
    Constrói a engine de perguntas e respostas com retrieval híbrido (Vector + BM25).
    Arquitetura:
      - Query rewrite:  gpt-5-mini   (reformula a query para melhor retrieval)
      - Retrieval:      embeddings + BM25 (fusão RRF, top 20)
      - Rerank:         gpt-5-nano   (LLMRerank, top 20 → top 5)
      - Answer:         gpt-5-chat-latest (tree_summarize com prompt restritivo)

    Retorna (query_engine, retriever, reranker, rewrite_llm).
    """
    setup_llm()
    rewrite_llm = OpenAI(model="gpt-5-mini", temperature=0.0)
    nano_llm    = OpenAI(model="gpt-5-nano",  temperature=0.0)

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
        print("  Modo: Retrieval Híbrido (Vector + BM25) → Reranking (gpt-5-nano) → top 5")
    else:
        retriever = vector_retriever
        print("  Modo: Retrieval Vetorial → Reranking (gpt-5-nano) → top 5")

    # 4. LLMRerank com gpt-5-nano: seleciona os 5 chunks mais relevantes dos 20 recuperados
    reranker = LLMRerank(
        top_n=5,
        choice_batch_size=10,
        llm=nano_llm,
    )

    # 5. tree_summarize sintetiza hierarquicamente com prompt restritivo
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=QA_PROMPT,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )

    return query_engine, retriever, reranker, rewrite_llm


def answer_question(query_engine, question: str, rewrite_llm=None):
    if rewrite_llm is not None:
        question = rewrite_query(question, rewrite_llm)
    response = query_engine.query(question)
    return response
