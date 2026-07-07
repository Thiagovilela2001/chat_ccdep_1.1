"""
rag_core — infraestrutura compartilhada pelas quatro engines RAG.

Contém as versões canônicas dos módulos que antes existiam como cópias em
rag_principal/, rag_agentic/, rag_raptor/ e rag_selfrag/: ingestão, chunking,
indexação, retrievers de texto/tabelas/séries, sandbox de execução, validação
numérica, logging, skill de mercado de trabalho e segurança de API.

O que é específico de cada engine (startup.py, query_interpreter.py, api.py e
os engines próprios — agent/raptor/self-rag/graph) permanece em <engine>/src/
e importa daqui:

    from rag_core.tables_retriever import TablesRetriever

O pacote é importável porque cada <engine>/src/__init__.py adiciona o diretório
pai da engine (raiz do repo localmente; /app no Docker) ao sys.path.
"""
