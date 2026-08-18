"""Adaptador de interpretação para o Agentic RAG."""
from rag_ccdep.core.query_interpreter import interpret_all_sources


def interpret_query(question: str, llm) -> dict:
    return interpret_all_sources(question, llm, engine_name="agentic")
