"""Meta RAG — camada de orquestração sobre as engines RAG existentes.

Garante que rag_core/ (no diretório-pai — raiz do repo localmente, /app no
Docker) seja importável.
"""
import os
import sys

_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
