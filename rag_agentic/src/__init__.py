"""Pacote da engine. Garante que rag_core/ (no diretório-pai da engine — raiz
do repo localmente, /app no Docker) seja importável a partir de qualquer modo
de execução: main.py, uvicorn, evaluate.py."""
import os
import sys

_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
