"""Comando único para iniciar engines e orquestrador."""

from __future__ import annotations

import argparse
import importlib
import sys

COMPONENTS = {
    "principal": "rag_ccdep.engines.principal.__main__",
    "agentic": "rag_ccdep.engines.agentic.__main__",
    "raptor": "rag_ccdep.engines.raptor.__main__",
    "selfrag": "rag_ccdep.engines.selfrag.__main__",
    "orchestrator": "rag_ccdep.orchestrator.__main__",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG CCDEP")
    parser.add_argument("component", choices=COMPONENTS)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return

    component = sys.argv[1]
    if component not in COMPONENTS:
        parser.error(f"componente inválido: {component}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    importlib.import_module(COMPONENTS[component]).main()


if __name__ == "__main__":
    main()
