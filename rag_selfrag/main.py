"""
Entry point do Self-RAG.

Modos de uso:
    python main.py              # inicia servidor FastAPI em :8003
    python main.py --port 9003  # porta customizada
    python main.py --cli        # loop interativo (sem servidor HTTP)
"""
import sys
import os
import argparse
import src  # noqa: F401 — bootstrap: torna rag_core/ importável

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _run_server(host: str, port: int) -> None:
    import uvicorn
    from rag_core.logger import setup_logging
    setup_logging()
    print(f"Iniciando Self-RAG em http://{host}:{port}")
    print(f"Documentacao interativa: http://127.0.0.1:{port}/docs\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


def _run_cli() -> None:
    import asyncio
    from dotenv import load_dotenv
    from src.startup import initialize
    from src.query_interpreter import interpret_query
    from rag_core.numerical_validator import validate_numbers, format_validation_report

    load_dotenv()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine, interp_llm = initialize(base_dir)

    print("=" * 52)
    print(" SELF-RAG PRONTO — modo CLI interativo")
    print(" (RETRIEVE? → ISREL → GENERATE → ISSUP)")
    print(" Digite 'sair' para encerrar.")
    print("=" * 52 + "\n")

    while True:
        try:
            pergunta = input("Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit", "q"}:
            print("Encerrando.")
            break

        try:
            interp = interpret_query(pergunta, interp_llm)
            print(f"  is_labor_market: {interp['is_labor_market']}\n")
            print("Self-RAG avaliando e buscando...\n")

            resposta, source_nodes = asyncio.run(
                engine.answer(
                    question=pergunta,
                    sources=interp["sources"],
                    rewritten_query=interp["rewritten_query"],
                    is_labor_market=interp.get("is_labor_market", False),
                )
            )

            print(f"Resposta:\n{resposta}\n")

            print("Validação numérica:")
            checks = validate_numbers(resposta, source_nodes)
            print(format_validation_report(checks))

            print("\nReferências:")
            for i, node in enumerate(source_nodes):
                fname = node.metadata.get("source_file") or node.metadata.get("file_name", "?")
                score = round((node.score or 0) / 10.0, 2)
                print(f"  [{i+1}] {fname} (relevância: {score:.2f})")

        except Exception as exc:
            print(f"\n[ERRO] {exc}\n")

        print("\n" + "-" * 40 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-RAG Estatístico SP")
    parser.add_argument("--cli",  action="store_true", help="Loop interativo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    if args.cli:
        _run_cli()
    else:
        _run_server(args.host, args.port)


if __name__ == "__main__":
    main()
