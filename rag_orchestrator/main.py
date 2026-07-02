"""
Entry point do Meta RAG (orquestrador inteligente).

Modos de uso:
    python main.py                 # servidor FastAPI em :8010
    python main.py --port 9010     # porta customizada
    python main.py --cli           # loop interativo (mostra rota + resposta)
    python main.py --route "..."    # só a decisão de roteamento de uma pergunta

Pré-requisito: as engines-alvo devem estar no ar (ex.: docker-compose up, ou
`cd rag_principal && python main.py --port 8000`). O orquestrador NÃO importa
transformers — apenas encaminha via HTTP.
"""
import argparse
import asyncio
import json
import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _run_server(host: str, port: int) -> None:
    import uvicorn
    print(f"Meta RAG em http://{host}:{port}  (docs: http://127.0.0.1:{port}/docs)")
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


def _run_route(question: str) -> None:
    from dotenv import load_dotenv
    from src.orchestrator import Orchestrator
    load_dotenv()
    result = asyncio.run(Orchestrator().route_only(question))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_cli() -> None:
    from dotenv import load_dotenv
    from src.orchestrator import Orchestrator
    load_dotenv()
    multi = os.getenv("ORCHESTRATOR_MULTI_ENGINE", "0") in ("1", "true", "True")
    orch = Orchestrator(multi_engine=multi)

    print("=" * 52)
    print(" META RAG — modo CLI. Digite 'sair' para encerrar.")
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
            break
        result = asyncio.run(orch.answer(pergunta))
        rota = result.get("route", {})
        print(f"\n  Rota: {rota.get('engine')} [{rota.get('mode')}] "
              f"— {rota.get('query_type')} (conf {rota.get('confidence')})")
        print(f"  {rota.get('reasoning')}\n")
        if result.get("error"):
            print(f"[ERRO] {result['error']}\n")
        else:
            print(f"Resposta:\n{result.get('answer')}\n")
        print("-" * 40 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta RAG — orquestrador")
    parser.add_argument("--cli", action="store_true", help="Loop interativo")
    parser.add_argument("--route", metavar="PERGUNTA", help="Só a decisão de roteamento")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if args.route:
        _run_route(args.route)
    elif args.cli:
        _run_cli()
    else:
        _run_server(args.host, args.port)


if __name__ == "__main__":
    main()
