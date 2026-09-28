"""
jev_trace.py — traça UMA pergunta por toda a camada de decisão Jev do projeto.

Diferente do `jev_smoke.py` (que verifica o contrato de cada ponto isoladamente),
aqui a pergunta percorre o caminho real: classificação → roteamento → decisões
internas da engine escolhida. Serve para ver *onde* cada julgamento entra e com
que confiança ele sai.

Só a camada de decisão é exercitada: recuperação e geração rodam nas engines
(imagem Docker). O script diz explicitamente quando o trace para.

Uso:
    python scripts/jev_trace.py
    python scripts/jev_trace.py --pergunta "Qual foi o saldo do CAGED em 2024?"
    python scripts/jev_trace.py --todos     # roda as seções de todas as engines
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from rag_core import jev  # noqa: E402
from rag_orchestrator.src.jev_analyzer import (  # noqa: E402
    _CHOICE_DIMENSIONS,
    _NOUL_DIMENSIONS,
    _ROUTING_DIMENSIONS,
    JevQueryAnalyzer,
    build_questions,
    build_state,
)
from rag_orchestrator.src.router import route  # noqa: E402
from rag_principal.src.jev_interpreter import JevSourceSelector  # noqa: E402
from rag_selfrag.src.self_rag_engine import _JEV_RETRIEVE  # noqa: E402

_PORTA_PADRAO = {
    "principal": "8000",
    "agentic": "8001",
    "raptor": "8002",
    "selfrag": "8003",
}


def _endpoint(engine: str) -> str:
    """
    URL de `/query` da engine, como o próprio orquestrador resolve: `RAG_<X>_URL`
    (o .env a define quando a porta do host não é a padrão) e, na falta dela, a
    porta de `RAG_<X>_PORT`.
    """
    explicit = (os.getenv(f"RAG_{engine.upper()}_URL") or "").strip().rstrip("/")
    if explicit:
        return f"{explicit}/query"
    port = (os.getenv(f"RAG_{engine.upper()}_PORT") or _PORTA_PADRAO[engine]).strip()
    return f"http://localhost:{port}/query"


def _rule(title: str) -> None:
    print(f"\n{'─' * 78}\n{title}\n{'─' * 78}")


def _section_classification(client: jev.JevClient, question: str) -> dict:
    _rule("1. CLASSIFICAÇÃO DA CONSULTA — 10 perguntas, uma requisição")
    started = time.perf_counter()
    response = client.system_one(build_state(question), build_questions())
    elapsed = time.perf_counter() - started
    if response is None:
        print("  a Jev não respondeu — o pipeline cairia no analisador LLM")
        return {}

    print(f"  modelo servido: {response.model} | {elapsed:.2f}s | usage: {response.usage or '—'}")
    print(f"\n  {'dimensão':18} {'valor':16} {'peso':>6} {'decisão':>8}  uso")
    for key in (*_CHOICE_DIMENSIONS, *_NOUL_DIMENSIONS):
        answer = response.answer(key)
        if answer is None:
            print(f"  {key:18} {'(sem resposta)':16}")
            continue
        used = "roteador" if key in _ROUTING_DIMENSIONS else "—"
        if answer.type == "choice":
            value = str(answer.choice)
            weight = f"{answer.probability(answer.choice):.2f}"
        else:
            value = f"p={answer.noul:.2f}"
            weight = "—"
        print(f"  {key:18} {value:16} {weight:>6} {answer.decisiveness():8.2f}  {used}")

    aggregate = response.weakest_decisiveness(_ROUTING_DIMENSIONS)
    binding = min(
        (key for key in _ROUTING_DIMENSIONS if key in response.answers),
        key=lambda key: response.decisiveness(key),
        default="—",
    )
    print(
        f"\n  confiança agregada (menor grau de decisão entre as que o roteador usa): "
        f"{aggregate:.2f} — limitada por '{binding}'"
    )
    print("  as dimensões marcadas '—' são observabilidade: não mudam rota nem confiança")
    return {key: response.answer(key) for key in response.answers}


def _section_route(analyzer: JevQueryAnalyzer, question: str) -> tuple[dict, object]:
    _rule("2. ROTEAMENTO — função pura, sem rede")
    analysis = analyzer.analyze(question)
    decision = route(analysis)
    print(f"  classificação usada pela rota: {analysis['reasoning']}")
    print(f"  modo: {decision.mode} | engines: {decision.engines or '— nenhuma (recusa)'}")
    print(f"  confiança da decisão: {decision.confidence}")
    if decision.scores:
        ranking = sorted(decision.scores.items(), key=lambda item: item[1], reverse=True)
        print("  pontuação: " + " | ".join(f"{key}={score:.1f}" for key, score in ranking))
    print(f"  {decision.reasoning}")
    return analysis, decision


def _section_principal(client: jev.JevClient, question: str) -> None:
    _rule("3. RAG PRINCIPAL — seleção de fontes (3 Noul, uma requisição)")
    started = time.perf_counter()
    choice = JevSourceSelector(client=client).select(question)
    elapsed = time.perf_counter() - started
    if choice is None:
        print("  a Jev não respondeu — as fontes viriam do JSON do LLM (como antes)")
        return
    print(f"  fontes: {choice['sources']} | {elapsed:.2f}s")
    print(f"  {choice['reasoning']}")
    print(f"  confiança do julgamento: {choice['confidence']} (o menor entre as três fontes)")
    print(f"  → o analysis_engine rodaria os retrievers: {', '.join(choice['sources'])}")


def _section_selfrag(client: jev.JevClient, question: str) -> None:
    _rule("4. SELF-RAG — RETRIEVE? (1 Noul)")
    response = client.system_one({"pergunta": question}, {"precisa_busca": jev.noul(_JEV_RETRIEVE)})
    if response is None:
        print("  a Jev não respondeu — o self-RAG cairia no prompt LLM original")
        return
    probability = response.noul("precisa_busca")
    print(f"  precisa buscar? p={probability} → {response.noul_decision('precisa_busca')}")
    print(f"  limiar em uso: RAG_JEV_THRESHOLD={jev.decision_threshold()}")
    print("  ISREL/ISSUP só fazem sentido com trechos recuperados — param aqui, nesta máquina")


def _section_next_steps(decision) -> None:
    _rule("5. O QUE O PIPELINE FARIA EM SEGUIDA")
    if not decision.engines:
        print("  nenhuma engine é chamada: a consulta é recusada antes da recuperação.")
        return
    for engine in decision.engines:
        print(f"  POST {_endpoint(engine)}  {{\"question\": ...}}")
    print("  aqui a recuperação e a geração não rodam: elas vivem na imagem Docker.")
    print("  para exercitar o caminho completo: docker compose up -d && "
          "python scripts/jev_trace.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Traça uma pergunta pela camada de decisão Jev.")
    parser.add_argument(
        "--pergunta",
        default="Quantos empregos formais foram criados no Estado de São Paulo em 2024?",
    )
    parser.add_argument("--todos", action="store_true", help="roda as seções de todas as engines")
    args = parser.parse_args()

    load_dotenv(_ROOT / ".env")
    if not jev.jev_enabled():
        print("Jev desativada: defina TYPESAFE_API_KEY (ou JEV_API_KEY) no .env.")
        return 1

    client = jev.JevClient()
    print(f"pergunta: {args.pergunta}")
    print(f"endpoint: {client.endpoint} | modelo: {client.model} | "
          f"limiar Noul: {jev.decision_threshold()}")

    _section_classification(client, args.pergunta)
    analyzer = JevQueryAnalyzer(client=client)
    _, decision = _section_route(analyzer, args.pergunta)

    engines = set(decision.engines) if not args.todos else {"principal", "selfrag"}
    if "principal" in engines:
        _section_principal(client, args.pergunta)
    if "selfrag" in engines:
        _section_selfrag(client, args.pergunta)

    _section_next_steps(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
