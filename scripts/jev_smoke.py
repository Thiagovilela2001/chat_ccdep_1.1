"""
jev_smoke.py — verificação real da integração com a TypeSafe Jev.

Faz as chamadas à API System One com o mesmo desenho usado em produção:
  1. classificação da consulta (10 perguntas: 8 Choice + 2 Noul) — a mesma do
     `rag_orchestrator/src/jev_analyzer.py`;
  2. roteamento da classificação obtida (função pura, sem rede);
  3. julgamento de relevância de três trechos em uma única requisição — o
     fan-out do ISREL do Self-RAG;
  4. veredito tipado de suporte factual (ISSUP);
  5. seleção de fontes do rag_principal (tabelas / séries / grafo) para duas
     perguntas de perfis diferentes.

Uso:
    python scripts/jev_smoke.py
    python scripts/jev_smoke.py --pergunta "evolução do desemprego em 2024"

Requer `TYPESAFE_API_KEY` no `.env` (ou no ambiente). Sai com código 1 e a
mensagem correspondente quando a chave está ausente ou a API não responde.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from rag_core import jev  # noqa: E402
from rag_orchestrator.src.jev_analyzer import (  # noqa: E402
    build_questions,
    build_state,
    JevQueryAnalyzer,
)
from rag_orchestrator.src.router import route  # noqa: E402
from rag_principal.src.jev_interpreter import JevSourceSelector  # noqa: E402
from rag_selfrag.src.self_rag_engine import _JEV_SUPPORT_CRITERIA  # noqa: E402

_TRECHOS = [
    "A taxa de desocupação na Região Metropolitana de São Paulo ficou em 7,1% "
    "no quarto trimestre de 2024, segundo a PNAD Contínua.",
    "O cardápio do restaurante da esquina mudou no mês passado.",
    "O rendimento médio real habitual do trabalho principal subiu 3,2% no "
    "acumulado de 2024 no Estado de São Paulo.",
]


def _print_answers(title: str, response: jev.JevResponse) -> None:
    print(f"\n== {title} ==")
    print(f"modelo servido: {response.model or 'desconhecido'}")
    for key, answer in response.answers.items():
        if answer.type == "choice":
            detail = f"{answer.choice} (conf={answer.confidence})"
        elif answer.type == "score":
            detail = f"{answer.score} (conf={answer.confidence})"
        else:
            detail = f"p={answer.noul}"
        print(f"  {key}: {detail}")
        if answer.probabilities:
            ranking = sorted(answer.probabilities.items(), key=lambda kv: kv[1], reverse=True)
            distribuicao = ", ".join(f"{option}={value:.2f}" for option, value in ranking[:4])
            print(f"      distribuição: {distribuicao}")
    if response.usage:
        print(f"  usage: {response.usage}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica a integração com a TypeSafe Jev.")
    parser.add_argument(
        "--pergunta",
        default="Qual foi a evolução da taxa de desocupação em São Paulo em 2024?",
    )
    args = parser.parse_args()

    load_dotenv(_ROOT / ".env")

    if not jev.jev_enabled():
        print(
            "Jev desativada: defina TYPESAFE_API_KEY (ou o alias JEV_API_KEY) no .env "
            "e garanta que RAG_JEV_ENABLED não seja 0."
        )
        return 1

    client = jev.JevClient()
    print(f"endpoint: {client.endpoint} | modelo: {client.model}")

    # 1. Classificação da consulta (mesmas perguntas do orquestrador).
    response = client.system_one(build_state(args.pergunta), build_questions())
    if response is None:
        print("A API da TypeSafe não respondeu: verifique chave, rede e limites.")
        return 1
    _print_answers(f"classificação de: {args.pergunta[:70]}", response)

    # 2. Roteamento a partir da classificação (sem rede).
    analyzer = JevQueryAnalyzer(client=client)
    analise = analyzer.analyze(args.pergunta)
    decisao = route(analise)
    print("\n== rota ==")
    print(f"  modo: {decisao.mode} | engines: {decisao.engines}")
    print(f"  confiança: {decisao.confidence} | reasoning: {analise['reasoning']}")

    # 3. ISREL em fan-out: uma pergunta por trecho, uma requisição.
    isrel = client.system_one(
        {
            "pergunta": args.pergunta,
            "trechos": [
                {"indice": index, "texto": trecho} for index, trecho in enumerate(_TRECHOS)
            ],
        },
        {
            f"trecho_{index}": jev.noul(
                f"O `trechos[{index}].texto` ajuda a responder a `pergunta`?"
            )
            for index in range(len(_TRECHOS))
        },
    )
    if isrel is None:
        print("A API não respondeu ao fan-out de relevância (ISREL).")
        return 1
    _print_answers("relevância dos trechos (ISREL)", isrel)
    mantidos = [
        index
        for index in range(len(_TRECHOS))
        if isrel.noul_decision(f"trecho_{index}") is True
    ]
    print(f"\n  trechos mantidos pelo limiar {jev.decision_threshold()}: {mantidos}")

    # 4. ISSUP: veredito tipado de suporte factual.
    suporte = client.system_one(
        {
            "contexto": _TRECHOS[0],
            "resposta": "A taxa de desocupação em São Paulo ficou em 7,1% em 2024.",
        },
        {"suporte": jev.choice("A `resposta` é sustentada por `contexto`?", _JEV_SUPPORT_CRITERIA)},
    )
    if suporte is not None:
        print(f"\n== suporte factual (ISSUP) ==\n  {suporte.answers}")

    # 5. Seleção de fontes do rag_principal — a decisão que escolhe os retrievers.
    seletor = JevSourceSelector(client=client)
    print("\n== fontes do rag_principal ==")
    for pergunta in (
        args.pergunta,
        "Quantos empregos com carteira assinada foram criados no Estado de São Paulo em 2024?",
    ):
        escolha = seletor.select(pergunta)
        if escolha is None:
            print(f"  sem resposta da Jev para: {pergunta[:70]}")
            continue
        print(f"  {pergunta[:70]}")
        print(f"    fontes: {escolha['sources']} | confiança: {escolha['confidence']}")
        print(f"    {escolha['reasoning']}")

    print("\nOK: integração operante.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
