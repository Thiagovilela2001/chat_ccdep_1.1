"""Probe de saúde do RAG Principal via HTTP.

Uso: python probe_rag_health.py [porta] [dominios] [saida]
     python probe_rag_health.py 8000 labor_market,demography probe_regress.json
Escreve o JSON de resultados e imprime resumo compacto.
"""
import json
import sys
import time
import urllib.error
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else "probe_results.json"
URL = f"http://127.0.0.1:{PORT}/query"

QUESTIONS = [
    ("labor_market", "Qual foi a taxa de desocupacao e o rendimento medio real no estado de Sao Paulo no trimestre mais recente divulgado pela PNAD Continua?"),
    ("economic_conjuncture", "Como o PIB do estado de Sao Paulo variou nos ultimos trimestres divulgados?"),
    ("social_protection", "Quantas familias sao beneficiarias do Bolsa Familia em Sao Paulo e qual o valor medio do beneficio?"),
    ("investment_trade", "Quais sao os principais produtos exportados por Sao Paulo e qual foi o saldo da balanca comercial paulista?"),
    ("demography", "Qual e a populacao do estado de Sao Paulo e como esta o envelhecimento populacional?"),
    ("out_of_domain", "Quem ganhou a Copa do Mundo de futebol de 2022 e qual foi o placar da final?"),
]


def ask(question: str, timeout: int = 240):
    data = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read().decode("utf-8"))
        return round(time.monotonic() - t0, 1), payload, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        return round(time.monotonic() - t0, 1), None, f"HTTP {exc.code}: {body}"
    except Exception as exc:  # noqa: BLE001
        return round(time.monotonic() - t0, 1), None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    questions = [item for item in QUESTIONS if ONLY is None or item[0] in ONLY]
    results = []
    for domain, question in questions:
        latency, payload, error = ask(question)
        record = {
            "domain": domain,
            "question": question,
            "latency_s": latency,
            "error": error,
            "payload": payload,
        }
        results.append(record)

        print("=" * 78)
        print(f"[{domain}] {question}")
        print(f"latencia: {latency}s")
        if error:
            print(f"ERRO: {error}")
            continue
        print(f"fontes_usadas: {payload.get('sources_used')}")
        print(f"query_reescrita: {payload.get('rewritten_query')}")
        v = payload.get("validation", {})
        cv = payload.get("citation_validation", {})
        print(f"validacao_numerica: {v.get('verified')}/{v.get('total')} | citacoes: {cv.get('verified')}/{cv.get('total')}")
        if v.get("unverified"):
            print(f"nao_verificados: {v['unverified'][:5]}")
        if cv.get("unverified"):
            print(f"citacoes_sem_suporte: {cv['unverified'][:3]}")
        print(f"n_fontes: {len(payload.get('sources', []))} | n_citacoes_numericas: {len(payload.get('numeric_citations', []))}")
        print("-" * 78)
        print(payload.get("answer"))
        print()

    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"Resultados completos em {OUTPUT}")


if __name__ == "__main__":
    main()
