import json
import urllib.request

def test_query(question: str):
    print("=" * 60)
    print(f"PERGUNTA: {question}")
    print("=" * 60)
    data = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/query",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            payload = json.loads(res.read().decode("utf-8"))
            print("\n[RESPOSTA]")
            print(payload.get("answer"))
            
            print(f"\n[FONTES RECUPERADAS - Total: {len(payload.get('sources', []))}]")
            for idx, s in enumerate(payload.get("sources", []), 1):
                file_name = s.get("file")
                score = s.get("score")
                page = s.get("page")
                excerpt = s.get("excerpt", "")[:120].replace("\n", " ")
                print(f"  {idx}. [{file_name}] (score: {score:.3f}, pag: {page}) -> {excerpt}...")

            print(f"\n[CITACOES NUMERICAS VALIDADAS - Total: {len(payload.get('numeric_citations', []))}]")
            for idx, c in enumerate(payload.get("numeric_citations", []), 1):
                val = c.get("value")
                f = c.get("file")
                page = c.get("page")
                claim = c.get("claim", "")[:80].replace("\n", " ")
                exp = c.get("explanation", "")
                print(f"  {idx}. Valor: '{val}' | Arquivo: {f} (pag {page}) | Trecho: \"{claim}\"")
                if exp:
                    print(f"     Info/Explicacao: {exp}")
    except Exception as e:
        print(f"Erro ao consultar: {e}")

if __name__ == "__main__":
    test_query("Qual foi a variação do PIB do estado de São Paulo no 1º trimestre de 2024?")
