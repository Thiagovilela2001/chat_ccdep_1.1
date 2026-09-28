# Medições

Artefatos crus de medição da engine Principal, movidos da raiz do repositório para
cá (o código não os referenciava). Nada aqui é dado de produção: são retratos de uma
execução, com carimbo de tempo e commit gravados dentro do próprio arquivo.

## Probes de regressão (`scripts/probe_rag_health.py`)

Cada arquivo é a saída do script, que pergunta uma questão por domínio e guarda
latência, fontes, validação numérica e a resposta completa.

```bash
python scripts/probe_rag_health.py <porta> <dominios> medicoes/<saida>.json
# exemplo: uma rodada completa dos seis domínios contra a engine na 8100
python scripts/probe_rag_health.py 8100 "" medicoes/probe_results.json
```

| Arquivo | Domínios cobertos | Observação |
|---|---|---|
| `probe_results.json` | os 6 (inclui `out_of_domain`) | rodada completa, 0 erros |
| `probe_regress.json` | `labor_market`, `demography` | 0 erros |
| `probe_regress_batch.json` | `labor_market`, `economic_conjuncture` | 0 erros |
| `probe_regress_final.json` | `labor_market`, `economic_conjuncture` | 0 erros |
| `probe_regress_rerank.json` | `labor_market`, `economic_conjuncture` | 0 erros |
| `probe_regress_nometa.json` | `labor_market`, `economic_conjuncture` | 1 erro registrado no JSON |

## Baseline RAGAS (`evaluate.py --split dev --seed 42`, judge Maritaca `sabia-4`)

| Arquivo | Commit medido | Faithfulness | Context precision | Context recall | Precisão numérica | p50 / p95 |
|---|---|---:|---:|---:|---:|---:|
| `probe_ragas_baseline_8c988ec.json` | `8c988ec` | 0,952 | 0,821 | 0,809 | 0,838 | 24,3 s / 35,7 s |
| `probe_ragas_head.json` | `21d48ef` | 0,823 | 0,543 | 0,724 | 0,946 | 30,1 s / 96,5 s |

As duas execuções são de 19/09/2026 (15:05 e 15:47 UTC) e são o par que documenta a
queda de recuperação entre `8c988ec` e `21d48ef` — útil como referência, não como
gate: nada no CI compara com esses números.

Os JSONs guardam também `details` por pergunta e o bloco `run` (seed, commit, versão
de Python e de `ragas`/`datasets`), o que permite reproduzir o contexto da medição.
