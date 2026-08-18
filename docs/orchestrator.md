# rag_orchestrator — Meta RAG (orquestrador inteligente)

Camada de decisão sobre as engines RAG existentes. Analisa cada pergunta,
**seleciona dinamicamente a melhor estratégia de recuperação** (single-best por
padrão) e encaminha a consulta para o `/query` da engine escolhida. Não altera
nenhuma engine e **não importa `transformers`** — apenas encaminha via HTTP.

## Arquitetura

| Arquivo | Responsabilidade |
|---|---|
| `src/rag_ccdep/orchestrator/registry.py` | Perfis, cliente HTTP autenticado e circuit breaker das engines |
| `src/rag_ccdep/orchestrator/query_analyzer.py` | Classificação semântica da consulta (LLM + fallback) |
| `src/rag_ccdep/orchestrator/router.py` | Política de seleção (scoring de perfis) — função pura |
| `src/rag_ccdep/orchestrator/fusion.py` | Execução multi-engine opcional + seleção da melhor resposta |
| `src/rag_ccdep/orchestrator/quality_gate.py` | Recusa + validação numérica reaproveitada das engines |
| `src/rag_ccdep/orchestrator/orchestrator.py` | Pipeline: analisar → rotear → health/failover → montar envelope |
| `src/rag_ccdep/orchestrator/api.py` | FastAPI `:8010` — `POST /query`, `POST /route`, `GET /health` |
| `rag-orchestrator` | Entrypoint: servidor, `--cli` ou `--route "pergunta"` |

## Como rodar

```bash
# 1. Suba as engines necessárias, cada uma em seu terminal:
python -m rag_ccdep.cli principal --port 8000
python -m rag_ccdep.cli agentic --port 8001
python -m rag_ccdep.cli raptor --port 8002
python -m rag_ccdep.cli selfrag --port 8003

# 2. Suba o orquestrador (roda no ambiente atual, sem transformers):
python -m rag_ccdep.cli orchestrator --port 8010

# 3. Use o chat React:  cd frontend && npm run dev
```

Inspeção sem executar engine (só a decisão de roteamento):
```bash
rag-orchestrator --route "Faça um panorama da economia paulista em 2024"
```

## Configuração (variáveis de ambiente)

| Var | Padrão | Efeito |
|---|---|---|
| `RAG_PRINCIPAL_URL` … `RAG_SELFRAG_URL` | `localhost:8000..8003` | URLs dos backends |
| `ORCHESTRATOR_MULTI_ENGINE` | `0` | `1` habilita execução multi-engine em ambiguidade |
| `ORCHESTRATOR_ANALYZER_MODEL` | `RAG_INTERP_MODEL` (`sabia-4`) | modelo da análise semântica |
| `RAG_BACKEND_API_KEY` | — | autentica chamadas do orquestrador para as engines |
| `RAG_REQUEST_TIMEOUT` | `180` | deadline total da consulta |
| `RAG_CIRCUIT_FAILURE_THRESHOLD` | `3` | falhas antes de abrir o circuito |
| `RAG_CIRCUIT_RECOVERY_SECONDS` | `30` | intervalo para nova tentativa |

O analyzer usa o provedor central de `src/rag_ccdep/core/llm.py` (`MARITACA_API_KEY`,
`OPENAI_API_KEY` ou `RAG_LLM_API_KEY`). Se ele falhar, aplica o fallback heurístico.

## Adicionar uma nova estratégia (extensibilidade)

Basta acrescentar **uma** entrada em `STRATEGIES` (`src/rag_ccdep/orchestrator/registry.py`) com o perfil
da nova engine e sua URL. O roteador passa a considerá-la automaticamente —
nenhum outro arquivo muda.

## Testes

```bash
python -m pytest tests/integration/test_router.py -q
python -m pytest tests/integration/test_pipeline.py -q
```
