# rag_orchestrator — Meta RAG (orquestrador inteligente)

Camada de decisão sobre as engines RAG existentes. Analisa cada pergunta,
**seleciona dinamicamente a melhor estratégia de recuperação** (single-best por
padrão) e encaminha a consulta para o `/query` da engine escolhida. Não altera
nenhuma engine e **não importa `transformers`** — apenas encaminha via HTTP.

## Arquitetura

| Arquivo | Responsabilidade |
|---|---|
| `src/registry.py` | Perfis, cliente HTTP autenticado e circuit breaker das engines |
| `src/query_analyzer.py` | Classificação semântica da consulta (LLM + fallback) |
| `src/router.py` | Política de seleção (scoring de perfis) — função pura |
| `src/fusion.py` | Execução multi-engine opcional + seleção da melhor resposta |
| `src/quality_gate.py` | Recusa + validação numérica reaproveitada das engines |
| `src/orchestrator.py` | Pipeline: analisar → rotear → health/failover → montar envelope |
| `src/api.py` | FastAPI `:8010` — `POST /query`, `POST /route`, `GET /health` |
| `main.py` | Entrypoint: servidor, `--cli` ou `--route "pergunta"` |

## Como rodar

```bash
# 1. Suba as engines-alvo (precisam do ambiente com transformers<5):
docker-compose up          # sobe principal:8000, agentic:8001, raptor:8002, selfrag:8003
#   ou individualmente:  cd rag_principal && python main.py --port 8000

# 2. Suba o orquestrador (roda no ambiente atual, sem transformers):
cd rag_orchestrator && python main.py --port 8010

# 3. Use o chat React:  cd frontend && npm run dev
```

Inspeção sem executar engine (só a decisão de roteamento):
```bash
python main.py --route "Faça um panorama da economia paulista em 2024"
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

O analyzer usa o provedor central de `rag_core/llm.py` (`MARITACA_API_KEY`,
`OPENAI_API_KEY` ou `RAG_LLM_API_KEY`). Se ele falhar, aplica o fallback heurístico.

## Adicionar uma nova estratégia (extensibilidade)

Basta acrescentar **uma** entrada em `STRATEGIES` (`src/registry.py`) com o perfil
da nova engine e sua URL. O roteador passa a considerá-la automaticamente —
nenhum outro arquivo muda.

## Testes

```bash
python tests/test_router.py     # lógica de roteamento (sem rede/LLM)
python tests/test_pipeline.py   # pipeline completo com mocks
```
