# RAG Estatístico SP — Instruções para Claude Code

## Skill automática: Mercado de Trabalho

Sempre que o usuário fizer uma pergunta ou pedir uma tarefa relacionada a
**mercado de trabalho**, leia e aplique a skill antes de responder:

    .agents/skills/labor_market_analysis/SKILL.md

Tópicos que ativam a skill (exemplos):
- Emprego, desemprego, desocupação, informalidade
- CAGED, RAIS, PNAD Contínua (PNADC), PED
- Salários, remuneração, rendimento do trabalho
- Taxa de desocupação, saldo de empregos, rotatividade
- Mercado de trabalho formal/informal

## Arquitetura do projeto

| Módulo | Função |
|---|---|
| `src/startup.py` | Inicialização: indexação, LLMs, retrievers, AnalysisEngine |
| `src/query_interpreter.py` | Roteia query para fontes + detecta `is_labor_market` |
| `src/analysis_engine.py` | Orquestra retrievers em paralelo e sintetiza resposta |
| `src/text_retriever.py` | Retrieval híbrido (Vector + BM25) para texto narrativo |
| `src/tables_retriever.py` | Extração de dados de tabelas estáticas via pandas |
| `src/timeseries_retriever.py` | Extração e análise de séries temporais via pandas |
| `src/labor_market_skill.py` | Carrega a skill e detecta queries de mercado de trabalho |
| `src/api.py` | FastAPI: endpoint POST /query |
| `main.py` | Entrypoint: servidor HTTP ou CLI interativo |

## Dados indexados

Boletins de Conjuntura Paulista (PDFs, 2022–2025) em `/data/`.
Banco vetorial: ChromaDB em `/chroma_db/`.

## Avaliação

`python evaluate.py --split dev|test|adversarial|all`
Métricas: Faithfulness, ContextPrecision, ContextRecall (RAGAS) + refusal_accuracy.
