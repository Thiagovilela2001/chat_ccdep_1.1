# AGENTS.md — Vibe Coding Guidelines

> Diretrizes de desenvolvimento e memória operacional para agentes de IA (Antigravity, Claude Code, Cursor, Codex). Mantenha este arquivo conciso e de alto sinal.

## Behavioral Guidelines

1. **Think before coding** — Formule e declare premissas antes de editar. Se existirem múltiplas interpretações plausíveis, apresente-as em vez de escolher silenciosamente. Esclareça a intenção antes de codificar.
2. **Simplicity first (Ponytail / YAGNI)** — Aplique a escada de decisão de 7 passos: código mínimo viável. Reutilize o que já existe no repositório (`rag_core`), use stdlib e recursos nativos antes de introduzir abstrações especulativas ou novas dependências.
3. **Surgical changes** — Altere apenas o que o pedido exige, na camada mais estreita e responsável. Mantenha estilo, preserve comentários e código adjacente.
4. **Goal-driven execution** — Transforme tarefas em metas com checagens de verificação executáveis (comando, teste ou asserção) antes de dar o passo como concluído.
5. **Orchestrator, not implementer** — A sessão principal planeja, orquestra e decide; trabalho delegável roda em ondas paralelas com subagentes especialistas, sem colisão de arquivos e com commit exclusivo do orquestrador (ver regra em `.agents/rules/parallel-subagent-driven-development.md`).
6. **High-signal communication (Caveman)** — Respostas diretas ao ponto, sem preâmbulos decorativos, preservando rigor técnico, código, números e mensagens exatas de erro intactos.

## Canonical Commands

- **Testes Unitários:** `pytest` (ou `pytest rag_principal/tests/`)
- **Benchmark RAG (RAGAS):** `python evaluate.py --split dev|test|adversarial|all`
- **Servidor Engine:** `python <engine>/main.py` (ex.: `python rag_principal/main.py`)
- **Orquestrador Meta-RAG:** `python rag_orchestrator/main.py`
- **Docker Compose:** `docker-compose up -d`

## Specialist Agent Routing Table

| Agente | Domínio / Quando Acionar |
|---|---|
| `orchestrator` | Planejamento de tarefas complexas, desdobramento em ondas paralelas, revisão cruzada e commits. |
| `rag-architect` | Modelos RAG (`rag_principal`, `rag_agentic`, `rag_raptor`, `rag_selfrag`), retrievers híbridos, ChromaDB, LlamaIndex e LLM factory (`rag_core/llm.py`). |
| `data-engineer` | Ingestão de PDFs, processamento tabular/séries temporais (`tables_retriever.py`, `timeseries_retriever.py`), validação numérica e chunking. |
| `backend-specialist` | Endpoints FastAPI (`api.py`), modelos Pydantic (`api_models.py`), middleware de segurança, CORS e rate limiting (`api_security.py`). |
| `test-engineer` | Testes automatizados (pytest), suítes de regressão, benchmarks de fidelidade/relevância (RAGAS). |
| `code-reviewer` | Revisão de diffs, conformidade de arquitetura, qualidade de código e detecção de bugs/regressões. |
