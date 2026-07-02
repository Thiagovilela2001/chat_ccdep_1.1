# meta_rag_ui — Frontend Streamlit do Meta RAG

Interface modular e desacoplada para o orquestrador Meta RAG. **Apenas consome**
o backend (`/query`, `/route`, `/health`): não roteia, não classifica, não
recupera documentos, não acessa o Chroma.

## Estrutura

```
meta_rag_ui/
├── app.py                       # entrypoint (wiring)
├── services/
│   └── api.py                   # MetaRagClient + ApiError (única porta ao backend)
├── utils/
│   ├── session.py               # histórico via st.session_state
│   └── formatting.py            # formatação pura (tempos, confiança, JSON)
└── components/
    ├── chat.py                  # campo de pergunta (botão + Enter) + histórico
    ├── sidebar.py               # conexão, sistema, engines, métricas
    ├── metrics.py               # badge inline + painel de métricas
    └── developer_panel.py       # modo desenvolvedor (QueryProfile, scores, tempos…)
```

## Como rodar

```bash
# 1. Backend no ar: engines-alvo + orquestrador
docker-compose up
cd rag_orchestrator && python main.py --port 8010

# 2. Frontend
streamlit run meta_rag_ui/app.py
```

Requer `streamlit` e `requests` (já presentes no ambiente).

## O que a interface mostra

- **Página principal:** campo de pergunta em destaque, botão "Perguntar" (Enter
  também envia), resposta em Markdown, spinner de carregamento, histórico da
  conversa.
- **Sidebar:** estado do sistema, engines registradas e disponibilidade, engine
  selecionada pelo Router, confiança, tempos (total/analyzer/router/engine),
  nº de documentos e modelos (LLM/embeddings).
- **Modo Desenvolvedor:** QueryProfile, EngineProfile, score por engine, motivo
  da escolha, estratégia de recuperação, documentos + metadados e tempo detalhado
  por etapa. Campos ainda não expostos pela API (prompt final, logs internos das
  engines) são sinalizados como indisponíveis — nunca inventados.

## Testes

```bash
python meta_rag_ui/tests/test_apptest.py   # renderização via streamlit AppTest
```
