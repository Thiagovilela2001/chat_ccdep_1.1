# Rodando o RAG CCDEP com Docker

## 1. Verifique o `.env`

Na raiz do projeto, mantenha um arquivo `.env` com:

```env
OPENAI_API_KEY=sua_chave_aqui
```

Esse arquivo não é copiado para a imagem. O `docker-compose.yml` injeta as variáveis apenas quando o container sobe.

## 2. Tipos de RAG expostos

O Compose sobe quatro APIs. O frontend fica no RAG principal e permite alternar entre elas:

| Serviço | Porta | Uso no frontend |
|---|---:|---|
| `rag-principal` | `8000` | RAG Principal |
| `rag-agentic` | `8001` | Agentic RAG |
| `rag-raptor` | `8002` | RAPTOR RAG |
| `rag-selfrag` | `8003` | Self-RAG |

## 3. Estrutura usada pelos volumes

O container lê os dados e persiste índices nestes caminhos:

| Pasta local | Caminho no container | Uso |
|---|---|---|
| `./data` | `/app/rag_principal/data` | PDFs e bases de entrada |
| `./rag_principal/chroma_db` | `/app/rag_principal/chroma_db` | Banco vetorial Chroma |
| `./rag_agentic/chroma_db` | `/app/rag_agentic/chroma_db` | Chroma do Agentic RAG |
| `./rag_raptor/chroma_db` | `/app/rag_raptor/chroma_db` | Chroma do RAPTOR RAG |
| `./rag_selfrag/chroma_db` | `/app/rag_selfrag/chroma_db` | Chroma do Self-RAG |
| `./rag_principal/graph_store` | `/app/rag_principal/graph_store` | Grafo, quando usado |
| `hf_cache` | `/cache/huggingface` | Cache do modelo de embeddings |

## 4. Suba a aplicação

```powershell
docker compose up --build
```

A primeira execução baixa o índice padrão já processado para Principal, Agentic
e Self-RAG. Cada engine mantém cópia isolada para evitar acesso concorrente ao
mesmo ChromaDB. RAPTOR mantém índice hierárquico próprio e pode demorar na
primeira construção.

## 5. Acesse

Frontend:

```text
http://127.0.0.1:8000/app/
```

Documentação das APIs:

```text
http://127.0.0.1:8000/docs  RAG Principal
http://127.0.0.1:8001/docs  Agentic RAG
http://127.0.0.1:8002/docs  RAPTOR RAG
http://127.0.0.1:8003/docs  Self-RAG
```

Health check:

```text
http://127.0.0.1:8000/health
```

Para subir apenas um serviço durante desenvolvimento:

```powershell
docker compose up --build rag-principal
docker compose up --build rag-agentic
docker compose up --build rag-raptor
docker compose up --build rag-selfrag
```

## Comandos úteis

Parar:

```powershell
docker compose down
```

Ver logs:

```powershell
docker compose logs -f
```

Rebuild limpo:

```powershell
docker compose build --no-cache
docker compose up
```
