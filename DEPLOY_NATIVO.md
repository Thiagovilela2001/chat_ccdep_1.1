# Deploy nativo (sem Docker)

> **Estado: procedimento escrito a partir da leitura do código e das medições desta
> sessão. NÃO foi executado em Linux — nenhum comando abaixo foi rodado num servidor.**
> Onde há incerteza, está marcado. O caminho Docker (`DOCKER.md`) é o único que o CI
> exercita hoje.

## O que esta máquina precisa ter

| Recurso | Valor | De onde vem |
|---|---|---|
| RAM por engine (com `bge-m3`) | **1,49 GB** | medido na engine rodando |
| RAM do orquestrador | baixa (não carrega embeddings) | leitura de `rag_orchestrator/src/` |
| Disco: modelo `bge-m3` | 2,2 GB | cache do Hugging Face, baixado no 1º start |
| Disco: índice | 746 MB instalados (400 MB comprimido) | baixado no 1º start |
| Disco: dependências Python | alguns GB (torch, transformers, opencv) | `pip install` |
| Start de mídia: 2,2 GB de modelo | precisa de saldo de banda | — |

Serviço mínimo para ficar online: **a engine Principal** (a orquestradora é opcional — a
SPA aceita apontar direto para a engine). Custo: ~1,49 GB de RAM.

## Passo 1 — pacotes de sistema

Lista idêntica à do `Dockerfile:10-18`, mais Python e git:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ghostscript libglib2.0-0 libgl1 poppler-utils tesseract-ocr \
  python3.11 python3.11-venv python3-pip git curl
```

**Python 3.11**, não 3.12: é a versão da imagem Docker do projeto e a do ambiente onde os
pinos foram validados (`transformers==4.46.3` está pinado de propósito — o comentário em
`requirements.txt:15-17` diz que a série 5.x quebra o ambiente). Rodar em 3.12 **não foi
testado por mim**. Se o seu Ubuntu for 24.04 (que traz 3.12), o 3.11 vem do PPA
`deadsnakes` — não verifiquei esse caminho.

## Passo 2 — código

O repositório é **público** (verificado), então clona sem autenticação:

```bash
git clone https://github.com/Thiagovilela2001/chat_ccdep_1.1.git
cd chat_ccdep_1.1
```

## Passo 3 — ambiente Python

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Baixa torch, transformers, sentence-transformers, chromadb e o `camelot-py[cv]` (OpenCV).
Conta vários minutos e alguns GB. **Não medi o tempo.**

## Passo 4 — `.env` (na raiz do projeto)

O `.env` está no `.gitignore`, então você cria à mão. O app o carrega com
`load_dotenv()` (`rag_principal/main.py:37` e `src/api.py:23`), que sobe diretórios a
partir do diretório de trabalho — por isso o arquivo na **raiz** é encontrado mesmo
rodando de dentro de `rag_principal/`.

```
MARITACA_API_KEY=<sua chave>
RAG_INDEX_READ_ONLY=1
RAG_CORS_ORIGINS=https://<seu-app>.vercel.app
RAG_RATE_LIMIT=5
RAG_RATE_WINDOW=60
# RAG_API_KEY=<opcional: restringe por chave em vez de deixar aberto>
```

- `MARITACA_API_KEY` é **obrigatória**: `startup.py:72` chama `require_api_key()` e a
  engine aborta sem ela.
- `RAG_INDEX_READ_ONLY=1` é **obrigatória no servidor sem corpus**. Sem ela, e com a pasta
  `data/` vazia, `sync_standard_index` levanta
  `Nenhum documento encontrado em '...'; índice anterior preservado` e o serviço **não
  sobe** (`rag_core/index_sync.py:89-92`). Com ela, a engine exige índice e cache BM25
  presentes e **ignora** diferenças do corpus (`index_sync.py:62-87`).

## Passo 5 — primeiro start

```bash
cd rag_principal
../.venv/bin/python main.py --host 127.0.0.1 --port 8000
```

No primeiro start, duas coisas acontecem sozinhas (`rag_core/index_bootstrap.py`):

1. baixa e valida `rag-principal-index-v4.tar.gz` (tag `vector-index-v4`, 400 MB) —
   dispensado se já houver índice local válido;
2. o `bge-m3` (2,2 GB) é baixado para o cache do Hugging Face.

Espere os dois downloads antes de concluir que travou. **Não medi esse tempo.**

`--host 127.0.0.1` é proposital: quem fica público é o proxy/túnel, não a engine. A imagem
Docker usa `0.0.0.0` porque lá dentro o proxy é outro contêiner.

## Passo 6 — manter de pé (systemd)

`/etc/systemd/system/rag-principal.service`:

```ini
[Unit]
Description=RAG Principal (chat_ccdep)
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/chat_ccdep_1.1/rag_principal
EnvironmentFile=/home/ubuntu/chat_ccdep_1.1/.env
ExecStart=/home/ubuntu/chat_ccdep_1.1/.venv/bin/python main.py --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-principal
journalctl -u rag-principal -f
```

`EnvironmentFile` é redundância: o app lê o `.env` sozinho, mas assim as variáveis também
existem no ambiente do serviço (útil para depurar). `TimeoutStartSec=1800` porque o
primeiro start baixa modelo e índice.

## Passo 7 — HTTPS sem domínio

Página em `https://` não pode chamar `http://` (o navegador bloqueia). Sem domínio:

```bash
# ARM (Ampere): troque amd64 por arm64
curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/
cloudflared tunnel --url http://127.0.0.1:8000
```

Ele imprime uma URL `https://<aleatorio>.trycloudflare.com`. **A URL muda a cada
reinício** — é modo de teste. Endereço fixo sem domínio: Tailscale Funnel
(`tailscale funnel 8000`).

## Passo 8 — apontar a Vercel

Sem configuração, a SPA publicada chama `https://<seu-app>.vercel.app:<porta>` — ela mesma
(`frontend/src/lib/api.js:39-46`). Duas saídas: definir `VITE_META_URL` (e afins) no build
na Vercel, ou usar a tela de Configurações do app, que grava no `localStorage` e tem
precedência (`api.js:53`). A segunda não exige rebuild.

## Passo 9 — o que prova que subiu

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health   # 200
```

Isso **não** prova geração. A prova é uma consulta real, porque a chamada ao LLM só
acontece no fim do pipeline:

```bash
curl -s -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
  -d '{"question":"Qual o saldo de empregos formais no estado de São Paulo?"}' | head -c 400
```

Critério: 200 **com texto**. Se vier `Falha interna ao processar a consulta`, o problema é
o provedor de LLM, não o servidor.

## Mensagens que eu já vi e o que significam

| Mensagem | Significado |
|---|---|
| `Falha interna ao processar a consulta` (500, resposta rápida) | provedor de LLM recusando — sem crédito dá `403 insufficient_funds` |
| `Índice portátil vazio em '...'; instalação incompleta` | o download do índice não completou |
| `Cache BM25 ausente em '...'` | idem: instalação incompleta |
| `Índice somente leitura: N diferença(s) no corpus ignorada(s)` | **normal** com `data/` vazia — é a comparação do corpus local com o manifesto |
| `Nenhum documento encontrado em '...'` | `RAG_INDEX_READ_ONLY` não está ligada e a `data/` está vazia |

## O que eu NÃO verifiquei

- Qualquer comando acima num servidor Linux — não tenho acesso ao seu.
- Instalação em Python 3.12 (a validada é 3.11).
- Tempo do `pip install`, do download do índice e do modelo.
- Se o `camelot-py[cv]` tem wheel para ARM — em x86 ele instala; em ARM pode compilar.
  **Teste o `pip install` como primeira coisa**, não como último passo.
- O `cloudflared`/Tailscale no alvo, e o comportamento do túnel sob uso.
