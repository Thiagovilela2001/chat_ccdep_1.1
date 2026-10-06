# Plano de implementação

A ideia é tratar primeiro o que quebra o CI e o que afeta a API em produção, depois o que muda a qualidade das respostas, e só no fim a documentação e a limpeza. Cada fase vira um commit separado, para facilitar revisão e reversão.

## Fase 0: Preparação (≈10 min)

1. Criar uma branch, por exemplo `fix/lint-reranker-flags`.
2. Salvar o estado atual com `git diff > /tmp/mudancas-locais.patch`, para não perder as mudanças locais caso algo dê errado.
3. Registrar a linha de base:
   - rodar `npm run lint` para confirmar os 4 erros;
   - rodar os testes do backend e anotar o resultado;
   - localizar o arquivo de resultados da avaliação de 05/10.

## Fase 1: Lint do frontend (`App.jsx`) (≈20 min)

1. **Funções sem uso.** Remover `annotateNumericCitations` e `NumericCitation`. Antes, conferir com `grep -rn "annotateNumericCitations\|NumericCitation" frontend/src` que não há outros usos. Se houver intenção de reaproveitá-las, o histórico do git as preserva, então não vale manter código morto.
2. **Parâmetro `onInspect`.** Remover o parâmetro da assinatura e dos pontos de chamada que o repassam. Se for uma prop que ainda vai ser usada em breve, renomear para `_onInspect`, mas só se o ESLint estiver configurado com `argsIgnorePattern: "^_"`.
3. **`catch {}` vazio na linha 1075.** Ver o que o `try` protege:
   - se a falha importa, tratar o erro, com fallback ou `console.warn`;
   - se for um caso de "ignorar de propósito", como um `JSON.parse` opcional, deixar um comentário explicando o motivo dentro do bloco.
4. **Validação.** `npm run lint` e `npm run build` precisam passar sem erros, e a tela que usava as citações deve ser testada manualmente.

## Fase 2: Reranker bloqueando o event loop (`analysis_engine.py:215`) (≈30 min)

1. Trocar a chamada síncrona por:

   ```python
   nodes = await asyncio.to_thread(
       self._reranker.postprocess_nodes, nodes, query_str=query
   )
   ```

   Os argumentos devem ser os mesmos da chamada atual.
2. Procurar no módulo outras chamadas síncronas pesadas dentro de funções `async`, como `retrieve`, `postprocess_nodes` e `complete`, e aplicar o mesmo tratamento.
3. **Concorrência.** Com o reranker rodando em threads, várias requisições podem chamá-lo ao mesmo tempo.
   - Confirmar que o cliente do LLM ou o modelo do reranker aguenta chamadas paralelas.
   - Se houver risco, por exemplo um cross-encoder local com uso pesado de memória ou um rate limit do provedor, limitar com um `asyncio.Semaphore`.
4. **Teste de regressão.** Criar um teste com um reranker falso que faz `time.sleep(2)`. Enquanto uma consulta roda, uma requisição ao `/health`, ou a outro endpoint leve, deve responder em menos de 2 s.
5. **Validação manual.** Subir a API com `RAG_LLM_RERANK=1`, disparar duas consultas simultâneas e verificar que elas avançam em paralelo.

## Fase 3: Flag `RAG_NUMERIC_VALIDATION` (`query_service.py`) (≈40 min)

1. Mover o `import os` para o bloco de imports no topo do arquivo.
2. **Decidir a semântica.** A recomendação é separar as responsabilidades:
   - `RAG_NUMERIC_VALIDATION` controla apenas a validação numérica;
   - as novas tentativas por recusa com fontes e por cobertura incompleta ficam sempre ligadas, ou ganham uma flag própria, como `RAG_ANSWER_RETRIES`, com padrão `1`.
3. **Centralizar a leitura.** Se o projeto já tiver um módulo de configuração, ler a variável lá. Se não, ler numa função auxiliar (`_flag_enabled("RAG_NUMERIC_VALIDATION", default=True)`) chamada em tempo de execução, e não no import, para que os testes possam alternar o valor com `monkeypatch.setenv`.
4. **Testes unitários.** Cobrir as combinações relevantes:
   - flag em `0`: a validação numérica não roda, mas as novas tentativas por recusa e por cobertura continuam acontecendo;
   - flag em `1`: tudo roda.
5. Adicionar a variável ao `.env.example`, se ele existir, com um comentário de uma linha.

## Fase 4: Reavaliar o prompt de estilo (`answer_style.py`) (≈1 h, mais o tempo de execução)

Esta fase vem depois das fases 1 a 3, para que a avaliação reflita o código que de fato vai ser commitado.

1. **Confirmar se a rodada de 05/10 é anterior à mudança.** Comparar a data do arquivo de resultados com `git log -p answer_style.py`. Como a mudança ainda é local, também vale olhar a data de modificação do arquivo.
2. **Controlar as variáveis.** Este é o ponto mais importante: se a rodada de 05/10 usou o `sabia-4` como judge e hoje o `.env` usa o DeepSeek, a comparação fica contaminada. Rodar com o mesmo judge e o mesmo provedor da rodada anterior, ou então rodar as duas versões do prompt agora, com a mesma configuração. A segunda opção é mais confiável.
3. **Executar** `python evaluate.py --split test` com o prompt novo, mantendo `RAG_NUMERIC_VALIDATION` fixo e anotado.
4. **Comparar estas métricas:**
   - nota do judge e acerto;
   - taxa de falha na validação numérica;
   - tamanho médio das respostas;
   - latência p50 e p95;
   - taxa de novas tentativas.
5. **Critério de decisão:**
   - manter o prompt novo só se a qualidade subir sem um aumento relevante de latência ou de falhas numéricas;
   - caso contrário, testar um meio-termo, como "apresente os números necessários para responder, sem transcrever tabelas inteiras", ou voltar à versão anterior.

## Fase 5: Documentação (≈20 min)

1. **CLAUDE.md:**
   - atualizar o provedor padrão e o judge;
   - documentar `RAG_NUMERIC_VALIDATION`, e `RAG_ANSWER_RETRIES` se ela for criada.
   - Antes de editar, confirmar se o DeepSeek no `.env` é o novo padrão do projeto ou só um experimento local. A documentação deve descrever o padrão do código e do `.env.example`, não o `.env` pessoal.
2. **README:** incluir as mesmas variáveis na seção de configuração.

## Fase 6: Disco e organização (≈40 min)

### ChromaDB duplicado

1. Descobrir qual caminho a engine usa, com `grep -rn "chroma_db\|persist_dir\|CHROMA" --include=*.py`, e checar também as variáveis de ambiente.
2. Confirmar que as duas cópias são idênticas, com `diff -rq chroma_db rag_principal/chroma_db`.
3. Mover a cópia sem uso para fora do projeto, rodar uma consulta de teste e só então apagá-la. Garantir que o caminho esteja no `.gitignore`.

### `.venv`

1. Recriar o ambiente com Python 3.11, para alinhar com o CI e o `pyproject`: `uv venv --python 3.11`, ou `python3.11 -m venv .venv`.
2. Reinstalar as dependências e rodar os testes.

### Arquivos soltos

1. **`scratch/`:** adicionar ao `.gitignore` ou apagar.
2. **`test_query_inspection.py`:** mover para `tests/`, se for um teste de verdade, ou para `scripts/`. Na raiz, o pytest pode coletá-lo sem querer.
3. **`count_pdf_pages.py` e `find_placeholders.py`:** mover para `scripts/`.
4. **Os dois `.pptx`:** tirar do versionamento com `git rm --cached` e guardá-los em `docs/` ou fora do repositório. Eles continuam no histórico, e só vale reescrevê-lo com `git filter-repo` se o tamanho do repositório for um problema real.
5. **Planos `.md` da raiz:** mover para `docs/plans/`.

## Checklist final antes do push

- [ ] O lint do frontend e o build passam.
- [ ] Os testes do backend passam, incluindo os novos testes de concorrência e da flag.
- [ ] A avaliação foi comparada com a mesma configuração de judge, e a decisão sobre o prompt está registrada.
- [ ] O CLAUDE.md e o README estão atualizados.
- [ ] A engine funciona com um único ChromaDB.
