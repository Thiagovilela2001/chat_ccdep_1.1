# Apresentação Nadia

Material de apresentação do projeto, retirado da raiz do repositório para este
diretório. Nada aqui é código: nenhum módulo, teste ou serviço lê estes arquivos.

| Arquivo | O que é |
|---|---|
| `apresentacao_nadia_conceitos.md` | roteiro de conceitos para plateia leiga: 7 blocos (~15 min), com os bullets de tela e a fala sugerida de cada slide |
| `Nadia_RAG_CCDEP.pptx` | deck de 25 slides |
| `Nadia_RAG_CCDEP_atualizado.pptx` | mesmo deck, versão revisada (25 slides) |
| `slide_content.json` | texto dos 25 slides, por `slide` e `shape` — bate 1:1 com a contagem de slides dos dois decks |

Observações:

- os quatro arquivos entraram juntos no commit `805c5df` ("docs(presentation): add Nadia concept deck");
- os dois `.pptx` têm 229 entradas cada e não trazem `docProps/core.xml`, ou seja, foram
  gerados por script e não guardam metadados de autor/revisão — a diferença entre "atualizado"
  e o original não é verificável pelo arquivo, só pelo conteúdo;
- o roteiro em Markdown descreve 17 slides, e o deck tem 25: as duas contagens não batem.
