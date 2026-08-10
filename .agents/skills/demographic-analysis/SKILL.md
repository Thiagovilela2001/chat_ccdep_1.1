---
name: demographic-analysis
description: >
  Estruturar e revisar análises demográficas brasileiras e paulistas. Usar em
  perguntas sobre população, crescimento populacional, estrutura etária,
  envelhecimento, fecundidade, natalidade, mortalidade, esperança de vida,
  migração, urbanização, projeções populacionais, bônus demográfico, razões de
  dependência e comparações territoriais ou entre censos.
---

# Análise Demográfica

## Objetivo

Interpretar estoques, fluxos, taxas e projeções demográficas sem misturar
conceitos, períodos, populações expostas ou recortes territoriais. Limitar
conclusões ao que as fontes recuperadas sustentarem.

## Fluxo

1. Identificar pergunta, indicador, unidade, período, data de referência,
   território e população de interesse.
2. Classificar medida como estoque, fluxo, taxa, proporção, índice, estimativa
   ou projeção.
3. Confirmar numerador, denominador, multiplicador e população exposta ao risco.
4. Distinguir contagem censitária, estimativa intercensitária e projeção por
   cenário; não encadear séries como se fossem equivalentes.
5. Separar crescimento total, crescimento vegetativo e saldo migratório.
6. Verificar mudanças de limites territoriais, cobertura, conceitos, data de
   referência e metodologia antes de comparar períodos ou áreas.
7. Comparar estruturas etárias por proporções ou taxas específicas; padronizar
   quando diferenças de composição puderem distorcer taxas brutas.
8. Tratar causalidade, tendências futuras e efeitos socioeconômicos somente
   quando documentados pelas fontes.
9. Informar incerteza, revisão, sub-registro, erro amostral ou pequenos números
   quando material para a conclusão.
10. Citar fonte, período e página para cada afirmação factual ou numérica.

## Referências

- Consultar [conceitos-metodos.md](references/conceitos-metodos.md) para fórmulas,
  seleção de fontes, comparabilidade e erros comuns.

<!-- rag-context:start -->
## Contexto para síntese RAG

- Confirmar indicador, unidade, período, data de referência, território,
  população de interesse e fonte antes de interpretar qualquer número.
- Distinguir estoque populacional de fluxos de nascimentos, óbitos e migração;
  não comparar contagem absoluta com taxa ou participação como se fossem a
  mesma medida.
- Explicitar numerador, denominador e multiplicador das taxas. Usar população
  exposta ao risco e períodos compatíveis.
- Não tratar Censo, estimativa intercensitária e projeção populacional como
  séries diretamente equivalentes. Informar cenário e horizonte ao citar
  projeções.
- Decompor crescimento populacional em crescimento vegetativo e saldo
  migratório somente quando componentes compatíveis estiverem documentados.
  Identificar saldo migratório calculado por resíduo como estimativa, não como
  fluxo observado.
- Comparar fecundidade, natalidade, mortalidade e esperança de vida sem trocar
  conceitos. Taxas brutas podem refletir estrutura etária; padronizar ou usar
  taxas específicas quando necessário e possível.
- Verificar mudanças de limites municipais, cobertura, sub-registro, conceitos,
  metodologia e data de referência antes de comparar áreas ou períodos.
- Não inferir coortes a partir de faixas etárias observadas em anos distintos
  sem acompanhar gerações e considerar mortalidade e migração.
- Tratar bônus demográfico, envelhecimento e impactos econômicos como
  interpretações, não resultados automáticos de uma única razão ou proporção.
- Não introduzir fatos demográficos externos. Usar este contexto apenas como
  regra de interpretação dos documentos recuperados.
<!-- rag-context:end -->
