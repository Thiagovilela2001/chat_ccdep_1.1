---
name: labor-market-analysis
description: >
  Estruturar e revisar análises do mercado de trabalho brasileiro e paulista.
  Usar em perguntas sobre emprego, desemprego, ocupação, desocupação,
  informalidade, subutilização, rendimento do trabalho, desigualdades de sexo
  e raça/cor, emprego formal, admissões, desligamentos, Caged, Rais, PNAD
  Contínua, diferenças setoriais ou regionais do trabalho.
---

# Análise de Mercado de Trabalho

## Objetivo

Interpretar indicadores laborais sem misturar fontes, populações, frequências,
territórios ou conceitos de formalidade. Limitar conclusões às fontes
recuperadas.

## Fluxo

1. Identificar indicador, unidade, fonte, frequência, período e território.
2. Distinguir estoque de ocupados, fluxo de admissões/desligamentos, taxa,
   participação, rendimento e massa de rendimentos.
3. Confirmar população e denominador de cada taxa.
4. Verificar pesos amostrais, reponderações e comparabilidade na PNAD Contínua.
5. Verificar sazonalidade e quebra Caged/Novo Caged antes de comparar períodos.
6. Preservar definição de formalidade ou informalidade usada por cada fonte.
7. Separar valores nominais e reais; informar deflator e data-base.
8. Comparar grupos, setores e regiões apenas sob conceitos compatíveis.
9. Citar fonte, período e página para cada afirmação factual ou numérica.

## Referências

- Consultar [general-methods.md](references/general-methods.md) para seleção de
  fontes, desenho amostral, séries temporais e integração PNAD/Caged/Rais.
- Consultar [seade-trabalho.md](references/seade-trabalho.md) para convenções,
  famílias documentais e recortes territoriais do acervo Seade Trabalho.
- Consultar [cnae_grupos_analiticos.md](resources/cnae_grupos_analiticos.md) ao
  harmonizar setores CNAE.
- Usar exemplos em `examples/` somente como ponto de partida; validar esquemas,
  tabelas e períodos contra documentação atual da fonte.

<!-- rag-context:start -->
### Checklist de Interpretação

- Confirmar indicador, unidade, fonte, frequência, período, território e
  população antes de interpretar qualquer número.
- Distinguir nível, taxa, participação, saldo, estoque e variação. Saldo Caged
  é admissões menos desligamentos; não equivale à taxa de desocupação.
- Comparar taxas em pontos percentuais e níveis em variação percentual.
- Na PNAD Contínua, preservar pesos, reponderações, população de referência e
  definição publicada. Considerar erro amostral quando material.
- Não equiparar automaticamente contribuição previdenciária, carteira assinada
  e informalidade. Usar definição explícita do documento.
- Separar valor nominal, rendimento real e massa real. Informar deflator e
  data-base.
- Verificar sazonalidade, efeito-base e quebras metodológicas antes de comparar
  meses, trimestres ou séries longas.
- Não somar regiões sobrepostas nem tratar estratos PNAD como regiões
  administrativas oficiais.
- Explicar mecanismos, causalidade e política pública somente quando pedidos e
  sustentados pelas fontes recuperadas.
- Não introduzir fatos externos. Usar este contexto somente como regra de
  interpretação dos documentos recuperados.

### Glossário de Indicadores-Chave

| Indicador | Definição operacional |
|---|---|
| Taxa de desocupação | Desocupados / força de trabalho × 100 |
| Taxa composta de subutilização | Desocupados, subocupados por insuficiência de horas e força de trabalho potencial / força de trabalho ampliada × 100 |
| Taxa de informalidade | Informais / ocupados × 100, conforme categorias definidas pela fonte |
| Saldo Caged | Admissões − desligamentos no período |
| Estoque formal | Vínculos ativos na data de referência da fonte |
| Rendimento médio real | Rendimento nominal corrigido pelo deflator e data-base informados |
| Massa de rendimento real | Soma dos rendimentos reais da população coberta |
<!-- rag-context:end -->
