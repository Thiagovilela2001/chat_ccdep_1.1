---
name: social-protection-analysis
description: >
  Estruturar e revisar análises de proteção social brasileiras e paulistas.
  Usar em perguntas sobre Cadastro Único ou CadÚnico, Programa Bolsa Família
  ou PBF, Benefício de Prestação Continuada ou BPC, transferência de renda,
  famílias beneficiárias, pobreza administrativa, Regra de Proteção, perfil
  dos inscritos e valores ou cobertura de benefícios sociais.
---

# Análise de Proteção Social

## Objetivo

Interpretar indicadores de CadÚnico, Bolsa Família e BPC sem misturar
populações, unidades, datas de referência ou conceitos administrativos.
Limitar conclusões ao conteúdo recuperado das fontes.

## Fluxo

1. Identificar programa, indicador, unidade, território, período e fonte.
2. Distinguir pessoa, família, benefício, pagamento e recurso financeiro.
3. Separar inscrição no CadÚnico, elegibilidade e recebimento efetivo.
4. Identificar se o número representa estoque em uma data, média mensal,
   total anual ou fluxo entre períodos.
5. Confirmar linha de renda, regra administrativa e vigência antes de comparar
   pobreza ou baixa renda.
6. Comparar taxas em pontos percentuais e níveis em variação percentual.
7. Deflacionar valores somente com índice e data-base documentados.
8. Tratar sobreposição entre programas somente quando a fonte a mensurar.
9. Citar fonte, período e página para cada afirmação factual ou numérica.

## Referências

- Consultar [concepts-methods.md](references/concepts-methods.md) para conceitos,
  fórmulas, comparabilidade e erros comuns.
- Consultar [seade-social.md](references/seade-social.md) para convenções dos
  boletins Seade Social e padrões de recuperação.
- Consultar [corpus-map.md](references/corpus-map.md) para escopo do acervo.
- Executar `python scripts/build_corpus_manifest.py` após incluir ou remover
  PDFs do acervo.

<!-- rag-context:start -->
## Contexto para síntese RAG

- Confirmar programa, indicador, unidade, território, período, data de
  referência e fonte antes de interpretar qualquer número.
- Não tratar inscritos no CadÚnico como beneficiários do Bolsa Família ou do
  BPC. Inscrição, elegibilidade e recebimento são estados distintos.
- Distinguir pessoas, famílias, benefícios, pagamentos e valores transferidos;
  não somar ou comparar unidades diferentes.
- Identificar estoque em uma data, média mensal, total anual e fluxo entre
  períodos. Não transformar um deles em outro sem dados suficientes.
- Aplicar linhas de pobreza, baixa renda e regras administrativas conforme a
  vigência informada. Valores de corte mudam no tempo.
- Informar diferenças entre percentuais em pontos percentuais. Calcular
  variação percentual somente a partir de níveis compatíveis.
- Separar valores nominais de reais. Informar deflator e data-base ao usar
  valores corrigidos.
- Não inferir causalidade, sobreposição de programas, erro cadastral ou efeito
  de política sem apoio explícito dos documentos recuperados.
- Não introduzir fatos externos. Usar este contexto somente como regra de
  interpretação das fontes recuperadas.
<!-- rag-context:end -->
