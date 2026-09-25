# Framework de Avaliação e Benchmark (Golden Dataset)

Este diretório reúne os conjuntos de referência (Ground Truth) e a documentação do framework de avaliação do **RAG Estatístico SP (CCDEP/Seade)**.

---

## 1. Estrutura dos Datasets (40 Perguntas Curadas)

O benchmark é composto por **40 perguntas curadas por especialistas de domínio**, divididas em splits estanques para evitar vazamento de dados (*data leakage*):

| Split | Arquivo | Qtd | Objetivo | Métrica Principal |
|---|---|:---:|---|---|
| **Dev** | `golden_dataset_dev.json` | 15 | Calibração de prompts, recuperação e regras de negócio | Faithfulness, ContextRecall, ContextPrecision |
| **Test** | `golden_dataset_test.json` | 15 | Teste cego de generalização do pipeline RAG | Faithfulness, ContextRecall, ContextPrecision |
| **Adversarial** | `golden_dataset_adversarial.json` | 10 | Teste de recusa a perguntas fora do escopo/acervo | Refusal Accuracy (100% no baseline) |
| **Consolidado** | `golden_dataset.json` | 40 | Visão unificada de todas as perguntas com tag `split` | Todas |

---

## 2. Cobertura Temática e Tipos de Consulta

### Domínios Cobertos
1. **Mercado de Trabalho (`labor_market`)**: Ocupação, desocupação, taxa de subutilização, rendimento, emprego formal (Caged), Rais, PNAD Contínua e informalidade por sexo/raça.
2. **Proteção Social (`social_protection`)**: CadÚnico, Programa Bolsa Família (PBF), Benefício de Prestação Continuada (BPC), linhas de pobreza e transferências de renda municipais.
3. **Conjuntura Econômica (`economic_conjuncture`)**: PIB paulista, produção da indústria de transformação, comércio varejista e setor de serviços.
4. **Investimento e Comércio Exterior (`investment_trade`)**: Exportações, importações e saldo comercial.
5. **Demografia (`demography`)**: Projeções e dinâmica populacional.
6. **Setorial e Regional (`sectoral_regional`)**: Diferenças regionais e cadeias produtivas no Estado de São Paulo.

### Tipos de Raciocínio Avaliados
- `lookup`: Extração direta de indicador pontual ou dado factual.
- `comparacao`: Contraste entre períodos, setores ou grupos demográficos.
- `tendencia`: Análise de séries históricas e trajetórias intertemporais.
- `calculo`: Variações percentuais, taxas acumuladas e saldos.
- `adversarial`: Perguntas propositalmente fora do acervo (outros estados, cidades sem cobertura, temas não estatísticos).

---

## 3. As 4 Dimensões de Avaliação Operacionais

O script [`evaluate.py`](../evaluate.py) afere o pipeline em 4 dimensões críticas:

1. **Fidelidade às Fontes (Faithfulness - RAGAS)**:
   - Verifica se cada asserção da resposta é estritamente suportada pelo contexto recuperado dos boletins.
   - O prompt obriga a citação explícita do arquivo PDF e página-fonte.
2. **Precisão Numérica**:
   - `rag_core/numerical_validator.py` valida e audita números, datas e percentuais extraídos contra as tabelas oficiais.
   - Retrievers determinísticos em Pandas (`tables_retriever.py` e `timeseries_retriever.py`) impedem que o LLM realize cálculos livres ou invente valores.
3. **Tempo de Resposta (Latência)**:
   - Coleta de latência por pergunta com percentis $p50$, $p90$ e $p95$.
   - Controle de orçamento e SLA de tempo em [`rag_core/runtime.py`](../rag_core/runtime.py).
4. **Recusa de Perguntas Fora do Acervo**:
   - Avaliação determinística com `REFUSAL_KEYWORDS`.
   - Evita alucinações em perguntas que o acervo documental não possui subsídios para responder.

---

## 4. Como Reproduzir a Avaliação

Para executar a avaliação automatizada no terminal:

```bash
# Rodar todos os splits (Dev + Test + Adversarial)
python evaluate.py --split all

# Rodar split específico
python evaluate.py --split dev
python evaluate.py --split test
python evaluate.py --split adversarial
```

Os resultados detalhados por pergunta são gravados em `evaluation/results_{split}.json`.
