CLASSIFY_PROMPT = """\
Classifique a pergunta do usuário em exatamente uma das categorias abaixo.

Categorias:
- lookup: busca de um valor pontual (ex: "qual foi o PIB em 2020?", "qual o saldo de empregos em março?")
- comparacao: comparação entre dois ou mais valores sem operação matemática (ex: "qual estado teve maior PIB?", "SP criou mais empregos que MG?")
- calculo: requer operação matemática explícita — variação, crescimento, porcentagem, média, soma (ex: "qual foi o crescimento do PIB entre 2010 e 2020?", "qual a variação percentual do desemprego?")
- tendencia: análise de padrão ao longo do tempo sem cálculo pontual (ex: "como evoluiu o emprego nos últimos anos?", "o PIB está crescendo ou caindo?")

Responda com UMA única palavra (sem pontuação, sem espaços extras): lookup, comparacao, calculo ou tendencia.

Pergunta: {question}
Tipo:"""

VALID_TYPES = {"lookup", "comparacao", "calculo", "tendencia"}


def classify_query(question: str, llm) -> str:
    """Classifica a pergunta e retorna o tipo: lookup, comparacao, calculo ou tendencia."""
    response = llm.complete(CLASSIFY_PROMPT.format(question=question))
    tipo = response.text.strip().lower().rstrip(".")
    if tipo not in VALID_TYPES:
        return "lookup"
    return tipo
