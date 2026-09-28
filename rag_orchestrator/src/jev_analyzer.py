"""
jev_analyzer.py — Classificação semântica da consulta via Jev (System One).

Substitui a chamada LLM de JSON livre do `QueryAnalyzer` por **uma requisição
typing-safe**: 8 perguntas `Choice` (dimensões categóricas da consulta) e 2 `Noul`
(sinais booleanos), todas avaliadas em paralelo contra o mesmo estado. Não há
texto livre para parsear: a saída já vem como opção escolhida + opções permitidas
+ distribuição de probabilidade + `confidence`.

O que foi medido (15 perguntas reais: as 10 do golden dataset + 5 casos difíceis,
via `scripts/jev_smoke.py` e medições em scratch):

- Confiança agregada. O mínimo passava por dimensões que não influenciam rota
  nenhuma: `needs_multi_hop` prendia o mínimo em 5 das 15 perguntas e
  `technical_terms` (sem nenhum consumidor no código) em 3. Por isso
  `technical_terms` deixou de ser perguntado e `needs_multi_hop` passou a ser
  **derivada** de `query_type` (a opção `multi_hop` já existe lá — eram duas
  perguntas sobre o mesmo fato, e elas discordavam: grau de decisão médio 0.40).
  Restringindo o mínimo às dimensões que o `router` consome, o agregado subiu de
  0.296 para 0.369 de média e a recusa por escopo passou a ser alcançável
  (0/15 → 1/15 casos com confiança ≥ 0.75).

- `Score` x `Choice` nas duas dimensões de espectro. Converter `retrieval_need` e
  `complexity` para escala foi testado com as duas formulações na MESMA requisição
  e **piorou**: a escala convida a resposta graduada e espalha a massa mesmo
  quando a crença categórica é firme — para "Qual foi a variação do PIB no 1º
  trimestre?", `complexity` deu {0: 0.66, 1: 0.17, 2: 0.17} como `Score` contra
  {baixa: 0.90, media: 0.07, alta: 0.05} como `Choice`. Margem média 0.24 contra
  0.46. As duas seguem como `Choice`, com critérios descritos por situação.

Confiança agregada final: o **menor** grau de decisão entre as dimensões que o
`router` consome (`query_type`, `priority`, `retrieval_need`, `complexity`,
`in_scope`). `retrieval_need` e `complexity` são hoje as que mais prendem o
mínimo — é aí que uma próxima medição deve mirar.

Campos de texto livre do contrato (`entities`, `period`, `linguistic_patterns`) e
`technical_terms` — que nenhum consumidor lê — não têm equivalente útil em Jev
(não gera texto / não é julgado) e permanecem nos defaults; o caminho LLM segue
preenchendo `technical_terms`.

Sem chave, com `RAG_JEV_ENABLED=0` ou em qualquer falha da API, `analyze()`
delega ao `QueryAnalyzer` (LLM) e este, por sua vez, à heurística: o dict de
saída é o mesmo em todos os caminhos.
"""
from __future__ import annotations

from typing import Any, Callable

from rag_core.jev import JevResponse, JevClient, choice, jev_enabled, model_name, noul
from rag_core.logger import get_logger

from .query_analyzer import _ALLOWED, _merge_defaults

log = get_logger(__name__)

# Dimensões categóricas (Choice) e booleanas (Noul) do contrato.
_CHOICE_DIMENSIONS = (
    "intent",
    "query_type",
    "semantic_domain",
    "specificity",
    "expected_answer",
    "priority",
    "retrieval_need",
    "complexity",
)
_NOUL_DIMENSIONS = (
    "is_labor_market",
    "in_scope",
)
# Dimensões que mudam a rota (ou o portão de recusa). A confiança agregada é o
# menor grau de decisão apenas entre elas; `needs_multi_hop` fica de fora porque
# é derivada de `query_type` e não deve contar duas vezes.
_ROUTING_DIMENSIONS = (
    "query_type",
    "priority",
    "retrieval_need",
    "complexity",
    "in_scope",
)

# Critérios concretos: níveis descritos por situação, não por adjetivo vago.
_CHOICE_CRITERIA: dict[str, dict[str, str]] = {
    "intent": {
        "consulta_dado": "pede um dado, valor ou número específico",
        "comparar": "pede comparação entre períodos, setores ou regiões",
        "resumir": "pede panorama ou resumo de um tema ou período",
        "explicar": "pede explicação de causa, conceito ou relação entre fatos",
        "verificar": "pede confirmação ou direção de um dado já mencionado",
    },
    "query_type": {
        "pontual": "um valor ou fato isolado em um trecho",
        "tabular": "dados organizados em tabela, com vários indicadores ou recortes",
        "temporal": "série ou evolução ao longo do tempo",
        "ampla": "panorama amplo, sem recorte específico",
        "comparativo": "contraste explícito entre dois ou mais itens ou períodos",
        "relacional": "ligação entre entidades distintas (ex.: setor que mais emprega)",
        "multi_hop": "exige encadear várias buscas até chegar à resposta",
        "verificacao": "confirmação de um dado específico pedido na pergunta",
    },
    "semantic_domain": {
        "emprego": "emprego, desocupação, ocupação, informalidade, rendimento do trabalho",
        "pib": "PIB, valor adicionado, atividade econômica agregada",
        "industria": "indústria, produção industrial, transformação",
        "precos": "inflação, índices de preços e custos",
        "comercio": "comércio, varejo, vendas",
        "servicos": "serviços, setor terciário",
        "geral": "tema transversal ou nenhum dos anteriores identificável",
    },
    "specificity": {
        "especifica": "indicador e recorte claros (ex.: taxa de desocupação em 2024)",
        "intermediaria": "tema definido com recorte parcial",
        "ampla": "tema genérico, sem indicador nem recorte",
    },
    "expected_answer": {
        "numerico": "um valor ou percentual",
        "tabela": "tabela de dados",
        "serie": "série temporal",
        "narrativo": "texto explicativo",
        "comparativo": "comparação estruturada entre itens",
    },
    "priority": {
        "precisao": "números exatos e verificação numérica importam mais",
        "abrangencia": "cobertura ampla do tema importa mais que o número exato",
    },
    "retrieval_need": {
        "lexical": "siglas, códigos e termos exatos importam (ex.: CAGED, RAIS, PNADC)",
        "hibrida": "termos exatos e conceitos importam igualmente",
        "semantica": "só conceitos e paráfrases importam, sem termo exato a casar",
    },
    "complexity": {
        "baixa": "resposta direta em um único trecho",
        "media": "exige combinar alguns trechos ou um cálculo simples",
        "alta": "exige encadear várias buscas, entidades ou cálculos",
    },
}

_NOUL_INSTRUCTIONS: dict[str, str] = {
    "is_labor_market": (
        "`pergunta` é sobre mercado de trabalho — emprego, desemprego, "
        "ocupação, informalidade, CAGED, RAIS, PNAD Contínua ou rendimento "
        "do trabalho?"
    ),
    "in_scope": (
        "`pergunta` pode ser respondida pelos Boletins de Conjuntura Paulista e "
        "pelo Seade Social sobre a economia e as estatísticas do Estado de São "
        "Paulo entre 2020 e 2025?"
    ),
}

# A instrução é a pergunta completa: o id não chega ao modelo (docs/primitives).
_CHOICE_INSTRUCTIONS: dict[str, str] = {
    "intent": "Em `pergunta`, qual é a intenção dominante do usuário?",
    "query_type": "Que tipo de consulta `pergunta` é, para fins de recuperação?",
    "semantic_domain": "Qual é o domínio econômico predominante de `pergunta`?",
    "specificity": "Qual é o grau de especificidade de `pergunta`?",
    "expected_answer": "Que formato de resposta `pergunta` espera?",
    "priority": "`pergunta` valoriza precisão numérica ou abrangência temática?",
    "retrieval_need": "Que tipo de recuperação `pergunta` exige?",
    "complexity": "Qual é a complexidade de responder `pergunta`?",
}


def build_state(question: str) -> dict:
    """
    Estado enviado à Jev: a pergunta em campo próprio mais o contexto do corpus,
    para que cada julgamento saiba contra o que está sendo avaliado.
    """
    return {
        "pergunta": question,
        "corpus": {
            "fonte": "Boletins de Conjuntura Paulista e Seade Social (Fundação Seade)",
            "tema": "economia e estatísticas do Estado de São Paulo",
            "periodo": "2020 a 2025",
        },
        "fora_do_escopo": [
            "indicadores nacionais ou de outros estados (ex.: Selic, PIB do Brasil)",
            "temas sem relação com economia e estatística paulista (ex.: esportes, receitas)",
        ],
    }


def build_questions() -> dict:
    """Todas as perguntas do roteamento — uma única requisição, avaliação paralela."""
    questions = {
        key: choice(_CHOICE_INSTRUCTIONS[key], criteria)
        for key, criteria in _CHOICE_CRITERIA.items()
    }
    questions.update(
        {key: noul(instruction) for key, instruction in _NOUL_INSTRUCTIONS.items()}
    )
    return questions


class JevQueryAnalyzer:
    """
    Classificador semântico via Jev. Mesmo contrato de `QueryAnalyzer.analyze`.

    `client` é injetável (duck-typed: precisa de `system_one`) e `fallback`
    aceita um chamável `question -> dict` ou um analisador com `.analyze`;
    sem eles, usa `JevClient` e `QueryAnalyzer`.
    """

    def __init__(
        self,
        client: Any = None,
        fallback: Any = None,
    ):
        self._client = client
        self._fallback = _as_fallback(fallback)
        self._served_model: str | None = None

    @property
    def model(self) -> str:
        """
        Modelo que classificou a consulta: o servido pela API quando já houve
        chamada (ex.: `jev-1.13.0`), senão o configurado (`TYPESAFE_MODEL`).
        Quando a classificação veio do reserva LLM, o campo `reasoning` da
        análise indica a troca.
        """
        return self._served_model or model_name()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = JevClient()
        return self._client

    def analyze(self, question: str) -> dict:
        """Dict de classificação, sempre completo — a LLM entra só como reserva."""
        if self._client is None and not jev_enabled():
            log.info("JevAnalyzer: Jev desabilitado — usando o analisador LLM.")
            return self._fallback_analysis(question)

        try:
            response = self._get_client().system_one(build_state(question), build_questions())
        except Exception as exc:  # contrato do cliente é "não lançar", mas não confiamos
            log.warning("JevAnalyzer: chamada falhou (%s) — usando fallback.", exc)
            response = None

        if not isinstance(response, JevResponse):
            return self._fallback_analysis(question)

        data = _to_classification(response)
        if data is None:
            return self._fallback_analysis(question)
        if response.model:
            self._served_model = response.model
        metadata = _merge_defaults(data)
        log.info("JevAnalyzer: %s", metadata["reasoning"])
        return metadata

    def _fallback_analysis(self, question: str) -> dict:
        if self._fallback is None:
            from .query_analyzer import QueryAnalyzer  # import tardio: evita ciclo

            self._fallback = QueryAnalyzer().analyze
        return self._fallback(question)


def _as_fallback(fallback: Any) -> Callable[[str], dict] | None:
    """Aceita um chamável `question -> dict` ou um analisador com `.analyze`."""
    if fallback is None:
        return None
    return fallback if callable(fallback) else fallback.analyze


def _to_classification(response: JevResponse) -> dict | None:
    """Converte respostas tipadas em dimensões do contrato; None se nada veio."""
    data: dict[str, Any] = {}
    for key in _CHOICE_DIMENSIONS:
        chosen = response.choice(key, allowed=_ALLOWED[key])
        if chosen is not None:
            data[key] = chosen
    for key in _NOUL_DIMENSIONS:
        decision = response.noul_decision(key)
        if decision is not None:
            data[key] = decision

    if not data:
        return None

    # Derivada, não perguntada: `query_type=multi_hop` já afirma exatamente isso.
    data["needs_multi_hop"] = data.get("query_type") == "multi_hop"

    weakest = response.weakest_decisiveness(_ROUTING_DIMENSIONS)
    data["confidence"] = 0.5 if weakest is None else round(weakest, 3)
    data["reasoning"] = _reasoning(response)
    return data


def _reasoning(response: JevResponse) -> str:
    """Resumo determinístico e auditável das probabilidades que sustentam a rota."""
    parts = []
    for key in ("query_type", "semantic_domain", "is_labor_market", "in_scope"):
        answer = response.answer(key)
        if answer is None:
            continue
        if answer.type == "choice" and answer.choice is not None:
            weight = answer.probability(answer.choice) or answer.confidence
            detail = f" ({weight:.2f})" if weight is not None else ""
            parts.append(f"{key}={answer.choice}{detail}")
        elif answer.type == "noul" and answer.noul is not None:
            parts.append(f"{key}={answer.noul:.2f}")
    summary = ", ".join(parts)
    return f"Jev {response.model or 'system-one'}: {summary}" if summary else "Jev: sem respostas"
