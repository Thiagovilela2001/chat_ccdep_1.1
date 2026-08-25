"""Validacao lexical de suporte para argumentos em texto livre."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .answer_policy import REFUSAL_TEXT

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{4,}", re.UNICODE)

_STOPWORDS = {
    "ainda", "alem", "algo", "anos", "apos", "aquela", "aquele", "aquelas",
    "aqueles", "assim", "cada", "como", "com", "contra", "depois", "desde",
    "dessa", "desse", "deste", "desta", "diferentes", "durante", "elas",
    "eles", "entre", "essa", "esse", "esta", "este", "foram", "havia",
    "isso", "isto", "mais", "maior", "maiores", "menor", "menos", "mesma",
    "mesmo", "muito", "nao", "nessa", "nesse", "neste", "nesta", "onde",
    "para", "pela", "pelo", "pelas", "pelos", "pois", "ponto", "porque",
    "quando", "quanto", "quase", "sobre", "tambem", "todos", "todas",
    "tres", "teve", "tiveram", "fornecem", "disponiveis", "suficiente",
    "responder", "solicitado", "documentos", "evidencia",
}


@dataclass(frozen=True)
class ArgumentCheck:
    sentence: str
    verified: bool
    missing_terms: list[str] = field(default_factory=list)
    support_ratio: float = 1.0


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _terms(text: str) -> list[str]:
    folded = _fold(text)
    terms = []
    for match in _WORD_RE.finditer(folded):
        term = match.group(0)
        if term in _STOPWORDS:
            continue
        terms.append(term)
    return terms


def _content_sentences(answer: str) -> list[str]:
    if REFUSAL_TEXT.lower() in (answer or "").lower():
        return []
    return [
        match.group(0).strip()
        for match in _SENTENCE_RE.finditer(answer or "")
        if len(match.group(0).strip()) >= 24
    ]


def validate_arguments(answer: str, source_nodes) -> list[ArgumentCheck]:
    """Verifica se os termos substantivos de cada frase aparecem nas fontes.

    Esta validacao e propositalmente conservadora: ela nao tenta provar semantica
    profunda, mas bloqueia frases que introduzem vocabulario analitico sem lastro
    lexical no material recuperado.
    """
    source_text = _fold("\n".join(str(node.get_content() or "") for node in source_nodes))
    source_terms = set(_terms(source_text))
    checks: list[ArgumentCheck] = []

    for sentence in _content_sentences(answer):
        terms = sorted(set(_terms(sentence)))
        if len(terms) < 3:
            checks.append(ArgumentCheck(sentence=sentence, verified=True))
            continue
        missing = [term for term in terms if term not in source_terms]
        support_ratio = (len(terms) - len(missing)) / len(terms)
        checks.append(ArgumentCheck(
            sentence=sentence,
            verified=support_ratio >= 0.72 and len(missing) <= 3,
            missing_terms=missing,
            support_ratio=round(support_ratio, 3),
        ))

    return checks
