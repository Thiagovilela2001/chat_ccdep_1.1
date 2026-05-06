"""
Labor Market Skill — carrega a skill especializada de mercado de trabalho.
"""
import os
import re

_LABOR_MARKET_KEYWORDS = {
    "emprego", "desemprego", "desocupação", "desocupado", "ocupação",
    "ocupado", "trabalhador", "trabalho", "mercado de trabalho",
    "informalidade", "informal", "formal", "carteira assinada", "clt",
    "caged", "rais", "pnadc", "pnad contínua", "ped",
    "remuneração", "salário", "renda do trabalho",
    "admissão", "desligamento", "saldo de empregos", "vagas",
    "taxa de desocupação", "taxa de emprego", "taxa de informalidade",
    "pea", "população economicamente ativa", "subocupação", "subutilização",
    "rotatividade", "turnover", "demissão", "contratação",
    "setor formal", "setor informal", "emprego formal", "emprego informal",
}

_SECTION_RE = {
    "glossary":  re.compile(
        r"### Glossário de Indicadores-Chave\n(.*?)(?=\n---|\n##|\Z)", re.DOTALL
    ),
    "checklist": re.compile(
        r"### Checklist de Interpretação\n(.*?)(?=\n---|\n###|\Z)", re.DOTALL
    ),
}


def is_labor_market_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _LABOR_MARKET_KEYWORDS)


class LaborMarketSkill:
    def __init__(self, base_dir: str):
        skill_path = os.path.join(
            base_dir,
            ".agents", "skills", "labor_market_analysis", "SKILL.md",
        )
        self._context = self._load(skill_path)

    def _load(self, path: str) -> str:
        if not os.path.exists(path):
            return ""

        with open(path, encoding="utf-8") as f:
            content = f.read()

        sections = []
        for name, pattern in _SECTION_RE.items():
            match = pattern.search(content)
            if match:
                sections.append(match.group(0).strip())

        return "\n\n".join(sections) if sections else ""

    def get_context(self) -> str:
        return self._context

    def is_loaded(self) -> bool:
        return bool(self._context)