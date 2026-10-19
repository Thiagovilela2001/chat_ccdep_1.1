"""
Avaliação científica do rag_principal: acerto de indicadores + qualidade do raciocínio.

Entrada: artefatos de `evaluate.py --generation-only` (uma ou mais repetições).
Saída: JSON com escores por resposta e relatório Markdown com estatística inferencial.

Dimensões:
  A. Acerto de indicadores (determinístico): números do gabarito presentes na resposta,
     com tolerância de arredondamento na precisão do gabarito e mesma unidade (% ou não).
  B. Raciocínio (LLM-juiz com rubrica 0-2): direção, comparação, unidade/escala,
     temporalidade, inferência suportada, síntese. Sinaliza "número certo, leitura errada".
  C. Validação do juiz: erros injetados no gabarito (sensibilidade) e gabarito intacto
     (especificidade); concordância entre duas passadas do juiz (kappa ponderado).
  D. Estatística: Wilson 95%, bootstrap por cluster (pergunta), Fleiss kappa entre
     repetições, McNemar exato, Spearman, Kruskal-Wallis com correção de Holm.

Uso:
    python evaluation/scientific_eval.py evaluation/runs/sci_20261016/*_rep*.json \
        --out evaluation/runs/sci_20261016
    python evaluation/scientific_eval.py --selftest
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy import stats

from rag_core.numerical_validator import _number_matches, _parse_decimal

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CRITERIA = ["direcao", "comparacao", "unidade_escala", "temporalidade", "inferencia", "sintese"]
CRITERIA_DESC = {
    "direcao": "sentido das variações e tendências (alta/queda/estabilidade)",
    "comparacao": "comparações e ordenamentos entre grupos, setores, regiões ou períodos",
    "unidade_escala": "unidades e escala (p.p. vs %, taxa vs nível, mil vs milhão, nominal vs real)",
    "temporalidade": "períodos de referência e bases de comparação (interanual, acumulado, trimestre)",
    "inferencia": "conclusões sustentadas pelos dados; sem causalidade, generalização ou extrapolação indevida",
    "sintese": "a leitura analítica global é coerente com o gabarito",
}
_YEAR_RANGE = (1990, 2035)
B = 10_000
RNG = np.random.default_rng(42)


# ── A. Acerto de indicadores ────────────────────────────────────────────────

def extract_numbers(text: str) -> list[tuple[Decimal, int, bool, str]]:
    """(valor absoluto, casas decimais, é percentual, texto bruto); anos excluídos."""
    out = []
    for m in _number_matches(text or ""):
        raw = m.group(1).strip()
        val = _parse_decimal(raw)
        if val is None:
            continue
        is_pct = raw.endswith("%") or bool(re.match(r"\s*(?:%|por cento)", text[m.end(1):m.end(1) + 12]))
        val = abs(val)
        if not is_pct and val == val.to_integral() and _YEAR_RANGE[0] <= val <= _YEAR_RANGE[1] and "." not in raw:
            continue
        decimals = len(re.sub(r"\D", "", raw.split(",")[1])) if "," in raw else 0
        out.append((val, decimals, is_pct, raw))
    return out


def indicator_score(ground_truth: str, answer: str) -> dict:
    gt = extract_numbers(ground_truth)
    ans = extract_numbers(answer)
    hits, misses = [], []
    for val, dec, pct, raw in gt:
        tol = Decimal("0.5") * Decimal(10) ** -dec
        ok = any(p == pct and abs(a - val) <= tol for a, _, p, _ in ans)
        (hits if ok else misses).append(raw)
    n = len(gt)
    return {
        "gt_numbers": n,
        "hits": hits,
        "misses": misses,
        "recall": len(hits) / n if n else None,
        "all_correct": (len(misses) == 0) if n else None,
    }


# ── B. Juiz de raciocínio ───────────────────────────────────────────────────

JUDGE_PROMPT = """Você é revisor(a) metodológico(a) de um artigo científico de análise socioeconômica.
Avalie a INTERPRETAÇÃO da resposta, não apenas os números. Use o GABARITO como verdade de referência
e os TRECHOS DAS FONTES para checar afirmações adicionais. Informação extra correta e suportada pelas
fontes NÃO é erro. Afirmação extra não suportada por fontes ou gabarito É erro de inferência.

Para cada critério dê nota 0, 1 ou 2:
  2 = correto/sem problemas; 1 = imprecisão menor que não muda a conclusão; 0 = erro que muda a conclusão.
  Se o critério não se aplica à pergunta, use null.
Critérios:
{criteria}

Também responda:
  numeros_corretos_interpretacao_errada: true se os valores citados estão corretos mas a leitura
    (direção, comparação, unidade, período ou conclusão) está errada; senão false.
  erros: lista de objetos {{"criterio": <um dos critérios>, "gravidade": "menor"|"grave", "trecho": <citação curta da resposta>, "explicacao": <1 frase>}}.

PERGUNTA:
{question}

GABARITO:
{ground_truth}

TRECHOS DAS FONTES (truncados):
{contexts}

RESPOSTA AVALIADA:
{answer}

Responda APENAS com JSON: {{"notas": {{"direcao": ..., ...}}, "numeros_corretos_interpretacao_errada": ..., "erros": [...]}}"""


def _judge_once(client, model, item: dict, contexts_chars: int, temperature: float) -> dict:
    ctx = "\n---\n".join(item.get("retrieved_contexts") or [])[:contexts_chars] or "(sem trechos)"
    prompt = JUDGE_PROMPT.format(
        criteria="\n".join(f"  - {k}: {v}" for k, v in CRITERIA_DESC.items()),
        question=item["question"], ground_truth=item["ground_truth"],
        contexts=ctx, answer=item["answer"],
    )
    last_err = None
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            notas = data.get("notas", {})
            data["notas"] = {k: (int(notas[k]) if notas.get(k) is not None else None) for k in CRITERIA}
            return data
        except Exception as exc:  # retry em JSON inválido/erro transitório
            last_err = exc
    return {"error": f"{type(last_err).__name__}: {last_err}"}


def reasoning_summary(j: dict) -> dict:
    notas = [v for v in j.get("notas", {}).values() if v is not None]
    errs = j.get("erros") or []
    return {
        "reasoning_score": (sum(notas) / (2 * len(notas))) if notas else None,
        "has_grave_error": any(e.get("gravidade") == "grave" for e in errs)
        or any(v == 0 for v in notas),
        "misinterpreted": bool(j.get("numeros_corretos_interpretacao_errada")),
    }


# ── C. Perturbações para validar o juiz ─────────────────────────────────────

_SWAPS = {
    "direcao": [("aumento", "redução"), ("alta", "queda"), ("cresceu", "recuou"), ("crescimento", "retração"),
                ("elevação", "diminuição"), ("expansão", "contração"), ("avançou", "caiu"), ("subiu", "caiu")],
    "comparacao": [("maior", "menor"), ("superior", "inferior"), ("acima", "abaixo"), ("mais", "menos")],
}
_CAUSAL = (" Essa variação decorreu exclusivamente da política adotada pelo governo estadual no período,"
           " o que comprova seu efeito causal direto.")


def _swap_words(text: str, pairs) -> str | None:
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for p in pairs for w in p) + r")\b", re.IGNORECASE)
    table = {a: b for a, b in pairs} | {b: a for a, b in pairs}
    new = pat.sub(lambda m: (lambda r: r.capitalize() if m.group(0)[0].isupper() else r)(table[m.group(0).lower()]), text)
    return new if new != text else None


def perturb(gt: str) -> dict[str, tuple[str, str]]:
    """{tipo: (texto, critério que deve cair)}; só tipos aplicáveis ao gabarito."""
    out = {"controle": (gt, "")}
    for kind in ("direcao", "comparacao"):
        new = _swap_words(gt, _SWAPS[kind])
        if new:
            out[kind] = (new, kind)
    if re.search(r"p\.\s?p\.|pontos? percentua", gt):
        out["unidade_escala"] = (re.sub(r"\s*(?:p\.\s?p\.|pontos? percentuais|ponto percentual)", "%", gt), "unidade_escala")
    elif "%" in gt:
        out["unidade_escala"] = (gt.replace("%", " p.p.", 1), "unidade_escala")
    out["inferencia"] = (gt + _CAUSAL, "inferencia")
    return out


# ── D. Estatística ──────────────────────────────────────────────────────────

def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return k / n, ci.low, ci.high


def cluster_bootstrap(values_by_cluster: dict[str, list[float]]) -> tuple[float, float, float]:
    """Média e IC95% percentil reamostrando perguntas (clusters), não respostas."""
    clusters = [np.array(v, float) for v in values_by_cluster.values() if v]
    if not clusters:
        return (float("nan"),) * 3
    point = float(np.mean(np.concatenate(clusters)))
    idx = RNG.integers(0, len(clusters), size=(B, len(clusters)))
    sums = np.array([c.sum() for c in clusters]); lens = np.array([len(c) for c in clusters])
    boots = sums[idx].sum(1) / lens[idx].sum(1)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = nº de repetições da pergunta i na categoria j (n fixo por linha)."""
    table = table[table.sum(1) == table.sum(1).max()]
    n = table.sum(1)[0]
    if n < 2 or len(table) == 0:
        return float("nan")
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def weighted_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    """Kappa de Cohen com pesos quadráticos (notas ordinais 0..k-1)."""
    if not a:
        return float("nan")
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den else 1.0


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mcnemar_exact(b: int, c: int) -> float:
    return stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


# ── Pipeline ────────────────────────────────────────────────────────────────

def load_answers(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        rep = re.search(r"rep(\d+)", os.path.basename(path))
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for d in data.get("details", []):
            if not d.get("reference"):
                continue
            rows.append({
                "id": d["id"], "rep": int(rep.group(1)) if rep else 1, "split": data.get("split"),
                "type": d.get("type"), "domain": d.get("domain"),
                "question": d["user_input"], "ground_truth": d["reference"], "answer": d["response"],
                "retrieved_contexts": d.get("retrieved_contexts", []),
                "numeric_precision": d.get("numeric_precision"), "latency_ms": d.get("latency_ms"),
                "token_usage": d.get("token_usage"),
                "error": d["response"].startswith("[ERRO]"),
            })
    return rows


def run(paths: list[str], out_dir: str, workers: int, contexts_chars: int, judge_passes: int,
        skip_probes: bool = False, prices: dict | None = None):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from evaluate import judge_client
    from rag_core import token_usage
    token_usage.install()
    model, client = judge_client()

    rows = load_answers(paths)
    print(f"{len(rows)} respostas, {len({r['id'] for r in rows})} perguntas, juiz={model}")
    for r in rows:
        r["indicators"] = indicator_score(r["ground_truth"], r["answer"])

    # Juiz: passada 0 com temperatura 0 (escore oficial); passadas extras medem confiabilidade.
    jobs = [(r, p) for r in rows for p in range(judge_passes)]
    temps = [0.0] + [0.7] * (judge_passes - 1)
    usage_before = token_usage.snapshot()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda rp: _judge_once(client, model, rp[0], contexts_chars, temps[rp[1]]), jobs))
    for (r, p), j in zip(jobs, results):
        r.setdefault("judge", [None] * judge_passes)[p] = j
    for r in rows:
        r["reasoning"] = reasoning_summary(r["judge"][0]) if "error" not in r["judge"][0] else None
    judge_usage = token_usage.diff(token_usage.snapshot(), usage_before)
    usage_before = token_usage.snapshot()

    # Validação do juiz: gabaritos perturbados, avaliados como se fossem respostas.
    seen, probes = set(), []
    for r in rows if not skip_probes else []:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        for kind, (text, target) in perturb(r["ground_truth"]).items():
            probes.append({**r, "answer": text, "kind": kind, "target": target})
    with ThreadPoolExecutor(workers) as ex:
        pj = list(ex.map(lambda p: _judge_once(client, model, p, contexts_chars, 0.0), probes))
    probe_rows = []
    for p, j in zip(probes, pj):
        if "error" in j:
            continue
        crit = j["notas"].get(p["target"]) if p["target"] else None
        detected = (crit is not None and crit < 2) if p["target"] else None
        flagged_any = any(v is not None and v < 2 for v in j["notas"].values()) or bool(j.get("erros"))
        probe_rows.append({"id": p["id"], "kind": p["kind"], "detected": detected,
                           "flagged_any": flagged_any, "judge": j})

    probe_usage = token_usage.diff(token_usage.snapshot(), usage_before)

    report = analyze(rows, probe_rows, judge_passes)
    report["judge_model"] = model
    report["judge_failures"] = {"answers": sum("error" in j for j in results), "probes": sum("error" in j for j in pj)}
    report["tokens"] = token_report(rows, judge_usage, probe_usage, prices or {})
    os.makedirs(out_dir, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "retrieved_contexts"} for r in rows]
    with open(os.path.join(out_dir, "scientific_eval.json"), "w", encoding="utf-8") as fh:
        json.dump({"report": report, "answers": slim, "judge_probes": probe_rows}, fh, ensure_ascii=False, indent=2, default=str)
    md = render_md(report)
    with open(os.path.join(out_dir, "relatorio_cientifico.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


def usage_cost(usage: dict, prices: dict) -> float | None:
    """US$ com prices = {modelo: [entrada_sem_cache, entrada_cache, saída]} por 1M tokens.
    Tokens de raciocínio já estão dentro de completion_tokens."""
    total, known = 0.0, False
    for model, u in (usage or {}).items():
        p = prices.get(model)
        if not p:
            continue
        known = True
        miss = u["prompt_tokens"] - u["cached_prompt_tokens"]
        total += (miss * p[0] + u["cached_prompt_tokens"] * p[1] + u["completion_tokens"] * p[2]) / 1e6
    return total if known else None


def _gen_total(r: dict) -> float | None:
    u = r.get("token_usage")
    return sum(v["prompt_tokens"] + v["completion_tokens"] for v in u.values()) if u else None


def token_report(rows: list[dict], judge_usage: dict, probe_usage: dict, prices: dict) -> dict:
    from rag_core.token_usage import merge
    gen = merge(*(r.get("token_usage") for r in rows))
    per_q = defaultdict(list)
    for r in rows:
        if _gen_total(r) is not None:
            per_q[r["id"]].append(_gen_total(r))
    buckets = {"geração (RAG)": gen, "juiz (respostas)": judge_usage, "juiz (validação)": probe_usage}
    gen_cost = usage_cost(gen, prices)
    return {
        "buckets": buckets,
        "cost_usd": {k: usage_cost(v, prices) for k, v in buckets.items()},
        "gen_tokens_per_answer": cluster_bootstrap(per_q),
        "gen_cost_per_answer_usd": gen_cost / len(rows) if rows and gen_cost is not None else None,
        "prices": prices,
    }


def _by(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def analyze(rows: list[dict], probes: list[dict], judge_passes: int) -> dict:
    ok = [r for r in rows if not r["error"] and r["reasoning"] and r["indicators"]["gt_numbers"]]
    R: dict = {"n_answers": len(rows), "n_valid": len(ok), "n_questions": len({r["id"] for r in rows}),
         "n_reps": len({r["rep"] for r in rows}), "n_errors": sum(r["error"] for r in rows)}

    def per_q(metric):
        d = defaultdict(list)
        for r in ok:
            v = metric(r)
            if v is not None:
                d[r["id"]].append(float(v))
        return d

    R["indicator_recall"] = cluster_bootstrap(per_q(lambda r: r["indicators"]["recall"]))
    R["indicator_all_correct"] = cluster_bootstrap(per_q(lambda r: r["indicators"]["all_correct"]))
    R["numeric_precision_vs_sources"] = cluster_bootstrap(per_q(lambda r: r["numeric_precision"]))
    R["reasoning_score"] = cluster_bootstrap(per_q(lambda r: r["reasoning"]["reasoning_score"]))
    R["grave_error_rate"] = cluster_bootstrap(per_q(lambda r: r["reasoning"]["has_grave_error"]))
    R["criteria"] = {c: cluster_bootstrap(per_q(lambda r, c=c: (r["judge"][0]["notas"][c] / 2)
                                                 if r["judge"][0]["notas"].get(c) is not None else None))
                     for c in CRITERIA}
    errs = defaultdict(int)
    for r in ok:
        for e in r["judge"][0].get("erros") or []:
            errs[(e.get("criterio"), e.get("gravidade"))] += 1
    R["error_counts"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(errs.items(), key=lambda kv: -kv[1])}

    # Dissociação número × interpretação (pergunta central).
    num_ok = [r["indicators"]["all_correct"] for r in ok]
    rea_ok = [not r["reasoning"]["has_grave_error"] for r in ok]
    a = sum(n and m for n, m in zip(num_ok, rea_ok)); b = sum(n and not m for n, m in zip(num_ok, rea_ok))
    c = sum((not n) and m for n, m in zip(num_ok, rea_ok)); d = sum((not n) and (not m) for n, m in zip(num_ok, rea_ok))
    R["contingency"] = {"num_ok&reason_ok": a, "num_ok&reason_err": b, "num_err&reason_ok": c, "num_err&reason_err": d}
    R["p_reason_err_given_num_ok"] = wilson(b, a + b)
    R["p_misinterpreted_flag"] = wilson(sum(r["reasoning"]["misinterpreted"] for r in ok), len(ok))
    R["mcnemar_p"] = mcnemar_exact(b, c)
    rec = [r["indicators"]["recall"] for r in ok]; rs = [r["reasoning"]["reasoning_score"] for r in ok]
    R["spearman_recall_reasoning"] = tuple(stats.spearmanr(rec, rs)) if len(set(rec)) > 1 and len(set(rs)) > 1 else (float("nan"), float("nan"))

    # Subgrupos: Kruskal-Wallis sobre médias por pergunta + Holm.
    R["subgroups"], pv = {}, {}
    for key in ("type", "domain"):
        groups = {}
        for g, rs_ in _by(ok, key).items():
            means = [np.mean([x["reasoning"]["reasoning_score"] for x in v]) for v in _by(rs_, "id").values()]
            rmeans = [np.mean([x["indicators"]["recall"] for x in v]) for v in _by(rs_, "id").values()]
            groups[g] = {"n_q": len(means), "reasoning": float(np.mean(means)), "indicator_recall": float(np.mean(rmeans)), "_m": means}
        valid = [v["_m"] for v in groups.values() if len(v["_m"]) >= 2]
        if len(valid) >= 2:
            try:
                pv[key] = stats.kruskal(*valid).pvalue
            except ValueError:  # todos os valores idênticos
                pv[key] = 1.0
        for v in groups.values():
            v.pop("_m")
        R["subgroups"][key] = groups
    R["kruskal_holm"] = {k: (pv[k], adj) for k, adj in holm(pv).items()}

    # Estabilidade entre repetições (teste-reteste do sistema).
    if R["n_reps"] > 1:
        tabs = {}
        for name, f in (("all_correct", lambda r: r["indicators"]["all_correct"]),
                        ("grave_error", lambda r: r["reasoning"]["has_grave_error"])):
            t = [[sum(1 for x in v if f(x)), sum(1 for x in v if not f(x))] for v in _by(ok, "id").values()]
            tabs[name] = fleiss_kappa(np.array(t))
        R["fleiss_kappa_reps"] = tabs
        sds = [np.std([x["reasoning"]["reasoning_score"] for x in v], ddof=1) for v in _by(ok, "id").values() if len(v) > 1]
        R["within_question_sd_reasoning"] = float(np.mean(sds)) if sds else None

    # Confiabilidade do juiz (passada 0 vs 1).
    if judge_passes > 1:
        kap = {}
        for cname in CRITERIA:
            pairs = [(r["judge"][0]["notas"][cname], r["judge"][1]["notas"][cname]) for r in ok
                     if "error" not in r["judge"][1] and r["judge"][0]["notas"].get(cname) is not None
                     and r["judge"][1]["notas"].get(cname) is not None]
            agree = sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")
            kap[cname] = (weighted_kappa([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs), agree)
        R["judge_weighted_kappa"] = kap

    # Validade do juiz (erros injetados).
    val = {}
    for kind, ps in _by(probes, "kind").items():
        if kind == "controle":
            val[kind] = {"specificity": wilson(sum(not p["flagged_any"] for p in ps), len(ps))}
        else:
            val[kind] = {"sensitivity": wilson(sum(bool(p["detected"]) for p in ps), len(ps))}
    R["judge_validation"] = val
    return R


def _f(t, pct=True):
    if t is None or any(np.isnan(x) for x in t):
        return "n/d"
    m, lo, hi = t
    s = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{s(m)} [{s(lo)}; {s(hi)}]"


def render_md(R: dict) -> str:
    L = ["# Avaliação científica — rag_principal", "",
         f"Juiz: `{R.get('judge_model')}` · {R['n_questions']} perguntas × {R['n_reps']} repetições = "
         f"{R['n_answers']} respostas ({R['n_valid']} válidas, {R['n_errors']} erros de execução). "
         "IC95%: bootstrap percentil por cluster de pergunta (B=10.000) ou Wilson para proporções. "
         f"Falhas do juiz: {R.get('judge_failures')}.", "",
         "## 1. Acerto de indicadores", "",
         "| Métrica | Estimativa [IC95%] |", "|---|---|",
         f"| Recall de indicadores do gabarito | {_f(R['indicator_recall'])} |",
         f"| Respostas com todos os indicadores corretos | {_f(R['indicator_all_correct'])} |",
         f"| Precisão numérica vs. fontes (validador) | {_f(R['numeric_precision_vs_sources'])} |", "",
         "## 2. Qualidade do raciocínio", "",
         "| Métrica | Estimativa [IC95%] |", "|---|---|",
         f"| Escore global de raciocínio (0–1) | {_f(R['reasoning_score'])} |",
         f"| Respostas com erro grave de interpretação | {_f(R['grave_error_rate'])} |"]
    L += [f"| Critério: {c} | {_f(R['criteria'][c])} |" for c in CRITERIA]
    L += ["", "Erros apontados pelo juiz (critério|gravidade: contagem): " +
          (", ".join(f"`{k}`: {v}" for k, v in R["error_counts"].items()) or "nenhum"), "",
          "## 3. Número certo × interpretação certa", "",
          "| | Raciocínio OK | Raciocínio com erro grave |", "|---|---:|---:|",
          f"| Indicadores todos corretos | {R['contingency']['num_ok&reason_ok']} | {R['contingency']['num_ok&reason_err']} |",
          f"| Algum indicador errado/ausente | {R['contingency']['num_err&reason_ok']} | {R['contingency']['num_err&reason_err']} |", "",
          f"- P(erro grave de interpretação | indicadores corretos) = {_f(R['p_reason_err_given_num_ok'])}",
          f"- Flag do juiz \"números certos, leitura errada\": {_f(R['p_misinterpreted_flag'])}",
          f"- McNemar exato (acerto numérico vs. acerto interpretativo): p = {R['mcnemar_p']:.4f}",
          f"- Spearman (recall de indicadores × escore de raciocínio): ρ = {R['spearman_recall_reasoning'][0]:.3f}, "
          f"p = {R['spearman_recall_reasoning'][1]:.4f}", "",
          "## 4. Subgrupos (médias por pergunta)", ""]
    for key, groups in R["subgroups"].items():
        L += [f"**{key}**", "", "| Grupo | n perguntas | Recall indicadores | Raciocínio |", "|---|---:|---:|---:|"]
        L += [f"| {g} | {v['n_q']} | {100 * v['indicator_recall']:.1f}% | {100 * v['reasoning']:.1f}% |"
              for g, v in sorted(groups.items(), key=lambda kv: kv[1]['reasoning'])]
        if key in R["kruskal_holm"]:
            p, adj = R["kruskal_holm"][key]
            L.append(f"\nKruskal-Wallis (raciocínio): p = {p:.4f}; Holm-ajustado = {adj:.4f}")
        L.append("")
    L += ["## 5. Confiabilidade e validade da medição", ""]
    if "fleiss_kappa_reps" in R:
        L += [f"- Estabilidade entre repetições (Fleiss κ): todos indicadores corretos = "
              f"{R['fleiss_kappa_reps']['all_correct']:.3f}; erro grave = {R['fleiss_kappa_reps']['grave_error']:.3f}; "
              f"DP intra-pergunta do raciocínio = {R['within_question_sd_reasoning']:.3f}"]
    if "judge_weighted_kappa" in R:
        L += ["- Concordância do juiz entre passadas (κ ponderado quadrático e concordância bruta, T=0 vs T=0,7; "
              "κ baixo com concordância alta indica paradoxo do kappa por notas concentradas em 2): " +
              ", ".join(f"{c} = {k:.3f} / {100 * ag:.0f}% (n={n})" for c, (k, n, ag) in R["judge_weighted_kappa"].items())]
    L += ["", "Validação do juiz por erros injetados no gabarito:", "",
          "| Perturbação | Métrica | Estimativa [IC95% Wilson] |", "|---|---|---|"]
    for kind, v in R["judge_validation"].items():
        for mname, t in v.items():
            L.append(f"| {kind} | {mname} | {_f(t)} |")
    T = R.get("tokens")
    if T:
        L += ["", "## 6. Tokens e custo (campo `usage` da API)", "",
              "| Etapa | Modelo | Chamadas | Entrada | Entrada em cache | Saída | Raciocínio | Custo US$ |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for bucket, usage in T["buckets"].items():
            for model, u in usage.items():
                c = usage_cost({model: u}, T["prices"])
                L.append(f"| {bucket} | {model} | {u['calls']} | {u['prompt_tokens']:,} | {u['cached_prompt_tokens']:,} | "
                         f"{u['completion_tokens']:,} | {u['reasoning_tokens']:,} | {'n/d' if c is None else f'{c:.4f}'} |")
        if not any(T["buckets"]["geração (RAG)"].values()):
            L.append("| geração (RAG) | — | artefatos sem `token_usage` (gerados antes da instrumentação) | | | | | |")
        per_answer = T["gen_cost_per_answer_usd"]
        L += ["", f"- Tokens de geração por resposta (entrada+saída): {_f(T['gen_tokens_per_answer'], pct=False)}",
              "- Custo de geração por resposta: " + ("n/d" if per_answer is None else f"US$ {per_answer:.5f}"),
              f"- Preços usados (US$/1M: entrada sem cache, entrada em cache, saída): {T['prices'] or 'não informados'}"]
    L += ["", "## Limitações", "",
          "- Amostra pequena (perguntas curadas); ICs largos e testes de subgrupo com baixo poder.",
          "- Juiz LLM da mesma família do gerador: possível viés de autopreferência; mitigado pela validação com erros injetados.",
          "- Casamento numérico por valor e unidade (%), sem conferência de rótulo do indicador; números do gabarito "
          "podem coincidir com outros números da resposta (viés para cima no recall).",
          "- Repetições capturam variação estocástica do gerador, não variação de corpus ou de perguntas.", ""]
    return "\n".join(L)


COMPARE_METRICS = {
    # nome: (função por resposta, maior é melhor?)
    "inferencia (0-1) [primária]": (lambda a: a["judge"][0]["notas"].get("inferencia") / 2
                                     if a["judge"][0]["notas"].get("inferencia") is not None else None, True),
    "raciocínio global (0-1)": (lambda a: a["reasoning"]["reasoning_score"], True),
    "erro grave (proporção)": (lambda a: float(a["reasoning"]["has_grave_error"]), False),
    "recall de indicadores": (lambda a: a["indicators"]["recall"], True),
    "todos indicadores corretos": (lambda a: float(bool(a["indicators"]["all_correct"])), True),
    "precisão numérica vs fontes": (lambda a: a["numeric_precision"], True),
    "tokens de geração / resposta": (_gen_total, False),
    "latência (s)": (lambda a: a["latency_ms"] / 1000 if a.get("latency_ms") else None, False),
}


def _per_question(answers: list[dict], f) -> dict[str, float]:
    d = defaultdict(list)
    for a in answers:
        if a.get("error") or not a.get("reasoning"):
            continue
        v = f(a)
        if v is not None:
            d[a["id"]].append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}


def compare(specs: list[str], out: str):
    """Comparação pareada por pergunta: braço − baseline (primeiro informado)."""
    arms = []
    for spec in specs:
        label, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            arms.append((label, json.load(fh)))
    base_label, base = arms[0]
    L = ["# Comparação de braços — rag_principal", "",
         f"Baseline: **{base_label}**. Diferença = braço − baseline, pareada por pergunta (média das repetições). "
         "IC95% bootstrap sobre perguntas (B=10.000); p de Wilcoxon pareado; Holm sobre as métricas secundárias.", ""]
    for label, arm in arms[1:]:
        rows, pvals = [], {}
        for name, (f, higher) in COMPARE_METRICS.items():
            b, a = _per_question(base["answers"], f), _per_question(arm["answers"], f)
            ids = sorted(set(a) & set(b))
            if len(ids) < 2:
                continue
            diff = np.array([a[i] - b[i] for i in ids])
            boots = diff[RNG.integers(0, len(diff), size=(B, len(diff)))].mean(1)
            try:
                p = stats.wilcoxon(diff, zero_method="zsplit").pvalue if np.any(diff) else 1.0
            except ValueError:
                p = 1.0
            rows.append((name, np.mean([b[i] for i in ids]), np.mean([a[i] for i in ids]), diff.mean(),
                         np.percentile(boots, 2.5), np.percentile(boots, 97.5), p, len(ids), higher))
            if "[primária]" not in name:
                pvals[name] = p
        adj = holm(pvals)
        L += [f"## {label}", "", "| Métrica | Baseline | Braço | Δ [IC95%] | p | p Holm | n |", "|---|---:|---:|---|---:|---:|---:|"]
        for name, bm, am, dm, lo, hi, p, n, higher in rows:
            better = (dm > 0) == higher and lo * hi > 0
            mark = " ✓" if better else (" ✗" if lo * hi > 0 else "")
            L.append(f"| {name} | {bm:.3f} | {am:.3f} | {dm:+.3f} [{lo:+.3f}; {hi:+.3f}]{mark} | {p:.4f} | "
                     f"{adj.get(name, p):.4f} | {n} |")
        gen_b = base["report"].get("tokens", {}).get("gen_cost_per_answer_usd")
        gen_a = arm["report"].get("tokens", {}).get("gen_cost_per_answer_usd")
        if gen_a is not None:
            L.append(f"\nCusto de geração por resposta: baseline {'n/d' if gen_b is None else f'US$ {gen_b:.5f}'} → braço US$ {gen_a:.5f}")
        L.append("")
    L.append("✓ = melhora com IC95% excluindo zero; ✗ = piora com IC95% excluindo zero.")
    md = "\n".join(L)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


def selftest():
    s = indicator_score("houve redução de 367 mil, ou 2,5%; 44,0% em pobreza em 2024",
                        "caiu 2,5%, menos 367 mil; 44% pobres")
    assert s["gt_numbers"] == 3 and s["all_correct"], s
    s = indicator_score("taxa de 7,8%", "taxa de 7,8 p.p.")
    assert s["all_correct"] is False, s
    s = indicator_score("taxa de 7,8%", "taxa de 7,9%")
    assert s["all_correct"] is False, s
    p = perturb("O emprego cresceu 2,0 p.p., acima da média.")
    assert "recuou" in p["direcao"][0] and "abaixo" in p["comparacao"][0] and "2,0%" in p["unidade_escala"][0], p
    assert abs(weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) - 1) < 1e-9
    assert abs(fleiss_kappa(np.array([[3, 0], [0, 3], [3, 0]])) - 1) < 1e-9
    m, lo, hi = cluster_bootstrap({"a": [1, 1], "b": [0, 0]})
    assert m == 0.5 and lo <= m <= hi
    assert holm({"x": 0.01, "y": 0.04}) == {"x": 0.02, "y": 0.04}
    u = {"m": {"calls": 1, "prompt_tokens": 1_000_000, "cached_prompt_tokens": 500_000,
               "completion_tokens": 1_000_000, "reasoning_tokens": 0, "calls_without_usage": 0}}
    assert abs(usage_cost(u, {"m": [0.2, 0.02, 1.0]}) - 1.11) < 1e-9
    assert usage_cost(u, {}) is None
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out", default="evaluation/runs/scientific")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--contexts-chars", type=int, default=12000)
    ap.add_argument("--judge-passes", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-probes", action="store_true", help="pula a validação do juiz por erros injetados")
    ap.add_argument("--prices", default=None,
                    help='JSON {"modelo": [entrada_sem_cache, entrada_cache, saída]} em US$/1M tokens')
    ap.add_argument("--compare", nargs="+", metavar="ROTULO=scientific_eval.json",
                    help="compara braços já avaliados; o primeiro é o baseline")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.compare:
        return compare(a.compare, os.path.join(a.out, "comparacao.md"))
    # checkpoints de geração duplicam as respostas do artefato final
    paths = [p for pat in a.inputs for p in glob.glob(pat) if "checkpoint" not in os.path.basename(p)]
    if not paths:
        ap.error("informe artefatos de geração")
    random.seed(42)
    run(paths, a.out, a.workers, a.contexts_chars, a.judge_passes,
        skip_probes=a.skip_probes, prices=json.loads(a.prices) if a.prices else None)


if __name__ == "__main__":
    main()
