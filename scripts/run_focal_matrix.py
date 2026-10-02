import os
import sys
import json
import subprocess

CONFIGS = {
    "A_controle": {
        "desc": "Síntese: default | Interp: default | Extract: default",
        "env": {
            "RAG_LLM_REASONING_EFFORT": "",
            "RAG_INTERP_REASONING_EFFORT": "",
            "RAG_EXTRACT_REASONING_EFFORT": "",
        },
    },
    "B_mecanica": {
        "desc": "Síntese: default | Interp: none | Extract: none",
        "env": {
            "RAG_LLM_REASONING_EFFORT": "",
            "RAG_INTERP_REASONING_EFFORT": "none",
            "RAG_EXTRACT_REASONING_EFFORT": "none",
        },
    },
    "C_extracao": {
        "desc": "Síntese: default | Interp: default | Extract: none",
        "env": {
            "RAG_LLM_REASONING_EFFORT": "",
            "RAG_INTERP_REASONING_EFFORT": "",
            "RAG_EXTRACT_REASONING_EFFORT": "none",
        },
    },
    "D_atual": {
        "desc": "Síntese: none | Interp: none | Extract: none",
        "env": {
            "RAG_LLM_REASONING_EFFORT": "none",
            "RAG_INTERP_REASONING_EFFORT": "none",
            "RAG_EXTRACT_REASONING_EFFORT": "none",
        },
    },
}

def analyze_pib_response(text: str) -> dict:
    has_quarterly_rates = any(x in text for x in ["1,8%", "1,80%", "1,3%", "1,30%"])
    has_april_rates = any(x in text for x in ["-1,1%", "-1,10%", "1,10% frente", "1,1% frente"])
    mentions_april = "abril" in text.lower()
    return {
        "has_quarterly_correct": has_quarterly_rates,
        "has_april_incorrect": has_april_rates,
        "mentions_april": mentions_april,
    }

def analyze_demography_response(text: str, retrieved_files: list) -> dict:
    recovered_2024 = any("2024" in f for f in retrieved_files)
    is_refusal = any(kw in text.lower() for kw in ["não estão sustentadas", "não há base", "dados recuperados referem-se"])
    return {
        "recovered_2024": recovered_2024,
        "is_refusal": is_refusal,
    }

def run_matrix(repetitions=2):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_bin = sys.executable
    out_dir = os.path.join(base_dir, "evaluation", "runs", "matrix_foco")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for cfg_key, cfg_info in CONFIGS.items():
        print(f"\n{'#' * 65}")
        print(f" CONFIGURAÇÃO: {cfg_key} — {cfg_info['desc']}")
        print(f"{'#' * 65}")

        for rep in range(1, repetitions + 1):
            out_file = os.path.join(out_dir, f"{cfg_key}_rep{rep}.json")
            print(f"\n--- [Executando {cfg_key} Repetição {rep}/{repetitions}] ---")

            run_env = os.environ.copy()
            run_env.update(cfg_info["env"])

            cmd = [
                python_bin,
                "evaluate.py",
                "--split", "dev",
                "--ids", "dev_economy_01,dev_demography_03",
                "--generation-only",
                "--seed", "42",
                "--output", out_file,
            ]

            res = subprocess.run(cmd, cwd=base_dir, env=run_env)
            if res.returncode != 0:
                print(f"[ERRO] Falha na execução da repetição {rep} de {cfg_key}")
                continue

            if not os.path.exists(out_file):
                print(f"[ERRO] Arquivo de saída {out_file} não foi gerado")
                continue

            with open(out_file, encoding="utf-8") as f:
                data = json.load(f)

            details_map = {d["id"]: d for d in data.get("details", [])}

            pib = details_map.get("dev_economy_01", {})
            demo = details_map.get("dev_demography_03", {})

            pib_text = pib.get("response", "")
            pib_diag = analyze_pib_response(pib_text)

            demo_text = demo.get("response", "")
            demo_contexts = demo.get("retrieved_contexts", [])
            demo_files = []
            for ctx in demo_contexts:
                for line in ctx.splitlines():
                    if line.startswith("Fonte:"):
                        demo_files.append(line.replace("Fonte:", "").strip())
            demo_diag = analyze_demography_response(demo_text, demo_files)

            row = {
                "config": cfg_key,
                "rep": rep,
                "pib_prec": pib.get("numeric_precision"),
                "pib_lat_s": round(pib.get("latency_ms", 0) / 1000, 2),
                "pib_ok": pib_diag["has_quarterly_correct"] and not pib_diag["has_april_incorrect"],
                "demo_prec": demo.get("numeric_precision"),
                "demo_lat_s": round(demo.get("latency_ms", 0) / 1000, 2),
                "demo_rec_2024": demo_diag["recovered_2024"],
                "demo_refusal": demo_diag["is_refusal"],
                "p50_s": round(data.get("summary", {}).get("latency_p50_ms", 0) / 1000, 2),
                "p95_s": round(data.get("summary", {}).get("latency_p95_ms", 0) / 1000, 2),
            }
            summary_rows.append(row)

    print("\n" + "=" * 80)
    print(" RESUMO DA MATRIZ EXPERIMENTAL")
    print("=" * 80)
    fmt = "{:<12} | {:<4} | {:<9} | {:<10} | {:<10} | {:<10} | {:<8} | {:<8}"
    print(fmt.format("Config", "Rep", "PIB Corr.", "PIB Prec.", "Demo Rec24", "Demo Prec.", "p50 (s)", "p95 (s)"))
    print("-" * 80)
    for r in summary_rows:
        pib_status = "OK (1T26)" if r["pib_ok"] else "FAIL (Abr)"
        demo_status = "Sim" if r["demo_rec_2024"] else ("Recusa" if r["demo_refusal"] else "Não")
        print(fmt.format(
            r["config"],
            r["rep"],
            pib_status,
            f"{r['pib_prec']:.3f}" if r["pib_prec"] is not None else "N/A",
            demo_status,
            f"{r['demo_prec']:.3f}" if r["demo_prec"] is not None else "N/A",
            f"{r['p50_s']:.2f}",
            f"{r['p95_s']:.2f}",
        ))

    summary_path = os.path.join(out_dir, "matrix_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"\nResumo gravado em: {summary_path}")

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_matrix(repetitions=reps)

