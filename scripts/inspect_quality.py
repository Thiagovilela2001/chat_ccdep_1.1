import json

data = json.load(open('evaluation/runs/candidate_selected/generation.json', encoding='utf-8'))
base = json.load(open('medicoes/probe_ragas_baseline_8c988ec.json', encoding='utf-8'))
gd = {x['id']: x for x in json.load(open('data/golden_dataset_dev.json', encoding='utf-8'))}

base_by_id = {x['id']: x for x in base['details']}

print(f"{'ID':<18} | {'Dominio':<18} | {'Tipo':<11} | {'Base Num':<8} | {'Cand Num':<8} | {'Delta':<7} | {'Status Factual'}")
print("-" * 105)

for item in data['details']:
    qid = item['id']
    domain = item.get('domain', '?')
    qtype = item.get('type', '?')
    
    cand_num = item.get('numeric_precision')
    b_item = base_by_id.get(qid, {})
    base_num = b_item.get('numeric_precision')
    
    cand_num_str = f"{cand_num:.3f}" if cand_num is not None else "N/A"
    base_num_str = f"{base_num:.3f}" if base_num is not None else "N/A"
    
    if cand_num is not None and base_num is not None:
        delta = cand_num - base_num
        delta_str = f"{delta:+.3f}"
    else:
        delta_str = "---"
        
    # Análise de alinhamento com Ground Truth
    resp = item.get('response', '')
    gt = gd.get(qid, {}).get('ground_truth', '')
    
    status = "OK"
    if qid == "dev_economy_01":
        if "1,8%" in resp and "1,3%" in resp and "0,6%" in resp:
            status = "EXCELENTE (PIB 1T 100% correto)"
        else:
            status = "ATENÇÃO"
    elif qid == "dev_demography_03":
        if "não" in resp.lower() or "não estão sustentadas" in resp.lower() or "não há" in resp.lower():
            status = "RECUSA SEGURA (sem alucinação)"
        else:
            status = "ALUCINAÇÃO"
    elif delta_str != "---" and float(delta_str) > 0.05:
        status = f"MELHOROU (+{float(delta_str):.2f})"
    elif delta_str != "---" and float(delta_str) < -0.05:
        status = f"PIOROU ({float(delta_str):.2f})"
    else:
        status = "ESTÁVEL / ALINHADO"
        
    print(f"{qid:<18} | {domain:<18} | {qtype:<11} | {base_num_str:<8} | {cand_num_str:<8} | {delta_str:<7} | {status}")

print("-" * 105)
print(f"Resumo: Precisao Numerica Media = {data['summary']['numeric_precision']:.3f} (Baseline = {base['summary']['numeric_precision']:.3f})")
