import json
import os

base = json.load(open('medicoes/probe_ragas_baseline_8c988ec.json', encoding='utf-8'))
cand = json.load(open('evaluation/runs/candidate_selected/generation.json', encoding='utf-8'))
gd = {d['id']: d for d in json.load(open('data/golden_dataset_dev.json', encoding='utf-8'))}

print(f"{'ID':<18} | {'Exp':<4} | {'Base Hit':<8} | {'Cand Hit':<8} | {'Base Ctx':<8} | {'Cand Ctx':<8}")
print("-" * 65)

total_exp = 0
total_base_hits = 0
total_cand_hits = 0

for d_b in base['details']:
    qid = d_b['id']
    d_c = next((x for x in cand['details'] if x.get('id') == qid), None)
    exp = gd.get(qid, {}).get('source_files', [])
    
    b_ctx = " ".join(d_b.get('retrieved_contexts', []))
    c_ctx = " ".join(d_c.get('retrieved_contexts', [])) if d_c else ""
    
    # Verifica tanto pelo nome do arquivo quanto por substrings relevantes do arquivo
    b_hit = 0
    c_hit = 0
    for f in exp:
        base_name = os.path.basename(f)
        # remove extensão .pdf para busca mais flexível
        stem = os.path.splitext(base_name)[0]
        if base_name in b_ctx or stem in b_ctx:
            b_hit += 1
        if base_name in c_ctx or stem in c_ctx:
            c_hit += 1
            
    total_exp += len(exp)
    total_base_hits += b_hit
    total_cand_hits += c_hit
    
    b_chunks = len(d_b.get('retrieved_contexts', []))
    c_chunks = len(d_c.get('retrieved_contexts', [])) if d_c else 0
    print(f"{qid:<18} | {len(exp):<4} | {b_hit}/{len(exp):<6} | {c_hit}/{len(exp):<6} | {b_chunks:<8} | {c_chunks:<8}")

print("-" * 65)
print(f"{'TOTAL':<18} | {total_exp:<4} | {total_base_hits}/{total_exp:<6} | {total_cand_hits}/{total_exp:<6}")
