import json

b = json.load(open('evaluation/baseline_judged_deepseek.json', encoding='utf-8'))
c = json.load(open('evaluation/results_dev_candidate.json', encoding='utf-8'))
b_map = {x['id']: x for x in b['details']}

print(f"{'ID':<18} | {'Base Faith':<10} | {'Cand Faith':<10} | {'Delta F':<7} | {'Base CRec':<9} | {'Cand CRec':<9} | {'Base Num':<8} | {'Cand Num':<8}")
print("-" * 95)

for item in c['details']:
    qid = item['id']
    b_item = b_map.get(qid, {})
    
    bf = b_item.get('faithfulness')
    cf = item.get('faithfulness')
    df = (cf - bf) if (cf is not None and bf is not None) else 0.0
    
    br = b_item.get('context_recall')
    cr = item.get('context_recall')
    
    bn = b_item.get('numeric_precision')
    cn = item.get('numeric_precision')
    
    bf_s = f"{bf:.3f}" if bf is not None else "N/A"
    cf_s = f"{cf:.3f}" if cf is not None else "N/A"
    df_s = f"{df:+.3f}" if (cf is not None and bf is not None) else "---"
    br_s = f"{br:.3f}" if br is not None else "N/A"
    cr_s = f"{cr:.3f}" if cr is not None else "N/A"
    bn_s = f"{bn:.3f}" if bn is not None else "N/A"
    cn_s = f"{cn:.3f}" if cn is not None else "N/A"
    
    print(f"{qid:<18} | {bf_s:<10} | {cf_s:<10} | {df_s:<7} | {br_s:<9} | {cr_s:<9} | {bn_s:<8} | {cn_s:<8}")

print("-" * 95)
print(f"{'MÉDIA':<18} | {b['summary']['faithfulness']:<10.3f} | {c['summary']['faithfulness']:<10.3f} | {c['summary']['faithfulness']-b['summary']['faithfulness']:<+7.3f} | {b['summary']['context_recall']:<9.3f} | {c['summary']['context_recall']:<9.3f} | {b['summary']['numeric_precision']:<8.3f} | {c['summary']['numeric_precision']:<8.3f}")

