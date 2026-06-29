#!/usr/bin/env python3
"""通用 XA reeval：对 {BASE}/k{kk}/MUL.v 取同口径 area(DC)+power(XA)+verilator MED。
用法: python reeval_xa_generic.py <BASE_REL_DIR>。输出 {BASE}/reeval_xa.csv。"""
import csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(_ROOT); sys.path.insert(0,_ROOT)
from run_power_sweep import evaluate_single_routing
BASE = sys.argv[1]
KS = [2,4,6,8,10,12,14,16,18,20]
MAXW = int(os.environ.get("REEVAL_WORKERS","6"))
jobs=[]
for k in KS:
    d=f"{BASE}/k{k:02d}"; info=json.load(open(f"{d}/best_info.json"))
    med=(info.get("measured_error") or {}).get("med")
    jobs.append((k,med,open(f"{d}/MUL.v").read()))
res={}
with ThreadPoolExecutor(max_workers=MAXW) as ex:
    futs={ex.submit(evaluate_single_routing,i,c,16,1.5):(k,med) for i,(k,med,c) in enumerate(jobs)}
    for fut in as_completed(futs):
        k,med=futs[fut]; r=fut.result(); res[k]=(med,r)
        print(f"k{k:02d}: success={r.get('success')} area={r.get('area')} power_mw={r.get('power_mw')} delay={r.get('delay')}",flush=True)
out=f"{BASE}/reeval_xa.csv"
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["design","med","area_dc","power_xa_mw","delay","success"])
    for k in KS:
        med,r=res[k]; w.writerow([f"k{k:02d}",med,r.get("area"),r.get("power_mw"),r.get("delay"),r.get("success")])
print("saved ->",out)
