#!/usr/bin/env python3
"""XA reeval over all k*/ subdirs of BASE: area(DC)+power(XA)+verilator MED(取自 best_info)。
同口径 reeval_xa_generic.py，但不写死 KS，自动 glob 现有 k*/（适配只跑了部分 k 的 sweep）。
用法: python reeval_xa_glob.py <BASE_REL_DIR> [workers]。输出 {BASE}/reeval_xa.csv。"""
import csv, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(_ROOT); sys.path.insert(0, _ROOT)
from run_power_sweep import evaluate_single_routing

BASE = sys.argv[1]
MAXW = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("REEVAL_WORKERS", "6"))
jobs = []
for d in sorted(glob.glob(f"{BASE}/k*/")):
    kk = os.path.basename(d.rstrip("/"))
    bi, mv = os.path.join(d, "best_info.json"), os.path.join(d, "MUL.v")
    if not (os.path.exists(bi) and os.path.exists(mv)):
        print("skip (no best_info/MUL.v):", kk); continue
    med = (json.load(open(bi)).get("measured_error") or {}).get("med")
    jobs.append((kk, med, open(mv).read()))
print(f"{BASE}: {len(jobs)} designs -> {[j[0] for j in jobs]}", flush=True)
res = {}
with ThreadPoolExecutor(max_workers=MAXW) as ex:
    futs = {ex.submit(evaluate_single_routing, i, c, 16, 1.5): (kk, med) for i, (kk, med, c) in enumerate(jobs)}
    for fut in as_completed(futs):
        kk, med = futs[fut]; r = fut.result(); res[kk] = (med, r)
        print(f"{kk}: success={r.get('success')} area={r.get('area')} power_mw={r.get('power_mw')} delay={r.get('delay')}", flush=True)
out = f"{BASE}/reeval_xa.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["design", "med", "area_dc", "power_xa_mw", "delay", "success"])
    for kk, _, _ in jobs:
        med, r = res[kk]; w.writerow([kk, med, r.get("area"), r.get("power_mw"), r.get("delay"), r.get("success")])
print("saved ->", out)
