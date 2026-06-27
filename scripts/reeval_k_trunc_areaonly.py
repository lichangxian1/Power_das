#!/usr/bin/env python3
"""XA reeval 纯k截断 only-area baseline 的 10 个 best RTL，取同口径 power(XA)+area(DC)。
在 ubuntu1 跑（das env），经 run_power_sweep.evaluate_single_routing SSH 到 EDA host。
输出 outputs/2026-06-27_k_trunc_areaonly/reeval_xa.csv（同 prior reeval 格式）。
"""
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
from run_power_sweep import evaluate_single_routing  # noqa: E402

BASE = "outputs/2026-06-27_k_trunc_areaonly"
KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MAX_WORKERS = int(os.environ.get("REEVAL_WORKERS", "6"))

jobs = []
for k in KS:
    d = f"{BASE}/k{k:02d}"
    info = json.load(open(f"{d}/best_info.json"))
    med = info["measured_error"]["med"]
    content = open(f"{d}/MUL.v").read()
    jobs.append((k, med, content))

results = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(evaluate_single_routing, i, c, 16, 1.5): (k, med)
            for i, (k, med, c) in enumerate(jobs)}
    for fut in as_completed(futs):
        k, med = futs[fut]
        r = fut.result()
        results[k] = (med, r)
        print(f"k{k:02d}: success={r.get('success')} area={r.get('area')} "
              f"power_mw={r.get('power_mw')} delay={r.get('delay')}", flush=True)

out = f"{BASE}/reeval_xa.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["design", "med", "area_dc", "power_xa_mw", "delay", "success"])
    for k in KS:
        med, r = results[k]
        w.writerow([f"k{k:02d}", med, r.get("area"), r.get("power_mw"),
                    r.get("delay"), r.get("success")])
print(f"saved -> {out}")
