#!/usr/bin/env python3
"""Coordinate-descent refinement of Algorithm-1 greedy result (paper's
"trial-and-error" step): re-optimize each (stage,col) slot given the FULL
assignment, repeat until a pass yields no change (max 3 passes).

Usage: python3 refine_assignment.py [n_vectors=2000000]
Reads assignment16.json -> writes assignment16_refined.json
"""
import json
import sys
import numpy as np
from sayadi_common import build_schedule, simulate, metrics
from run_algorithm1 import CAND, vectors, TABLE8

n_vec = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
a, b = vectors(16, n_vec)
exact = a.astype(np.int64) * b.astype(np.int64)
maxp = (2 ** 16 - 1) ** 2

with open('assignment16.json') as f:
    raw = json.load(f)
out = {}
for mul, fam in (('mul1', 'ACFGI'), ('mul2', 'ACFGII')):
    asg = {tuple(int(x) for x in k.split(',')): tuple(v) for k, v in raw[mul].items()}
    boxes, final_cols, _ = build_schedule(16, fam)
    slots = sorted(asg.keys())

    def nmed_of(assignment):
        ap = simulate(16, boxes, final_cols, assignment, a, b)
        return float(np.mean(np.abs(ap - exact))) / maxp

    cur = nmed_of(asg)
    print(f"-- {mul}: initial NMED={cur:.5f}")
    for it in range(3):
        changed = 0
        for slot in slots:
            family = asg[slot][0]
            best_k, best = asg[slot][1], cur
            for k in CAND[family]:
                if k == asg[slot][1]:
                    continue
                trial = dict(asg); trial[slot] = (family, k)
                v = nmed_of(trial)
                if v < best - 1e-15:
                    best_k, best = k, v
            if best_k != asg[slot][1]:
                print(f"   pass{it} s{slot[0]}c{slot[1]}: {family}-{asg[slot][1]} -> {family}-{best_k}  NMED {cur:.5f}->{best:.5f}")
                asg[slot] = (family, best_k); cur = best; changed += 1
        print(f"   pass{it}: {changed} changes, NMED={cur:.5f}")
        if not changed:
            break
    av, bv = vectors(16, 10_000_000, seed=777)
    ap = simulate(16, boxes, final_cols, asg, av, bv)
    m = metrics(16, ap, av, bv)
    print(f"[{mul}-16 refined(10M)] ER={m['ER%']:.2f}% NMED={m['NMED']:.4f} MRED={m['MRED']:.3f}")
    print(f"   paper: {TABLE8[mul]}")
    out[mul] = {f"{s[0]},{s[1]}": list(v) for s, v in asg.items()}

with open('assignment16_refined.json', 'w') as f:
    json.dump(out, f, indent=1)
print("saved assignment16_refined.json")
