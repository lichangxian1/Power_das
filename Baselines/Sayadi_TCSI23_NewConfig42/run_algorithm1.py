#!/usr/bin/env python3
"""Algorithm 1 (greedy NMED per (stage,column) slot) + 8-bit validation.

Usage:
  python run_algorithm1.py check8      # schedule dump + paper-Fig9/10 assignment metrics vs Table VII
  python run_algorithm1.py greedy8     # run greedy on 8-bit, compare choices with Fig9/10
  python run_algorithm1.py greedy16    # run greedy on 16-bit, save assignment json + metrics vs Table VIII
"""
import json
import sys
import numpy as np
from sayadi_common import build_schedule, simulate, metrics, zones

CAND = {'ACFGI': range(1, 5), 'ACFGII': range(1, 13), 'AC6G': range(1, 17)}

# Fig.9 / Fig.10 transcription (stage, col) -> (family, k)
PAPER_8 = {
    'mul1': {(1, 5): ('ACFGI', 4), (1, 6): ('ACFGI', 4), (1, 7): ('ACFGI', 4),
             (1, 8): ('ACFGI', 4), (1, 9): ('ACFGI', 4), (1, 10): ('ACFGI', 2),
             (1, 11): ('AC6G', 12), (1, 12): ('AC6G', 14),
             (2, 5): ('ACFGI', 4), (2, 6): ('ACFGI', 4), (2, 7): ('ACFGI', 4),
             (2, 8): ('ACFGI', 4), (2, 9): ('ACFGI', 4), (2, 10): ('ACFGI', 3),
             (2, 11): ('AC6G', 7), (2, 13): ('AC6G', 7)},
    'mul2': {(1, 5): ('ACFGII', 1), (1, 6): ('ACFGII', 1), (1, 7): ('ACFGII', 1),
             (1, 8): ('ACFGII', 1), (1, 9): ('ACFGII', 5), (1, 10): ('ACFGII', 11),
             (1, 11): ('AC6G', 12), (1, 12): ('AC6G', 14),
             (2, 5): ('ACFGII', 1), (2, 6): ('ACFGII', 1), (2, 7): ('ACFGII', 1),
             (2, 8): ('ACFGII', 1), (2, 9): ('ACFGII', 1), (2, 10): ('ACFGII', 10),
             (2, 11): ('AC6G', 7), (2, 13): ('AC6G', 7)},
}
TABLE7 = {'mul1': {'ER%': 99.93, 'NMED': 0.018, 'MRED': 0.509, 'MaxED': 7120},
          'mul2': {'ER%': 98.86, 'NMED': 0.017, 'MRED': 0.151, 'MaxED': 7148}}
TABLE8 = {'mul1': {'ER%': 100.0, 'NMED': 0.010, 'MRED': 0.119},
          'mul2': {'ER%': 99.98, 'NMED': 0.009, 'MRED': 0.066}}


def vectors(N, n=None, seed=12345):
    if N == 8:
        a, b = np.meshgrid(np.arange(256, dtype=np.uint32),
                           np.arange(256, dtype=np.uint32))
        return a.ravel(), b.ravel()
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 1 << N, n, dtype=np.uint32),
            rng.integers(0, 1 << N, n, dtype=np.uint32))


def dump_schedule(N, fam):
    boxes, final_cols, ns = build_schedule(N, fam)
    print(f"== N={N} {fam}: zones(trunc<= {zones(N)[0]}, mid<= {zones(N)[1]}, up<= {zones(N)[2]}), {ns} stages")
    for st in range(1, ns + 1):
        per = {}
        for bx in boxes:
            if bx.stage == st:
                per.setdefault(bx.col, []).append((len(bx.inputs), bx.family if not bx.searchable else '?'))
        print(f" stage{st}: " + "  ".join(f"c{c}:{v}" for c, v in sorted(per.items())))
    print(" final rows: " + " ".join(f"c{c}:{len(s)}" for c, s in sorted(final_cols.items()) if s))
    return boxes, final_cols, ns


def greedy(N, fam, a, b, verbose=True):
    boxes, final_cols, ns = build_schedule(N, fam)
    slots = []
    for bx in boxes:
        if bx.searchable and bx.slot not in [s for s, _ in slots]:
            slots.append((bx.slot, 'AC6G' if bx.family == 'AC6G' else fam))
    slots.sort(key=lambda t: (t[0][0], t[0][1]))
    asg = {}
    maxp = (2 ** N - 1) ** 2
    exact = a.astype(np.int64) * b.astype(np.int64)
    for slot, family in slots:
        best_k, best_nmed = None, None
        for k in CAND[family]:
            asg[slot] = (family, k)
            approx = simulate(N, boxes, final_cols, asg, a, b)
            nmed = float(np.mean(np.abs(approx - exact))) / maxp
            if best_nmed is None or nmed < best_nmed - 1e-15:
                best_k, best_nmed = k, nmed
        asg[slot] = (family, best_k)
        if verbose:
            print(f"  slot s{slot[0]} c{slot[1]}: {family}-{best_k}  NMED={best_nmed:.6f}")
    return asg, boxes, final_cols


def report(name, N, boxes, final_cols, asg, a, b, ref):
    approx = simulate(N, boxes, final_cols, asg, a, b)
    m = metrics(N, approx, a, b)
    print(f"[{name}] ER={m['ER%']:.2f}% NMED={m['NMED']:.4f} MRED={m['MRED']:.3f} "
          f"MaxED={m['MaxED']} MED={m['MED']:.1f}")
    print(f"   paper: {ref}")
    return m


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check8'
    if mode == 'check8':
        a, b = vectors(8)
        for mul, fam in (('mul1', 'ACFGI'), ('mul2', 'ACFGII')):
            boxes, final_cols, _ = dump_schedule(8, fam)
            report(f"{mul}-8 paper-fig", 8, boxes, final_cols, PAPER_8[mul], a, b, TABLE7[mul])
    elif mode == 'greedy8':
        a, b = vectors(8)
        for mul, fam in (('mul1', 'ACFGI'), ('mul2', 'ACFGII')):
            print(f"-- greedy {mul}-8")
            asg, boxes, final_cols = greedy(8, fam, a, b)
            diff = {s: (asg.get(s), PAPER_8[mul].get(s)) for s in set(asg) | set(PAPER_8[mul])
                    if asg.get(s) != PAPER_8[mul].get(s)}
            print("  vs Fig9/10 diffs:", diff if diff else "NONE (exact match)")
            report(f"{mul}-8 greedy", 8, boxes, final_cols, asg, a, b, TABLE7[mul])
    elif mode == 'greedy16':
        n_greedy = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
        a, b = vectors(16, n_greedy)
        out = {}
        for mul, fam in (('mul1', 'ACFGI'), ('mul2', 'ACFGII')):
            print(f"-- greedy {mul}-16 ({n_greedy} vectors)")
            asg, boxes, final_cols = greedy(16, fam, a, b)
            av, bv = vectors(16, 10_000_000, seed=777)
            report(f"{mul}-16 greedy(10M eval)", 16, boxes, final_cols, asg, av, bv, TABLE8[mul])
            out[mul] = {f"{s[0]},{s[1]}": list(v) for s, v in asg.items()}
        with open('assignment16.json', 'w') as f:
            json.dump(out, f, indent=1)
        print("saved assignment16.json")
