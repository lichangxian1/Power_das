#!/usr/bin/env python3
"""Calibrate the Zhang reduction structure against paper Table III/IV (NMED/MRED)."""
import itertools
import random
from zhang_core import GoldenOps, reduce_columns


def pp_columns(a, b, N):
    cols = [[] for _ in range(2 * N)]
    for i in range(N):
        bi = (b >> i) & 1
        for j in range(N):
            cols[i + j].append(((a >> j) & 1) & bi)
    return cols


def product(a, b, N, cfg):
    cols = pp_columns(a, b, N)
    for k in cfg["trunc"]:
        cols[k] = []
    fin = reduce_columns(GoldenOps(), cols, cfg)
    v = cfg["const"]
    for k in range(len(fin)):
        v += sum(fin[k]) << k
    return v


def metrics(N, cfg, sample=None, seed=1):
    Dmax = 1 << (2 * N)
    if sample is None and N <= 8:
        pairs = ((a, b) for a in range(1 << N) for b in range(1 << N)); total = 1 << (2 * N)
    else:
        rng = random.Random(seed); n = sample
        pairs = ((rng.randrange(1 << N), rng.randrange(1 << N)) for _ in range(n)); total = n
    ae = se = sw = rn = mx = 0; rs = 0.0
    for (a, b) in pairs:
        ex = a * b; ap = product(a, b, N, cfg); e = ap - ex
        if e: sw += 1
        ae += abs(e); se += e; mx = max(mx, abs(e))
        if ex: rs += abs(e) / ex; rn += 1
    return dict(ER=sw / total * 100, NMED=(ae / total) / Dmax, MRED=rs / rn,
                MED=ae / total, bias=se / total, maxerr=mx)


def base(N):
    if N == 8:
        return {"trunc": set(range(4)), "const": 6, "inexact": {4, 5, 6, 7}}
    return {"trunc": set(range(6)), "const": 30, "inexact": set(range(6, 16))}


if __name__ == "__main__":
    print("== Proposed (uniform) 8b ==  target NMED1.90e-3 MRED4.27%")
    c = base(8); c.update(hybrid=False, and_all=False, espo_cols=set(), and_cols=set())
    m = metrics(8, c)
    print(f"   NMED={m['NMED']*1e3:.3f}e-3 MRED={m['MRED']*1e2:.3f}% MED={m['MED']:.1f} bias={m['bias']:+.1f}")

    print("== ProposedH 8b ==  target NMED1.14e-3 MRED2.59%")
    inex = [4, 5, 6, 7]
    rows = []
    for espo in [set(inex), {5, 6, 7}, {6, 7}, {6}, {4, 6}]:
        for ar in range(0, 3):
            for andc in itertools.combinations(inex, ar):
                c = base(8); c.update(hybrid=True, and_all=False,
                                      espo_cols=espo, and_cols=set(andc))
                m = metrics(8, c)
                nm, mr = m['NMED'] * 1e3, m['MRED'] * 1e2
                sc = abs(nm - 1.14) / 1.14 + abs(mr - 2.59) / 2.59
                rows.append((sc, espo, set(andc), nm, mr, m['MED'], m['bias']))
    rows.sort(key=lambda x: x[0])
    for sc, espo, andc, nm, mr, med, bias in rows[:6]:
        print(f"   espo={sorted(espo)} and={sorted(andc)}: NMED={nm:.3f}e-3 "
              f"MRED={mr:.3f}% MED={med:.1f} bias={bias:+.1f} score={sc:.3f}")
