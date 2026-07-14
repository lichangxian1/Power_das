#!/usr/bin/env python3
"""Sayadi-inspired sub-std approximate CT42 design-space enumeration.

目标：4 入 (a,b,c,d) → (sum, carry, cout) 权重 (1,2,2)，standalone DC 面积
< 原生 2xFA1D0 = 5.712 µm² 的近似 4:2 cell（substd 硬约束，见
select_substd_cells.py / OUTER_CELL_SEARCH.md §3.2.7）。

方法（受 Sayadi TCSI'23 启发）：放弃 parity 型 sum（XOR 富集 = comp42n 全军
覆没的根因），在门级模板池（literal/2-3-4 输入门/AO/OA 复合/深度2组合）上
枚举 (sum, carry) 组合 + cout∈{0,便宜项}，精确计算 P(1)=0.25 下的
wae/bias/er/maxe，用 tcbn28 12T 粗成本表估面积，输出 (wae, est_area) 帕累托
与 P/N 平衡候选清单。最终以 dc_char.tcl 同口径 DC 表征裁决。

用法: python -m Appr_Comp.cellsolver.enum_substd42 [--top 40]
"""
import argparse
from itertools import combinations

import numpy as np

BITS = np.array([[(p >> i) & 1 for p in range(16)] for i in range(4)], dtype=np.uint8)
# 输入 i∈{0,1,2,3} = a,b,c,d；pattern p 的第 i 位 = 变量 i 的取值
NAMES = "abcd"
PPAT = np.array([(0.25 ** bin(p).count("1")) * (0.75 ** (4 - bin(p).count("1")))
                 for p in range(16)])
VTRUE = np.array([bin(p).count("1") for p in range(16)], dtype=np.int8)

# tcbn28hpcplusbwp12t40p140 粗面积表(µm², LEF 量级；最终以 DC 表征为准)
COST = {"lit": 0.0, "not": 0.336, "and2": 0.7, "or2": 0.7, "nand2": 0.5,
        "nor2": 0.5, "xor2": 1.5, "xnor2": 1.5, "and3": 0.85, "or3": 0.85,
        "nand3": 0.7, "nor3": 0.7, "and4": 1.0, "or4": 1.0, "nand4": 0.85,
        "nor4": 0.85, "ao22": 1.0, "oa22": 1.0, "aoi22": 0.85, "oai22": 0.85,
        "maj3": 1.4, "const": 0.2, "op2": 0.7, "opx": 1.5}


def tt_of(fn):
    return int(sum((1 << p) for p in range(16) if fn(*(int(BITS[i][p]) for i in range(4)))))


def add(pool, tt, cost, expr):
    old = pool.get(tt)
    if old is None or cost < old[0]:
        pool[tt] = (cost, expr)


def build_pool():
    pool = {}
    add(pool, 0, COST["const"], "1'b0")
    add(pool, 0xFFFF, COST["const"], "1'b1")
    for i in range(4):
        add(pool, tt_of(lambda *x, i=i: x[i]), COST["lit"], NAMES[i])
        add(pool, tt_of(lambda *x, i=i: 1 - x[i]), COST["not"], f"~{NAMES[i]}")
    # 2 输入门(全对)
    for i, j in combinations(range(4), 2):
        vi, vj = NAMES[i], NAMES[j]
        add(pool, tt_of(lambda *x: x[i] & x[j]), COST["and2"], f"({vi} & {vj})")
        add(pool, tt_of(lambda *x: x[i] | x[j]), COST["or2"], f"({vi} | {vj})")
        add(pool, tt_of(lambda *x: 1 - (x[i] & x[j])), COST["nand2"], f"~({vi} & {vj})")
        add(pool, tt_of(lambda *x: 1 - (x[i] | x[j])), COST["nor2"], f"~({vi} | {vj})")
        add(pool, tt_of(lambda *x: x[i] ^ x[j]), COST["xor2"], f"({vi} ^ {vj})")
        add(pool, tt_of(lambda *x: 1 - (x[i] ^ x[j])), COST["xnor2"], f"~({vi} ^ {vj})")
    # 3/4 输入 AND/OR 族 + MAJ3
    for r, tag in ((3, "3"), (4, "4")):
        for c in combinations(range(4), r):
            names = [NAMES[k] for k in c]
            add(pool, tt_of(lambda *x: int(all(x[k] for k in c))), COST[f"and{tag}"],
                "(" + " & ".join(names) + ")")
            add(pool, tt_of(lambda *x: int(any(x[k] for k in c))), COST[f"or{tag}"],
                "(" + " | ".join(names) + ")")
            add(pool, tt_of(lambda *x: 1 - int(all(x[k] for k in c))), COST[f"nand{tag}"],
                "~(" + " & ".join(names) + ")")
            add(pool, tt_of(lambda *x: 1 - int(any(x[k] for k in c))), COST[f"nor{tag}"],
                "~(" + " | ".join(names) + ")")
    for c in combinations(range(4), 3):
        names = [NAMES[k] for k in c]
        add(pool, tt_of(lambda *x: int(x[c[0]] + x[c[1]] + x[c[2]] >= 2)), COST["maj3"],
            f"(({names[0]} & {names[1]}) | ({names[2]} & ({names[0]} | {names[1]})))")
    # AO22/OA22 (3 种配对划分)
    for (i, j), (k, l) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        vi, vj, vk, vl = NAMES[i], NAMES[j], NAMES[k], NAMES[l]
        add(pool, tt_of(lambda *x: (x[i] & x[j]) | (x[k] & x[l])), COST["ao22"],
            f"(({vi} & {vj}) | ({vk} & {vl}))")
        add(pool, tt_of(lambda *x: (x[i] | x[j]) & (x[k] | x[l])), COST["oa22"],
            f"(({vi} | {vj}) & ({vk} | {vl}))")
    # 深度2：op(g1, g2)，g 取 literal/2输入门（成本≤1.5 的项）
    base = [(tt, c, e) for tt, (c, e) in pool.items() if c <= 1.5]
    for (t1, c1, e1), (t2, c2, e2) in combinations(base, 2):
        if t1 == t2:
            continue
        add(pool, t1 & t2, c1 + c2 + COST["op2"], f"({e1} & {e2})")
        add(pool, t1 | t2, c1 + c2 + COST["op2"], f"({e1} | {e2})")
        add(pool, t1 ^ t2, c1 + c2 + COST["opx"], f"({e1} ^ {e2})")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--max_est", type=float, default=4.6, help="est 面积上限(µm²)")
    args = ap.parse_args()

    pool = build_pool()
    items = sorted(((c, tt, e) for tt, (c, e) in pool.items()))
    print(f"pool: {len(items)} distinct functions")

    tts = np.array([tt for _, tt, _ in items], dtype=np.int64)
    costs = np.array([c for c, _, _ in items])
    tab = ((tts[:, None] >> np.arange(16)) & 1).astype(np.int8)   # (N,16)

    # cout 候选：0 或 便宜项(≤1.0)
    cout_idx = [i for i in range(len(items)) if costs[i] <= 1.0]
    zero_i = int(np.where(tts == 0)[0][0])

    results = []
    N = len(items)
    for co in ([zero_i] + [i for i in cout_idx if i != zero_i]):
        co_t = tab[co].astype(np.int16)
        co_c = costs[co]
        # sum × carry 全组合(分块)
        for s0 in range(0, N, 256):
            S = tab[s0:s0 + 256].astype(np.int16)                  # (bs,16)
            budget = args.max_est - co_c - costs[s0:s0 + 256]
            ok_c = costs[None, :] <= budget[:, None]               # (bs,N)
            if not ok_c.any():
                continue
            V = S[:, None, :] + 2 * tab[None, :, :].astype(np.int16) + 2 * co_t
            E = V - VTRUE[None, None, :]
            wae = (np.abs(E) * PPAT).sum(-1)
            bias = (E * PPAT).sum(-1)
            wae[~ok_c] = 9e9
            flat = np.argsort(wae, axis=None)[: 4000]
            for f in flat:
                si, ci = s0 + f // N, f % N
                if wae[f // N, f % N] > 0.36:
                    break
                results.append((float(wae[f // N, f % N]), float(bias[f // N, f % N]),
                                float(costs[si] + costs[ci] + co_c), si, ci, co))
    # (wae, est) 帕累托 + bias 分组代表
    results.sort(key=lambda r: (round(r[0], 6), r[2]))
    seen, out = set(), []
    best_cost = 9e9
    for r in results:
        key = (tts[r[3]], tts[r[4]], tts[r[5]])
        if key in seen:
            continue
        seen.add(key)
        out.append(r + ("front" if r[2] < best_cost - 1e-9 else "",))
        if r[2] < best_cost:
            best_cost = r[2]
        if len(out) >= args.top * 20:
            break
    print(f"{'wae':>8} {'bias':>8} {'est':>5}  sum | carry | cout")
    shown = 0
    for wae, bias, est, si, ci, co, tag in out:
        if shown >= args.top:
            break
        # 只展示帕累托点与每个 wae 档的 bias 多样性代表
        if not tag and shown > 5:
            continue
        shown += 1
        print(f"{wae:8.4f} {bias:+8.4f} {est:5.2f}  {items[si][2]} | {items[ci][2]} | "
              f"{items[co][2]}  {tag}")
    # 附:全帕累托(wae 升序, est 严格降)
    print("\n=== (wae,est) pareto ===")
    front, best = [], 9e9
    for r in sorted(results, key=lambda x: (x[0], x[2])):
        if r[2] < best - 1e-9:
            front.append(r)
            best = r[2]
    for wae, bias, est, si, ci, co in front:
        print(f"{wae:8.4f} {bias:+8.4f} {est:5.2f}  {items[si][2]} | {items[ci][2]} | "
              f"{items[co][2]}")


if __name__ == "__main__":
    main()
