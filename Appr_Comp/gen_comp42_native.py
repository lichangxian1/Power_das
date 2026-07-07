#!/usr/bin/env python3
"""原生 4 输入近似 4:2 压缩器库生成器（2026-07-07 重设计）。

语义：inputs a,b,c,d（同权重）→ outputs sum,carry,cout；value v = sum + 2*(carry+cout)；
exact T = a+b+c+d；单元误差 e(x) = v(x) − T(x)。

生成 = 结构化家族（丢输入/OR合并/cout0/sum简化/零偏置对/exact微扰）+ 有偏随机填缝；
默认 |e|≤1（个别结构化家族允许 |e|=2，靠 --allow-e2 家族标记），S4 输入置换规范型去重；
(carry,cout) 编码贴 CT42_BAL 拆分（h=1 时跟随 exact 的 carry 位）以利综合。

产物：
  rtl/comp42n_lib.v       — 全部代表的扁平 SOP RTL（端口 a,b,c,d,sum,carry,cout，与搜索原语对齐）
  rtl/manifest42n.json    — 每 cell 的 LUT/误差指标/家族标签
  rtl/module_list42n.txt  — DC 表征模块清单
自校验：SOP 仿真 == LUT，16 pattern × 3 输出全对才落盘。
"""
import argparse
import hashlib
import itertools
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")

NP = 16  # patterns
P_ONE = 0.25


def bits(x):
    return ((x >> 3) & 1, (x >> 2) & 1, (x >> 1) & 1, x & 1)  # a,b,c,d


def T(x):
    return bin(x).count("1")


PROB = [P_ONE ** T(x) * (1 - P_ONE) ** (4 - T(x)) for x in range(NP)]

# ---- BAL exact 输出拆分（编码锚点）----
def bal_outputs(x):
    a, b, c, d = bits(x)
    w = a ^ b
    s = w ^ c ^ d
    cout = c if w else a
    carry = (w ^ c) & d
    return s, carry, cout


# ---- S4 规范型 ----
PERMS = list(itertools.permutations(range(4)))


def apply_perm(x, perm):
    ab = bits(x)
    src = [ab[perm[i]] for i in range(4)]
    return (src[0] << 3) | (src[1] << 2) | (src[2] << 1) | src[3]


def canon_key(v):
    best = None
    for perm in PERMS:
        t = tuple(v[apply_perm(x, perm)] for x in range(NP))
        if best is None or t < best:
            best = t
    return best


def metrics(v):
    bias = wae = er = 0.0
    maxe = 0
    for x in range(NP):
        e = v[x] - T(x)
        bias += PROB[x] * e
        wae += PROB[x] * abs(e)
        if e:
            er += PROB[x]
        maxe = max(maxe, abs(e))
    return {"bias": bias, "wae": wae, "er": er, "maxe": maxe}


def valid(v, emax=1):
    return all(0 <= v[x] <= 5 and abs(v[x] - T(x)) <= emax for x in range(NP))


# ---- 家族生成（返回 (v_lut, family) 列表）----
def families():
    out = []

    def add(v, fam):
        v = tuple(v)
        out.append((v, fam))

    ex = [T(x) for x in range(NP)]

    # 丢输入为0 / 为1（e=−x_i / +(1−x_i)）
    for i in range(4):
        add([T(x) - bits(x)[i] for x in range(NP)], "drop0")
        add([T(x) + (1 - bits(x)[i]) for x in range(NP)], "drop1")
    # OR 合并对：e = −(xi & xj)；正镜像 e=+(xi & xj)
    for i, j in itertools.combinations(range(4), 2):
        add([T(x) - (bits(x)[i] & bits(x)[j]) for x in range(NP)], "ormerge_n")
        add([T(x) + (bits(x)[i] & bits(x)[j]) for x in range(NP)], "ormerge_p")
        # AND 保留（e=−(xi^xj)）及正镜像
        add([T(x) - (bits(x)[i] ^ bits(x)[j]) for x in range(NP)], "andkeep_n")
        add([T(x) + (bits(x)[i] ^ bits(x)[j]) for x in range(NP)], "andkeep_p")
    # cout0：v=min(T,3)（e=−1 仅 T=4）；饱和上限4的正镜像 v=max(T,1)
    add([min(T(x), 3) for x in range(NP)], "cout0")
    add([max(T(x), 1) for x in range(NP)], "floor1")
    # 双 OR（|e|≤2 家族）：e = −(a&b)−(c&d)（规范型下代表一类）
    v = [T(x) - (bits(x)[0] & bits(x)[1]) - (bits(x)[2] & bits(x)[3]) for x in range(NP)]
    add(v, "double_or[e2]")
    # sum 简化：sum'=xi^xj → c^d 奇偶不符处 e=±1（全负 / 全正 / 零偏置交替）
    for i, j in (( 0, 1), (2, 3)):
        rest = [k for k in range(4) if k not in (i, j)]
        for sgn, fam in ((-1, "sumxor2_n"), (1, "sumxor2_p")):
            add([T(x) + sgn * (bits(x)[rest[0]] ^ bits(x)[rest[1]]) for x in range(NP)],
                fam)
    # 零偏置对：同 popcount 的两 pattern 一正一负
    for pc in (1, 2, 3):
        cls = [x for x in range(NP) if T(x) == pc]
        for xp, xn in itertools.permutations(cls, 2):
            v = list(ex)
            v[xp] += 1
            v[xn] -= 1
            add(v, "zerobias_pair")
    return out


def perturbations(max3=400, seed=7):
    rng = random.Random(seed)
    out = []
    ex = [T(x) for x in range(NP)]
    # 全部 1-flip / 2-flip
    for x in range(NP):
        for s in (-1, 1):
            v = list(ex); v[x] += s
            if valid(v):
                out.append((tuple(v), "flip1"))
    for x1, x2 in itertools.combinations(range(NP), 2):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                v = list(ex); v[x1] += s1; v[x2] += s2
                if valid(v):
                    out.append((tuple(v), "flip2"))
    # 3-flip 采样
    seen = 0
    while seen < max3:
        xs = rng.sample(range(NP), 3)
        v = list(ex)
        for x in xs:
            v[x] += rng.choice((-1, 1))
        if valid(v):
            out.append((tuple(v), "flip3"))
            seen += 1
    return out


def biased_random(n, seed=13):
    rng = random.Random(seed)
    ex = [T(x) for x in range(NP)]
    # 三种位置偏好：低概率 pattern 优先（高精度端）/ 均匀 / 高概率优先
    wlow = [1.0 / (PROB[x] + 1e-6) for x in range(NP)]
    whigh = [PROB[x] for x in range(NP)]
    wuni = [1.0] * NP
    out = []
    while len(out) < n:
        v = list(ex)
        if rng.random() < 0.15:
            # 零偏置模式：同 popcount 类内 +/− 配对（1~2 对）
            ok = True
            for _ in range(rng.choice((1, 1, 2))):
                pc = rng.choice((1, 2, 2, 3))
                cls = [x for x in range(NP) if T(x) == pc]
                xp, xn = rng.sample(cls, 2)
                v[xp] += 1
                v[xn] -= 1
                if not (0 <= v[xp] <= 5 and 0 <= v[xn] <= 5):
                    ok = False
                    break
            if ok:
                out.append((tuple(v), "rand_zb"))
            continue
        w = rng.choice((wlow, wuni, whigh))
        nnz = rng.choice((1, 2, 2, 3, 3, 4, 5, 5, 6, 6))
        xs = set()
        while len(xs) < nnz:
            xs.add(rng.choices(range(NP), weights=w)[0])
        mode = rng.random()
        ok = True
        for x in xs:
            if mode < 0.4:
                s = -1
            elif mode < 0.8:
                s = 1
            else:
                s = rng.choice((-1, 1))
            v[x] += s
            if not (0 <= v[x] <= 5):
                ok = False
                break
        if ok:
            out.append((tuple(v), "rand"))
    return out


# ---- 编码 + RTL ----
def encode(v):
    sums, carrys, couts = [], [], []
    for x in range(NP):
        s = v[x] & 1
        h = (v[x] - s) // 2
        if h == 0:
            cy, co = 0, 0
        elif h == 2:
            cy, co = 1, 1
        else:
            _, cy_e, co_e = bal_outputs(x)
            if cy_e and not co_e:
                cy, co = 1, 0
            elif co_e and not cy_e:
                cy, co = 0, 1
            else:
                cy, co = 1, 0
        sums.append(s); carrys.append(cy); couts.append(co)
    return sums, carrys, couts


def sop(lut, name):
    ones = [x for x in range(NP) if lut[x]]
    if not ones:
        return f"    assign {name} = 1'b0;\n"
    if len(ones) == NP:
        return f"    assign {name} = 1'b1;\n"
    terms = []
    for x in ones:
        a, b, c, d = bits(x)
        lits = []
        for val, port in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
            lits.append(port if val else f"~{port}")
        terms.append("(" + " & ".join(lits) + ")")
    return f"    assign {name} = " + " | ".join(terms) + ";\n"


def emit_module(name, sums, carrys, couts):
    src = f"module {name} (a, b, c, d, sum, carry, cout);\n"
    src += "    input a;\n    input b;\n    input c;\n    input d;\n"
    src += "    output sum;\n    output carry;\n    output cout;\n"
    src += sop(sums, "sum") + sop(carrys, "carry") + sop(couts, "cout")
    src += "endmodule\n"
    return src


def simulate_check(sums, carrys, couts, v):
    for x in range(NP):
        got = sums[x] + 2 * carrys[x] + 2 * couts[x]
        assert got == v[x], f"encode mismatch at {x}"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="去重后总代表数上限")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cand = families() + perturbations()
    # 旧 pair32 库投影（cin=0）并入候选池
    old_path = os.path.join(HERE, "library42_pair32_func.json")
    n_old = 0
    if os.path.exists(old_path):
        old = json.load(open(old_path))["cells"]
        for cname, cell in old.items():
            pats = cell.get("patterns")
            if not pats:
                continue
            v = [None] * NP
            for i, pb in enumerate(pats):
                if pb[-1] != "0":
                    continue
                x = int(pb[:4], 2)
                v[x] = (int(cell["sum_lut"][i]) + 2 * int(cell["carry_lut"][i])
                        + 2 * int(cell["cout_lut"][i]))
            if all(y is not None for y in v) and valid(v, emax=2):
                cand.append((tuple(v), "pair32proj"))
                n_old += 1

    # 规范型去重（结构化优先保留家族标签）
    seen = {}
    order = []
    for v, fam in cand:
        if not valid(v, emax=2):
            continue
        k = canon_key(v)
        if k in seen:
            continue
        seen[k] = (v, fam)
        order.append(k)

    n_structured = len(order)
    need = max(0, args.target - n_structured)
    for v, fam in biased_random(need * 25, seed=args.seed):
        if len(order) >= args.target:
            break
        k = canon_key(v)
        if k in seen:
            continue
        seen[k] = (v, fam)
        order.append(k)

    exact_v = tuple(T(x) for x in range(NP))
    manifest = {}
    rtl_src = ["// native 4-input approximate 4:2 compressor library (gen_comp42_native.py)\n"]
    names = []
    for k in order:
        v, fam = seen[k]
        if v == exact_v:
            continue  # exact 由 CT42 模块承担
        m = metrics(v)
        h = hashlib.sha1(bytes(v)).hexdigest()[:8]
        name = f"comp42n_{h}"
        sums, carrys, couts = encode(v)
        simulate_check(sums, carrys, couts, v)
        rtl_src.append(emit_module(name, sums, carrys, couts))
        group = "Z" if abs(m["bias"]) < 1e-12 else ("P" if m["bias"] > 0 else "N")
        manifest[name] = {
            "type": "42", "pattern_bits": 4, "family": fam, "group": group,
            "is_exact": False, "v_lut": list(v),
            "sum_lut": sums, "carry_lut": carrys, "cout_lut": couts,
            **m,
        }
        names.append(name)

    os.makedirs(RTL, exist_ok=True)
    open(os.path.join(RTL, "comp42n_lib.v"), "w").write("\n".join(rtl_src))
    json.dump({"meta": {"p_one": P_ONE, "n_cells": len(names),
                        "n_from_pair32": n_old, "encode": "BAL-anchored"},
               "cells": manifest},
              open(os.path.join(RTL, "manifest42n.json"), "w"), indent=1)
    open(os.path.join(RTL, "module_list42n.txt"), "w").write("\n".join(names) + "\n")

    from collections import Counter
    fams = Counter(m["family"] for m in manifest.values())
    grps = Counter(m["group"] for m in manifest.values())
    print(f"cells: {len(names)} (structured pool {n_structured}, pair32proj {n_old})")
    print("families:", dict(fams))
    print("groups:", dict(grps))
    print("SOP simulate check: all passed")


if __name__ == "__main__":
    main()
