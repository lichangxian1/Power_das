#!/usr/bin/env python3
"""阶段 0：枚举近似 3:2 / 2:2 压缩器，计算加权误差指标，按输入置换去重，导出 JSON。

纯 Python，无任何 EDA 依赖。详见 Appr_Comp/PLAN.md 第 2 节「阶段 0」。

核心约定
--------
- 一个 n 输入压缩器由其真值表完全定义：对每个输入 pattern 给出一个输出值
  V_approx = sum + 2 * carry ∈ {0,1,2,3}（sum, carry ∈ {0,1}）。
- 精确值 V_exact = popcount(inputs)。单元误差 e(x) = V_approx(x) - V_exact(x)。
- 默认只保留 |e(x)| <= MAX_E 的候选（近似压缩器标准约束，默认 MAX_E=1）。
- 加权误差按逐位概率：P(bit=1) 可逐位配置，默认 1/4（AND 第一级 PP 分布）。

用法
----
    python Appr_Comp/enumerate_compressors.py            # 默认 3/4-1/4, |e|<=1
    python Appr_Comp/enumerate_compressors.py --p-one 0.5
    python Appr_Comp/enumerate_compressors.py --max-e 2
"""
import argparse
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def value_to_sum_carry(v: int):
    """V = sum + 2*carry  ->  (sum, carry)。"""
    return v & 1, (v >> 1) & 1


def pattern_prob(pattern, p_one):
    """P(pattern) = Π_i  [p_one[i] if bit==1 else 1-p_one[i]]。"""
    p = 1.0
    for bit, p1 in zip(pattern, p_one):
        p *= p1 if bit == 1 else (1.0 - p1)
    return p


def canonical_key(combo, patterns, index_of):
    """输入置换等价类的规范键：对所有输入位置换取字典序最小的真值表。

    综合后的 area/power/delay 对输入置换不变，故每个等价类只需综合一个代表。
    """
    n = len(patterns[0])
    best = None
    for perm in itertools.permutations(range(n)):
        permuted = tuple(
            combo[index_of[tuple(p[perm[k]] for k in range(n))]] for p in patterns
        )
        if best is None or permuted < best:
            best = permuted
    return best


def is_symmetric(combo, v_exact):
    """输出是否只依赖 popcount（即对输入置换完全对称）。"""
    by_pc = {}
    for v_app, ve in zip(combo, v_exact):
        if ve in by_pc and by_pc[ve] != v_app:
            return False
        by_pc[ve] = v_app
    return True


def enumerate_compressors(n_inputs, max_e, p_one, zero_eps=1e-12):
    patterns = list(itertools.product([0, 1], repeat=n_inputs))
    index_of = {p: i for i, p in enumerate(patterns)}
    v_exact = [sum(p) for p in patterns]
    probs = [pattern_prob(p, p_one) for p in patterns]

    # 每个 pattern 的合法 V_approx 选择（|e|<=max_e 且 0<=V<=3）
    choices = [
        [v for v in range(4) if abs(v - ve) <= max_e] for ve in v_exact
    ]

    candidates = []
    for combo in itertools.product(*choices):
        e = [combo[i] - v_exact[i] for i in range(len(patterns))]
        wse = sum(probs[i] * e[i] for i in range(len(patterns)))
        wae = sum(probs[i] * abs(e[i]) for i in range(len(patterns)))
        er = sum(probs[i] for i in range(len(patterns)) if e[i] != 0)
        pos_prob = sum(probs[i] for i in range(len(patterns)) if e[i] > 0)
        neg_prob = sum(probs[i] for i in range(len(patterns)) if e[i] < 0)
        max_abs_e = max(abs(x) for x in e)

        if wse > zero_eps:
            group = "P"
        elif wse < -zero_eps:
            group = "N"
        else:
            group = "Z"
        is_exact = all(x == 0 for x in e)

        candidates.append(
            {
                "patterns": ["".join(map(str, p)) for p in patterns],
                "v_exact": v_exact,
                "v_approx": list(combo),
                "sum_lut": [value_to_sum_carry(v)[0] for v in combo],
                "carry_lut": [value_to_sum_carry(v)[1] for v in combo],
                "e": e,
                "weighted_signed_error": wse,
                "weighted_absolute_error": wae,
                "error_rate": er,
                "max_error": max_abs_e,
                "positive_error_prob": pos_prob,
                "negative_error_prob": neg_prob,
                "group": "exact" if is_exact else group,
                "is_symmetric": is_symmetric(combo, v_exact),
                "canon_key": "".join(
                    map(str, canonical_key(combo, patterns, index_of))
                ),
            }
        )
    return candidates


def summarize(candidates, label):
    n = len(candidates)
    canon = {c["canon_key"] for c in candidates}
    sym = sum(1 for c in candidates if c["is_symmetric"])
    by_group = {}
    for c in candidates:
        by_group[c["group"]] = by_group.get(c["group"], 0) + 1
    print(f"[{label}]")
    print(f"  候选总数 (|e|<=max_e)          : {n}")
    print(f"  输入置换等价类 (需综合的代表)   : {len(canon)}")
    print(f"  完全对称 (只依赖 popcount)      : {sym}")
    print(f"  按误差方向分组                  : {by_group}")
    return {
        "total": n,
        "canonical_classes": len(canon),
        "symmetric": sym,
        "by_group": by_group,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-one", type=float, default=0.25,
                    help="P(input bit = 1)，所有输入位统一取此值 (默认 0.25)")
    ap.add_argument("--max-e", type=int, default=1,
                    help="单元误差上界 |e(x)| <= max_e (默认 1)")
    ap.add_argument("--out-dir", default=HERE, help="JSON 输出目录")
    args = ap.parse_args()

    print(f"配置: P(1)={args.p_one}, |e|<= {args.max_e}\n")

    out = {}
    for label, n_inputs, fname in [
        ("3:2 compressor", 3, "cand_32.json"),
        ("2:2 compressor", 2, "cand_22.json"),
    ]:
        p_one = [args.p_one] * n_inputs
        cands = enumerate_compressors(n_inputs, args.max_e, p_one)
        stats = summarize(cands, label)
        meta = {
            "n_inputs": n_inputs,
            "p_one": p_one,
            "max_e": args.max_e,
            "stats": stats,
        }
        path = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            json.dump({"meta": meta, "candidates": cands}, f, indent=2)
        print(f"  -> 写入 {path}\n")
        out[label] = stats

    return out


if __name__ == "__main__":
    main()
