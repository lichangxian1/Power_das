#!/usr/bin/env python
"""从 library42_native.json 提取真值表解析特征 + DC 标签。

特征（全部零成本，不需要任何综合）：
- 原始 LUT 位（16 bit x 3 输出）
- 每输出在 p_one=0.25 偏置下的信号概率 p1、随机向量 toggle=2*p1*(1-p1)
- 每输入对每输出的布尔影响度 I_i(f)=P(f(X)!=f(X^e_i))（p 偏置测度）
- 总影响度（平均敏感度）、minterm 数、与 exact 的汉明距离
"""
import json, itertools, sys
import numpy as np

P_ONE = 0.25

def minterm_probs(n, p):
    probs = np.zeros(2 ** n)
    for m in range(2 ** n):
        pr = 1.0
        for i in range(n):
            pr *= p if (m >> i) & 1 else (1 - p)
        probs[m] = pr
    return probs

def influences(lut, n, probs):
    """I_i = sum_m probs[m] * [f(m) != f(m^ (1<<i))]"""
    lut = np.asarray(lut)
    out = []
    for i in range(n):
        flipped = np.array([lut[m ^ (1 << i)] for m in range(2 ** n)])
        out.append(float(np.sum(probs * (lut != flipped))))
    return out

def main():
    lib = json.load(open("/home/lee/Power_das/Appr_Comp/library42_native.json"))
    cells = lib["cells"]
    n = 4
    probs = minterm_probs(n, P_ONE)
    # exact 4:2 (含 cin=0): v = a+b+c+d, sum=v&1, carry/cout 按 CT42_BAL 语义。
    # 直接用库里 is_exact 的 cell 或按 v_lut 生成 exact 参考：
    exact_v = np.array([bin(m).count("1") for m in range(16)])

    rows = []
    for name, c in (cells.items() if isinstance(cells, dict) else ((c.get("name", str(i)), c) for i, c in enumerate(cells))):
        sum_l = np.array(c["sum_lut"]); car_l = np.array(c["carry_lut"]); cout_l = np.array(c["cout_lut"])
        v_l = np.array(c["v_lut"])
        feat = {}
        feat["name"] = name
        for lab, lut in (("sum", sum_l), ("carry", car_l), ("cout", cout_l)):
            p1 = float(np.sum(probs * lut))
            feat[f"{lab}_p1"] = p1
            feat[f"{lab}_toggle"] = 2 * p1 * (1 - p1)
            infl = influences(lut, n, probs)
            for i, v in enumerate(infl):
                feat[f"{lab}_infl_{i}"] = v
            feat[f"{lab}_infl_tot"] = sum(infl)
            feat[f"{lab}_ones"] = int(lut.sum())
            for m in range(16):
                feat[f"{lab}_bit{m}"] = int(lut[m])
        feat["v_err_l1"] = float(np.sum(probs * np.abs(v_l - exact_v)))
        feat["hamming_v"] = int(np.sum(v_l != exact_v))
        feat["family"] = c.get("family", "?")
        feat["is_exact"] = bool(c.get("is_exact", False))
        # labels
        feat["area"] = c["area"]; feat["dyn_mw"] = c["dyn_mw"]
        feat["leak_mw"] = c["leak_mw"]; feat["tmax"] = c["tmax"]
        rows.append(feat)

    import csv
    keys = list(rows[0].keys())
    out = "/tmp/claude-1000/-home-lee-Power-das/c973f85e-6eb5-4da8-b295-c00dfc39f184/scratchpad/pwrpred/tt42.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows, {len(keys)} cols -> {out}")
    fams = {}
    for r in rows: fams[r["family"]] = fams.get(r["family"], 0) + 1
    print("families:", fams)

if __name__ == "__main__":
    main()
