#!/usr/bin/env python
"""从 library.json（3:2/2:2）提取真值表特征 + DC 标签。输入数 n 由 LUT 长度推断。"""
import json, csv
import numpy as np

P_ONE = 0.25
BASE = "/tmp/claude-1000/-home-lee-Power-das/c973f85e-6eb5-4da8-b295-c00dfc39f184/scratchpad/pwrpred"

def minterm_probs(n, p):
    probs = np.zeros(2 ** n)
    for m in range(2 ** n):
        pr = 1.0
        for i in range(n):
            pr *= p if (m >> i) & 1 else (1 - p)
        probs[m] = pr
    return probs

def influences(lut, n, probs):
    lut = np.asarray(lut)
    return [float(np.sum(probs * (lut != np.array([lut[m ^ (1 << i)] for m in range(2 ** n)]))))
            for i in range(n)]

def main():
    lib = json.load(open("/home/lee/Power_das/Appr_Comp/library.json"))
    rows = []
    for name, c in lib["cells"].items():
        sum_l = np.array(c["sum_lut"]); car_l = np.array(c["carry_lut"])
        n = int(np.log2(len(sum_l)))
        probs = minterm_probs(n, P_ONE)
        exact_v = np.array([bin(m).count("1") for m in range(2 ** n)])
        v_l = np.array(c["v_approx"])
        feat = {"name": name, "ctype": c["type"], "n_in": n}
        for lab, lut in (("sum", sum_l), ("carry", car_l)):
            p1 = float(np.sum(probs * lut))
            feat[f"{lab}_p1"] = p1
            feat[f"{lab}_toggle"] = 2 * p1 * (1 - p1)
            infl = influences(lut, n, probs)
            for i in range(3):
                feat[f"{lab}_infl_{i}"] = infl[i] if i < n else 0.0
            feat[f"{lab}_infl_tot"] = sum(infl)
            feat[f"{lab}_ones"] = int(lut.sum())
            for m in range(8):
                feat[f"{lab}_bit{m}"] = int(lut[m]) if m < len(lut) else 0
        feat["v_err_l1"] = float(np.sum(probs * np.abs(v_l - exact_v)))
        feat["hamming_v"] = int(np.sum(v_l != exact_v))
        feat["group"] = c.get("group", "?")
        feat["is_exact"] = bool(c.get("is_exact", False))
        feat["area"] = c["area"]; feat["dyn_w"] = c["dyn_w"]
        feat["tmax"] = c["tmax"]
        rows.append(feat)
    out = f"{BASE}/tt32.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    import collections
    print(f"wrote {len(rows)} rows -> {out}", collections.Counter(r["ctype"] for r in rows))

if __name__ == "__main__":
    main()
