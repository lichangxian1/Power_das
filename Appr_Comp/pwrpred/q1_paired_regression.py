#!/usr/bin/env python3
"""Q1 配对回归：功耗预测器的 standalone Δp 能否追踪 in-design 实测 ΔP？

（solver 路线图 Step 2,判据见 OUTER_CELL_SEARCH.md §3.2。零 EDA 成本,纯本地数据。）

Part A —— cell 边际(最干净):outputs/2026-07-04_20_paired_cell_compare 的 6 对
  同布线 exact vs with_cell DC 实测。预测 Δp = Σ_cells (dyn_lib(cell) − dyn_lib(exact 同型)),
  取自 library.json 的 standalone DC 表征标签(预测器复现它 MAPE ~1%,直接用标签即测假设本身)。

Part B —— 设计级(粗):design_xa.csv 121 设计,run 内两两配对,预测 Δ = Δsum_dyn_lib,
  实测 Δ = Δpower_xa_mw。分两层:同 n_pp_active(截断深度相同,差异≈cell+布线)/全部 run 内对。

判据(Claude×Codex 2026-07-10 定稿):Spearman ≥ 0.8 且符号一致率 ≥ 85%(只在
|ΔP| 超噪声地板的 pair 上算)→ 预测器可进 solver loss;0.6–0.8 → 仅 tie-breaking;
<0.6 → 弃用,PPA 项保持纯面积。

用法: python q1_paired_regression.py   (无依赖 EDA;需 numpy)
"""
import csv
import json
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB = os.path.join(ROOT, "Appr_Comp/library.json")
PAIRED = os.path.join(ROOT, "outputs/2026-07-04_20_paired_cell_compare/paired_delta.csv")
DESIGN = os.path.join(HERE, "design_xa.csv")

# 噪声地板:DC 跨口径波动 ±0.8%(FEASIBILITY 双档复检),取 1% 保守;XA 同用 1%。
NOISE_FRAC = 0.01


def spearman(x, y):
    """Spearman 秩相关(避免依赖 scipy):对秩做 Pearson。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # 并列值取平均秩
    for arr, r in ((x, rx), (y, ry)):
        for v in np.unique(arr):
            m = arr == v
            if m.sum() > 1:
                r[m] = r[m].mean()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def verdict(rho, sign_ok, n_signed):
    sc = sign_ok / n_signed if n_signed else float("nan")
    if n_signed and rho >= 0.8 and sc >= 0.85:
        v = "PASS: 预测器可进 solver loss"
    elif rho >= 0.6:
        v = "TIE-BREAK ONLY: 仅离散化后打平时用"
    else:
        v = "FAIL: 弃用,PPA 项保持纯面积"
    return sc, v


def part_a():
    cells = json.load(open(LIB))["cells"]
    exact_dyn = {}  # type -> standalone dyn (W)
    for name, c in cells.items():
        if c.get("is_exact"):
            exact_dyn[c["type"]] = float(c["dyn_w"])
    assert "32" in exact_dyn and "22" in exact_dyn, f"library 缺 exact 表征: {exact_dyn}"

    rows = list(csv.DictReader(open(PAIRED)))
    print(f"=== Part A: 6 对同布线 DC 配对 (exact vs with_cell) ===")
    print(f"exact standalone dyn: comp32={exact_dyn['32']*1e3:.4f}mW  comp22={exact_dyn['22']*1e3:.4f}mW")
    pred, meas, base, labels = [], [], [], []
    for r in rows:
        dp = 0.0
        for nm in filter(None, r["cell_names"].split(";")):
            c = cells[nm]
            dp += float(c["dyn_w"]) - exact_dyn[c["type"]]
        pred.append(dp * 1e3)  # W -> mW
        meas.append(float(r["delta_power_mw"]))
        base.append(float(r["power_exact_mw"]))
        labels.append(r["label"])
    pred, meas, base = map(np.array, (pred, meas, base))

    print(f"{'pair':<14}{'n_cells':>8}{'pred_dP(mW)':>13}{'meas_dP(mW)':>13}{'floor(mW)':>11}  同号?")
    floor = NOISE_FRAC * base
    n_ok = n_signed = 0
    for i, lb in enumerate(labels):
        above = abs(meas[i]) > floor[i]
        agree = np.sign(pred[i]) == np.sign(meas[i])
        tag = ("Y" if agree else "N") if above else "- (低于地板,不计)"
        if above:
            n_signed += 1
            n_ok += int(agree)
        n_cells = len([x for x in rows[i]["cell_names"].split(";") if x])
        print(f"{lb:<14}{n_cells:>8}{pred[i]:>13.4f}{meas[i]:>13.4f}{floor[i]:>11.3f}  {tag}")
    rho = spearman(pred, meas)
    sc, v = verdict(rho, n_ok, n_signed)
    print(f"Spearman={rho:.3f}  符号一致率={n_ok}/{n_signed}"
          f"{f'={sc:.0%}' if n_signed else ''}  → {v}")
    return rho, sc, n_signed


def part_b():
    rows = list(csv.DictReader(open(DESIGN)))
    for r in rows:
        r["sum_dyn_lib"] = float(r["sum_dyn_lib"])
        r["power_xa_mw"] = float(r["power_xa_mw"])
        r["n_pp_active"] = int(r["n_pp_active"])
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    def eval_pairs(same_pp):
        pred, meas, floors = [], [], []
        for run, ds in by_run.items():
            for a, b in combinations(ds, 2):
                if same_pp and a["n_pp_active"] != b["n_pp_active"]:
                    continue
                pred.append(a["sum_dyn_lib"] - b["sum_dyn_lib"])
                meas.append(a["power_xa_mw"] - b["power_xa_mw"])
                floors.append(NOISE_FRAC * 0.5 * (a["power_xa_mw"] + b["power_xa_mw"]))
        pred, meas, floors = map(np.array, (pred, meas, floors))
        if not len(pred):
            return
        above = np.abs(meas) > floors
        signed = above & (pred != 0)
        agree = np.sign(pred[signed]) == np.sign(meas[signed])
        rho = spearman(pred, meas)
        sc, v = verdict(rho, int(agree.sum()), int(signed.sum()))
        strat = "同 n_pp_active(≈同截断,差异=cell+布线)" if same_pp else "全部 run 内对(含结构差异)"
        print(f"[{strat}] n_pairs={len(pred)}  Spearman={rho:.3f}  "
              f"符号一致率={int(agree.sum())}/{int(signed.sum())}"
              f"{f'={sc:.0%}' if signed.sum() else ''}  → {v}")
        return rho, sc

    print(f"\n=== Part B: 121 设计 XA 集, run 内配对 (Δsum_dyn_lib vs Δpower_xa) ===")
    eval_pairs(same_pp=True)
    eval_pairs(same_pp=False)


if __name__ == "__main__":
    part_a()
    part_b()
