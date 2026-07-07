#!/usr/bin/env python3
"""原生 4:2 库：合并 DC 表征 → library42_native.json → P/N/Z 分组 Pareto 选型（含 tmax 闸门）。

选型规则（沿用 3:2 库经验 + CT42 教训）：
  1) 只保留真省面积的 cell：area < area(CT42_BAL 锚点)
  2) 时序闸门：tmax <= tmax(CT42_BAL) * (1+tol)，避免把 v1 的时序压力买回来
  3) 按 bias 分 P/N/Z 组，各组在 (wae, area) 与 (wae, dyn) 求 Pareto 并集
  4) 各组按 log-wae 均匀取 k 个（Z 组按 er 排）
产物：library42_native.json、selected_compressors42_native.json、pareto_comp42n.png
"""
import argparse
import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PUNIT = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12}


def parse_reports(rep_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(rep_dir, "ppa_*.rpt"))):
        for line in open(path):
            if not line.startswith("PPA mod="):
                continue
            if "ERROR=" in line:
                m = re.match(r"PPA mod=(\S+) ERROR=(.*)", line.strip())
                out[m.group(1)] = {"error": m.group(2)}
                continue
            kv = dict(re.findall(r"(\w+)=(\S+)", line.strip()))
            def f(k):
                v = kv.get(k)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            dyn = f("dyn")
            if dyn is not None and kv.get("dyn_u") in PUNIT:
                dyn *= PUNIT[kv["dyn_u"]] * 1e3  # -> mW
            leak = f("leak")
            if leak is not None and kv.get("leak_u") in PUNIT:
                leak *= PUNIT[kv["leak_u"]] * 1e3
            out[kv["mod"]] = {
                "area": f("area"), "dyn_mw": dyn, "leak_mw": leak,
                "tmax": f("tmax"), "Tsum": f("Tsum"),
                "Tcarry": f("Tcarry"), "Tcout": f("Tcout"),
            }
    return out


def pareto(points, xkey, ykey):
    pts = sorted((p for p in points if p.get(xkey) is not None and p.get(ykey) is not None),
                 key=lambda p: (p[xkey], p[ykey]))
    front, best = [], math.inf
    for p in pts:
        if p[ykey] < best:
            front.append(p)
            best = p[ykey]
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-dir", default=os.path.join(HERE, "reports42n"))
    ap.add_argument("--k-per-group", type=int, default=8)
    ap.add_argument("--k-z", type=int, default=4)
    ap.add_argument("--tmax-tol", type=float, default=0.05)
    ap.add_argument("--maxe-cap", type=int, default=2)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(HERE, "rtl", "manifest42n.json")))["cells"]
    ppa = parse_reports(args.rep_dir)

    cells = {}
    n_err = 0
    for name, m in manifest.items():
        r = ppa.get(name)
        if r is None or "error" in r or r.get("area") is None:
            n_err += 1
            continue
        cells[name] = {**m, **r}
    anchors = {k: ppa[k] for k in ("FA", "HA", "CT42_TPL", "CT42_FLAT", "CT42_BAL") if k in ppa}
    lib = {"meta": {"p_one": 0.25, "mode": "compile_medium", "anchors": anchors,
                    "n_cells": len(cells), "n_failed": n_err},
           "cells": cells}
    json.dump(lib, open(os.path.join(HERE, "library42_native.json"), "w"), indent=1)
    print(f"library42_native.json: {len(cells)} cells ({n_err} failed) anchors={list(anchors)}")
    if "CT42_BAL" not in anchors:
        raise SystemExit("missing CT42_BAL anchor")
    a0 = anchors["CT42_BAL"]["area"]
    t0 = anchors["CT42_BAL"]["tmax"] * (1 + args.tmax_tol)
    print(f"gates: area < {a0}, tmax <= {t0:.3f}, maxe <= {args.maxe_cap}")

    pool = [{"name": n, **c} for n, c in cells.items()
            if c["area"] < a0 and (c["tmax"] is None or c["tmax"] <= t0)
            and c["maxe"] <= args.maxe_cap]
    print(f"pool after gates: {len(pool)}")

    selected = {}
    for grp, k in (("P", args.k_per_group), ("N", args.k_per_group), ("Z", args.k_z)):
        g = [p for p in pool if p["group"] == grp]
        if not g:
            print(f"group {grp}: empty")
            continue
        xkey = "wae" if grp != "Z" else "er"
        front = {p["name"]: p for p in pareto(g, xkey, "area")}
        front.update({p["name"]: p for p in pareto(g, xkey, "dyn_mw")})
        cand = sorted(front.values(), key=lambda p: p[xkey])
        if len(cand) > k:  # log-x 均匀覆盖
            xs = [math.log(max(p[xkey], 1e-9)) for p in cand]
            lo, hi = xs[0], xs[-1]
            picked, used = [], set()
            for i in range(k):
                tgt = lo + (hi - lo) * i / max(k - 1, 1)
                j = min((abs(x - tgt), jj) for jj, x in enumerate(xs) if jj not in used)[1]
                used.add(j)
                picked.append(cand[j])
            cand = sorted(picked, key=lambda p: p[xkey])
        for i, p in enumerate(cand):
            alias = f"comp42n_apx_{'pos' if grp=='P' else 'neg' if grp=='N' else 'zb'}_{i+1}"
            selected[p["name"]] = {**p, "alias": alias}
        print(f"group {grp}: front={len(front)} -> selected {len(cand)}: "
              + ", ".join(f"{p['name']}(wae={p['wae']:.4g},A={p['area']:.2f})" for p in cand))

    json.dump({"meta": {"source": "library42_native.json", "anchor_area": a0,
                        "anchor_tmax": anchors["CT42_BAL"]["tmax"],
                        "gates": {"tmax_tol": args.tmax_tol, "maxe_cap": args.maxe_cap}},
               "cells": selected},
              open(os.path.join(HERE, "selected_compressors42_native.json"), "w"), indent=1)
    print(f"selected_compressors42_native.json: {len(selected)} cells")

    # 4 面板图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"P": "#d62728", "N": "#1f77b4", "Z": "#2ca02c"}
    pool_named = {p["name"] for p in pool}
    for ax, ykey, ylab in ((axes[0][0], "area", "area (um^2)"),
                           (axes[0][1], "dyn_mw", "dyn power (mW)"),
                           (axes[1][0], "tmax", "tmax (ns)")):
        for grp in ("P", "N", "Z"):
            g = [ {"name": n, **c} for n, c in cells.items() if c["group"] == grp]
            xs = [max(p["wae"], 1e-5) for p in g]
            ys = [p[ykey] for p in g]
            ax.scatter(xs, ys, s=8, alpha=0.25, c=colors[grp], label=f"{grp} (all)")
        sel = list(selected.values())
        ax.scatter([max(p["wae"], 1e-5) for p in sel], [p[ykey] for p in sel],
                   s=90, marker="*", c=[colors[p["group"]] for p in sel],
                   edgecolors="k", lw=0.5, zorder=5, label="selected")
        if ykey == "area":
            ax.axhline(a0, color="k", ls="--", lw=1, label="CT42_BAL anchor")
        if ykey == "tmax":
            ax.axhline(anchors["CT42_BAL"]["tmax"], color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("WAE (p=1/4) [log]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    ax = axes[1][1]
    for grp in ("P", "N", "Z"):
        g = [c for c in cells.values() if c["group"] == grp]
        ax.scatter([c["bias"] for c in g], [c["area"] for c in g], s=8, alpha=0.25, c=colors[grp])
    sel = list(selected.values())
    ax.scatter([p["bias"] for p in sel], [p["area"] for p in sel], s=90, marker="*",
               c=[colors[p["group"]] for p in sel], edgecolors="k", lw=0.5, zorder=5)
    ax.axhline(a0, color="k", ls="--", lw=1)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("bias (signed, p=1/4)")
    ax.set_ylabel("area (um^2)")
    ax.grid(alpha=0.25)
    fig.suptitle("native 4:2 approx compressor library — DC char (compile medium) & selection")
    fig.tight_layout()
    out = os.path.join(HERE, "pareto_comp42n.png")
    fig.savefig(out, dpi=150)
    print("plot ->", out)


if __name__ == "__main__":
    main()
