#!/usr/bin/env python3
"""阶段 2：近似压缩器库 Pareto 分组 + 代表选取 + 可视化。

读 Appr_Comp/library.json（阶段1 产出），对 3:2 / 2:2 分别：
  - 按 weighted_signed_error(bias) 符号分组 P(>0) / N(<0) / Z(≈0) / exact；
  - 误差用 weighted_absolute_error (wae = E[|e|]) 作精度轴；
  - 在 (wae, area)/(wae, power)/(wae, delay) 上对 P、N 各求 Pareto front；
  - 每组选若干代表（按 wae 升序，限制在 wae<=cap 的可用区），记录 bias 供 +/- 配对；
  - 画图：每种类型一张 4 面板图（误差-面积/功耗/延迟 + bias 谱）。

输出：
  outputs/<日期>_appr_comp_pareto/pareto_comp32.png, pareto_comp22.png
  Appr_Comp/selected_compressors.json  （友好名 -> module 名 + 指标）
"""
import argparse
import json
import os
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GROUP_COLOR = {"P": "tab:red", "N": "tab:blue", "Z": "tab:gray", "exact": "black"}


def beijing_date():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


def load_cells(lib_path, ctype):
    d = json.load(open(lib_path))
    out = []
    for name, v in d["cells"].items():
        if v.get("error") or v["type"] != ctype:
            continue
        if v.get("area") is None or v.get("dyn_w") is None:
            continue
        out.append({
            "name": name,
            "type": ctype,
            "group": v["group"],
            "bias": v["weighted_signed_error"],
            "wae": v["weighted_absolute_error"],
            "er": v["error_rate"],
            "maxe": v["max_error"],
            "area": v["area"],
            "power_mw": v["dyn_w"] * 1e3,            # W -> mW
            "delay_ns": v["tmax"] if v.get("tmax") is not None else 0.0,  # 常数cell无弧->0
        })
    return out


def pareto_front(cells, xkey, ykey):
    """返回 (xkey,ykey) 同时最小化的非支配点（lower-left 更优）。"""
    pts = sorted(cells, key=lambda c: (c[xkey], c[ykey]))
    front, best_y = [], float("inf")
    for c in pts:
        if c[ykey] < best_y - 1e-12:
            front.append(c)
            best_y = c[ykey]
    return front


def select_reps(cells, k, cap, exact_area):
    """从 (wae, area) 的 Pareto front 选 k 个代表。

    只保留真正省面积的 cell（area < exact），再限 wae<=cap，按 wae 升序均匀取；
    避免选到「只省功耗、面积反而 >= exact」的无意义点。
    """
    saving = [c for c in cells if c["area"] < exact_area - 1e-9] or cells
    front = pareto_front(saving, "wae", "area")
    usable = [c for c in front if c["wae"] <= cap] or front
    usable = sorted(usable, key=lambda c: c["wae"])
    if len(usable) <= k:
        return usable
    idx = [round(i * (len(usable) - 1) / (k - 1)) for i in range(k)]
    seen, reps = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            reps.append(usable[i])
    return reps


def plot_type(cells, ctype, reps_by_group, out_png):
    by_g = {g: [c for c in cells if c["group"] == g] for g in ("P", "N", "Z", "exact")}
    exact = by_g["exact"][0] if by_g["exact"] else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [("area", "area (µm²)", axes[0, 0]),
              ("power_mw", "dyn power (mW)", axes[0, 1]),
              ("delay_ns", "max delay tmax (ns)", axes[1, 0])]

    for ykey, ylabel, ax in panels:
        for g in ("Z", "P", "N"):
            pts = by_g[g]
            if pts:
                ax.scatter([c["wae"] for c in pts], [c[ykey] for c in pts],
                           s=14, alpha=0.35, color=GROUP_COLOR[g], label=f"{g} (n={len(pts)})")
        # Pareto fronts for P and N
        for g in ("P", "N"):
            if by_g[g]:
                fr = pareto_front(by_g[g], "wae", ykey)
                fr = sorted(fr, key=lambda c: c["wae"])
                ax.plot([c["wae"] for c in fr], [c[ykey] for c in fr],
                        "-", color=GROUP_COLOR[g], lw=1.3, alpha=0.8)
        if exact:
            ax.scatter([exact["wae"]], [exact[ykey]], marker="*", s=260,
                       color="black", zorder=5, label="exact")
        # selected reps
        for g, reps in reps_by_group.items():
            for c in reps:
                ax.scatter([c["wae"]], [c[ykey]], marker="o", s=90,
                           facecolors="none", edgecolors=GROUP_COLOR[g], linewidths=2, zorder=6)
                ax.annotate(c["alias"].split("_")[-1], (c["wae"], c[ykey]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("weighted |error|  E[|e|]  (LSB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"comp{ctype}: error vs {ylabel}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # 4th panel: bias spectrum (signed) vs area
    ax = axes[1, 1]
    for g in ("Z", "P", "N"):
        pts = by_g[g]
        if pts:
            ax.scatter([c["bias"] for c in pts], [c["area"] for c in pts],
                       s=14, alpha=0.35, color=GROUP_COLOR[g])
    if exact:
        ax.scatter([0], [exact["area"]], marker="*", s=260, color="black", zorder=5)
    for g, reps in reps_by_group.items():
        for c in reps:
            ax.scatter([c["bias"]], [c["area"]], marker="o", s=90, facecolors="none",
                       edgecolors=GROUP_COLOR[g], linewidths=2, zorder=6)
            ax.annotate(c["alias"].split("_")[-1], (c["bias"], c["area"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("signed bias  E[e]  (LSB)    (neg <-- | --> pos)")
    ax.set_ylabel("area (um^2)")
    ax.set_title(f"comp{ctype}: bias spectrum (for +/- pairing)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Approximate {ctype[0]}:{ctype[1]} compressor — Pareto & selection",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def process_type(lib_path, ctype, k, cap, out_dir):
    cells = load_cells(lib_path, ctype)
    by_g = {g: [c for c in cells if c["group"] == g] for g in ("P", "N", "Z", "exact")}

    exact_area = by_g["exact"][0]["area"] if by_g["exact"] else float("inf")
    reps_by_group = {}
    for g in ("P", "N"):
        reps = select_reps(by_g[g], k, cap, exact_area)
        reps = sorted(reps, key=lambda c: c["wae"])
        tag = "pos" if g == "P" else "neg"
        for i, c in enumerate(reps, 1):
            c["alias"] = f"comp{ctype}_apx_{tag}_{i}"
        reps_by_group[g] = reps
    if by_g["exact"]:
        by_g["exact"][0]["alias"] = f"comp{ctype}_exact"

    out_png = os.path.join(out_dir, f"pareto_comp{ctype}.png")
    plot_type(cells, ctype, reps_by_group, out_png)

    selected = {}
    if by_g["exact"]:
        e = by_g["exact"][0]
        selected[e["alias"]] = e
    for g in ("P", "N"):
        for c in reps_by_group[g]:
            selected[c["alias"]] = c
    return selected, out_png, {g: len(by_g[g]) for g in by_g}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=os.path.join(HERE, "library.json"))
    ap.add_argument("--k", type=int, default=3, help="每组(P/N)选几个代表")
    ap.add_argument("--cap", type=float, default=0.5, help="代表的 wae 上限")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "outputs", f"{beijing_date()}_appr_comp_pareto")
    os.makedirs(out_dir, exist_ok=True)

    all_selected = {}
    for ctype in ("32", "22"):
        selected, png, counts = process_type(args.lib, ctype, args.k, args.cap, out_dir)
        all_selected.update(selected)
        print(f"\n=== comp{ctype}  组规模 {counts}  -> {png}")
        print(f"  {'alias':22s} {'module':13s} {'bias':>7s} {'wae':>6s} {'ER':>5s} "
              f"{'area':>6s} {'mW':>6s} {'ns':>5s}")
        for alias, c in selected.items():
            if c["type"] != ctype:
                continue
            print(f"  {alias:22s} {c['name']:13s} {c['bias']:+7.3f} {c['wae']:6.3f} "
                  f"{c['er']:5.2f} {c['area']:6.2f} {c['power_mw']:6.3f} {c['delay_ns']:5.2f}")

    sel_path = os.path.join(HERE, "selected_compressors.json")
    json.dump({"meta": {"k": args.k, "cap": args.cap, "out_dir": out_dir},
               "selected": all_selected}, open(sel_path, "w"), indent=2)
    print(f"\n-> {sel_path}\n-> 图: {out_dir}/pareto_comp32.png, pareto_comp22.png")


if __name__ == "__main__":
    main()
