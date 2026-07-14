#!/usr/bin/env python3
"""sub-std 重选型:只保留 standalone 面积 < 原生标准单元(FA1D0/HA1D0/2xFA1D0)的
近似 cell——即使按虚高的 standalone 口径记账,进设计后也必然比原生 exact 小,
真实面积节省有硬保证(2026-07-11 H1 审计的选型修正)。

产出:
  outputs/2026-07-11_cell_pareto/substd_cell_pareto.png   帕累托图(含原生 std 线
      与旧菜单对照)
  outputs/2026-07-11_cell_pareto/substd_selection.json     新选型提案(逐族 Pareto
      前沿 + 全部合格 cell,含 wae/bias/group,P/N 平衡信息)
用法: python -m Appr_Comp.cellsolver.select_substd_cells
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

NATIVE = {"22": ("HA1D0", 2.184), "32": ("FA1D0", 2.856), "42": ("2xFA1D0", 5.712)}
# 旧训练菜单(对照用)
OLD_MENU = {
    "22": ["comp22_a4", "comp22_e4", "comp22_54", "comp22_50", "comp22_zero"],
    "32": ["comp32_b994", "comp32_aa54", "comp32_fe54", "comp32_a994",
           "comp32_5550", "comp32_5500", "comp32_f555", "comp32_5555",
           "comp32_ff55", "comp32_zero"],
}


def pareto(pts):
    out = []
    for p in sorted(pts, key=lambda x: (x[0], x[1])):
        if not out or p[1] < out[-1][1] - 1e-12:
            out.append(p)
    return out


# 文献参照(仅绘图,不进选型 JSON): Sayadi TCSI'23 无 cin/cout 近似 4:2,
# 同 dc_char 口径,见 /home/lee/Baselines/Sayadi_TCSI23_NewConfig42/
SAYADI_JSON = "/home/lee/Baselines/Sayadi_TCSI23_NewConfig42/sayadi_cells.json"


def overlay_sayadi(ax):
    if not os.path.exists(SAYADI_JSON):
        return
    cells = json.load(open(SAYADI_JSON))["cells"]
    pts, seen = [], set()
    for n, c in sorted(cells.items()):
        if c.get("error"):
            continue
        nm = "ac6g(16var)" if n.startswith("sayadi_ac6g") else n.replace("sayadi_", "")
        if nm in seen:
            continue
        seen.add(nm)
        pts.append((nm, float(c["wae"]), float(c.get("area") or 0.0)))
    if not pts:
        return
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], marker="D", s=60,
               c="#9b59b6", edgecolors="k", linewidths=0.5, zorder=5,
               label="Sayadi TCSI'23 (ref, no cin/cout)")
    for nm, x, y in pts:
        ax.annotate(nm, (x, y), fontsize=7, xytext=(4, -10),
                    textcoords="offset points", color="#6a3d9a")
    # comp42s 未过硬标准的家族成员(空心橙菱形, 标注 ×0.262 校准后真实成本)
    ch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "substd42_char.json")
    if os.path.exists(ch):
        cells = json.load(open(ch))["cells"]
        fail = [(n.replace("comp42s_", ""), float(c["wae"]), float(c["area"]))
                for n, c in sorted(cells.items())
                if c.get("area") is not None and c["area"] >= 5.712]
        if fail:
            ax.scatter([p[1] for p in fail], [p[2] for p in fail], marker="D", s=60,
                       facecolors="none", edgecolors="#e67e22", linewidths=1.4,
                       zorder=5, label="comp42s standalone>bar (probe-passed, see note)")
            for nm, x, y in fail:
                ax.annotate(f"{nm} (~{y*0.262:.1f} real)", (x, y), fontsize=6.5,
                            xytext=(4, 3), textcoords="offset points", color="#a35510")
            ax.text(0.02, 0.60,
                    "in-context probe 07-12 (94-slot swap, full-netlist DC):\n"
                    "orha real ~1.68 um2/slot, or4ao ~1.23  << 5.712  PASS\n"
                    "power -4.4 uW/slot; see outputs/2026-07-12_orha_probe/",
                    transform=ax.transAxes, fontsize=7.5, color="#14611f",
                    bbox=dict(fc="#eaf7ec", ec="#2ca02c", lw=0.8))


def main():
    lib = json.load(open("Appr_Comp/library.json"))["cells"]
    lib42 = json.load(open("Appr_Comp/library42_native.json"))
    c42 = lib42.get("cells") or {}

    fams = {}
    for t in ("22", "32"):
        rows = []
        seen_canon = {}
        for name, c in lib.items():
            if c["type"] != t or c.get("is_exact"):
                continue
            a = c.get("area")
            if a is None or float(a) >= NATIVE[t][1]:
                continue
            row = dict(
                name=name, area=float(a),
                wae=float(c.get("weighted_absolute_error", 0)),
                bias=float(c.get("weighted_signed_error", 0)),
                er=float(c.get("error_rate", 0)),
                maxe=float(c.get("max_error", 0)),
                group=c.get("group"),
                dyn_mw=(float(c["dyn_w"]) * 1e3) if c.get("dyn_w") else None,
                canon=c.get("canon_key"),
            )
            # canon 同函数去重:保留面积最小(并列取 wae 小)
            key = row["canon"] or name
            old = seen_canon.get(key)
            if old is None or (row["area"], row["wae"]) < (old["area"], old["wae"]):
                seen_canon[key] = row
        rows = sorted(seen_canon.values(), key=lambda r: (r["wae"], r["area"]))
        fams[t] = rows
    # T42
    rows42 = [dict(name=n, area=float(c.get("area") or 9e9),
                   wae=float(c.get("wae", 0)), group=c.get("group"),
                   bias=float(c.get("bias", 0)), er=float(c.get("er", 0)),
                   maxe=float(c.get("maxe", 0)),
                   dyn_mw=c.get("dyn_mw"))
              for n, c in c42.items()
              if not c.get("is_exact") and float(c.get("area") or 9e9) < NATIVE["42"][1]]
    fams["42"] = rows42

    out_dir = "outputs/2026-07-11_cell_pareto"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    sel = {}
    for col, t in enumerate(("22", "32", "42")):
        ax = axes[col]
        rows = fams[t]
        nname, nval = NATIVE[t]
        ax.axhline(nval, color="#2ca02c", lw=1.4, ls="--",
                   label=f"native std {nname} = {nval}")
        if not rows:
            ax.text(0.5, 0.5, "0 / 1999 cells qualify\n(all comp42n areas 17-23,\n"
                    "far above 2xFA1D0=5.71)",
                    transform=ax.transAxes, ha="center", fontsize=11, color="#a33")
            ax.set_title(f"T{t}: no sub-std cells")
            ax.set_xlabel("wae per use (LSB)")
            ax.set_ylabel("standalone area (um^2)")
            if t == "42":
                overlay_sayadi(ax)
            ax.legend(fontsize=8)
            sel[t] = {"pareto": [], "all_qualified": []}
            continue
        pts = [(r["wae"], r["area"], r["name"]) for r in rows]
        front = pareto([(x, y) for x, y, _ in pts])
        # 建议菜单:T32 全收;T22 收 wae≤0.625(高 wae 尾巴无用);保 P/N/Z 平衡
        prop_th = {"22": 0.625, "32": 9e9}.get(t, 0)
        proposed = {r["name"] for r in rows if r["wae"] <= prop_th}
        gcol = {"P": "#d62728", "N": "#1f77b4", "Z": "#2ca02c"}
        by_r = {r["name"]: r for r in rows}
        for x, y, name in pts:
            g = str(by_r[name].get("group"))
            ax.scatter(x, y, s=46, c=gcol.get(g, "#9aa0a6"), zorder=3)
            if name in proposed:
                ax.scatter(x, y, s=170, facecolors="none",
                           edgecolors="#333", lw=1.2, zorder=4)
            ax.annotate(name.split("_", 1)[-1], (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", color="#333")
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.step(fx, fy, where="post", color="#888", lw=1.2, alpha=0.6,
                label="Pareto front (wae vs area)")
        for g, c in gcol.items():
            ax.scatter([], [], c=c, label=f"bias group {g}")
        ax.scatter([], [], s=120, facecolors="none", edgecolors="#333",
                   label="proposed menu")
        # 旧菜单对照(叉号,含不合格的——它们落在绿线上方)
        for name in OLD_MENU.get(t, []):
            c = lib.get(name)
            if not c or c.get("area") is None:
                continue
            ax.scatter(float(c.get("weighted_absolute_error", 0)),
                       float(c["area"]), marker="x", s=42, c="#1f77b4",
                       zorder=2, alpha=0.7)
        ax.scatter([], [], marker="x", c="#1f77b4", label="old menu (for contrast)")
        if t == "42":
            overlay_sayadi(ax)
        ax.set_xlabel("wae per use (LSB)")
        ax.set_ylabel("standalone area (um^2)")
        ax.set_title(f"T{t}: {len(rows)} qualified (dedup by canon), "
                     f"{len(front)} on front")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        # 选型提案:建议菜单(P/N/Z 平衡) + 全部合格
        sel[t] = {
            "proposed_menu": [r for r in rows if r["name"] in proposed],
            "all_qualified": rows,
        }
    fig.suptitle(
        "Re-selection under hard constraint: standalone area < native std cell\n"
        "(guarantees real in-design saving regardless of characterization "
        "inflation; error requirement relaxed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    png = os.path.join(out_dir, "substd_cell_pareto.png")
    fig.savefig(png, dpi=150)
    print("plot ->", png)

    json.dump(sel, open(os.path.join(out_dir, "substd_selection.json"), "w"),
              indent=1, ensure_ascii=False)
    print("selection ->", os.path.join(out_dir, "substd_selection.json"))
    for t in ("22", "32"):
        print(f"\n=== T{t} 建议菜单(P/N/Z 平衡) ===")
        print(f"{'name':<16}{'wae':>8}{'bias':>8}{'group':>6}{'area':>7}")
        for r in sorted(sel[t]["proposed_menu"], key=lambda r: r["wae"]):
            print(f"{r['name']:<16}{r['wae']:>8}{r['bias']:>8}{str(r['group']):>6}"
                  f"{r['area']:>7}")


if __name__ == "__main__":
    main()
