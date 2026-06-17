#!/usr/bin/env python3
"""用已有的 dc_timing_sweep.json 重画对比图。一次性生成两种朝向、共四张图：
  - 横坐标=延迟:  area_vs_delay.png / power_vs_delay.png
  - 纵坐标=延迟:  delay_vs_area.png / delay_vs_power.png

包含 json 里出现的全部设计（eda/proxy/dw/arith/arith2…）。不重跑 DC，仅重绘。

用法:
    python scripts/replot_dc_sweep_delay_x.py \
        --json outputs/.../dc_timing_sweep_..._plus_arith/dc_timing_sweep.json
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "eda": ("EDA-guided", "#d62728", "s"),
    "proxy": ("Proxy-guided", "#1f77b4", "o"),
    "dw": ("DesignWare (a*b)", "#2ca02c", "^"),
    "arith": ("Arith-DAS (05-06)", "#9467bd", "D"),
    "arith2": ("Arith-DAS (05-11)", "#8c564b", "P"),
}
ORDER = ["eda", "proxy", "dw", "arith2"]  # arith(05-06, 紫色) 已按要求剔除


def _good(rows):
    return [
        r
        for r in rows
        if r.get("area") is not None
        and r.get("delay") is not None
        and r.get("power_mw") is not None
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="dc_timing_sweep.json")
    ap.add_argument("--out_dir", default=None, help="默认与 json 同目录")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    results = data["results"]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))

    order = [d for d in ORDER if d in results]

    specs = [
        # x = delay
        ("delay", "area", "DC Delay (ns)", "DC Area (μm²)", "area_vs_delay.png", "Area vs Delay"),
        ("delay", "power_mw", "DC Delay (ns)", "DC Power (mW)", "power_vs_delay.png", "Power vs Delay"),
        # y = delay
        ("area", "delay", "DC Area (μm²)", "DC Delay (ns)", "delay_vs_area.png", "Delay vs Area"),
        ("power_mw", "delay", "DC Power (mW)", "DC Delay (ns)", "delay_vs_power.png", "Delay vs Power"),
    ]
    written = []
    for xkey, ykey, xlabel, ylabel, fname, title in specs:
        fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
        for design in order:
            pts = sorted(_good(results[design]), key=lambda r: r[xkey])
            if not pts:
                continue
            xs = [p[xkey] for p in pts]
            ys = [p[ykey] for p in pts]
            label, color, marker = STYLE.get(design, (design, "#7f7f7f", "x"))
            ax.plot(
                xs, ys, marker=marker, color=color,
                linewidth=2.0, markersize=6.5, label=label,
            )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}  (DC timing-constraint sweep, 16-bit)", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
        print(f"Wrote plot: {path}", flush=True)
    return written


if __name__ == "__main__":
    main()
