#!/usr/bin/env python3
"""在一份已有的 dc_timing_sweep.json 上，对其中**全部**设计补跑一组新的
target delay（细化采样点），按 target_delay merge 回 results，并重画两种朝向
的对比图（delay_vs_* 与 *_vs_delay）。

与 dc_timing_sweep_ppa.py 完全相同的远端 DC 评估 / 并发调度 / 绘图风格。
设计的网表内容直接来自 json 里记录的 paths（dw 使用内置 behavioral a*b）。

用法（细化 1.0~1.4，新增 1.1 和 1.3 两个点）:
    python scripts/refine_dc_sweep_delays.py \
        --merge_json outputs/.../dc_timing_sweep_..._plus_arith/dc_timing_sweep.json \
        --add_delays "1.1 1.3" --workers 4 --run_timeout 2400
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/

import run_power_sweep
import dc_timing_sweep_ppa as sweep

# 额外设计的图例样式（与 replot_dc_sweep_delay_x.py 保持一致）
EXTRA_STYLE = {
    "arith": ("Arith-DAS", "#9467bd", "D"),
    "arith2": ("Arith-DAS (05-11)", "#8c564b", "P"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge_json", required=True, help="已有 dc_timing_sweep.json")
    ap.add_argument(
        "--add_delays",
        required=True,
        help="要补跑的 target delays（空格/逗号分隔），对所有设计都跑",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bit_width", type=int, default=16)
    ap.add_argument("--run_timeout", type=int, default=None)
    ap.add_argument("--out_dir", default=None, help="默认原地覆盖 json 所在目录")
    ap.add_argument("--annotate", action="store_true", default=False)
    args = ap.parse_args()

    if args.run_timeout is not None:
        run_power_sweep.EDA_RUN_TIMEOUT = args.run_timeout
        print(f"[override] EDA_RUN_TIMEOUT = {args.run_timeout}s", flush=True)

    merge_json = (
        args.merge_json
        if os.path.isabs(args.merge_json)
        else os.path.join(_REPO_ROOT, args.merge_json)
    )
    with open(merge_json) as f:
        data = json.load(f)
    results = data["results"]
    delays = data["delays"]
    paths = data.get("paths", {})
    base_run_dir = data.get("run_dir")

    # 把 json 里出现的所有设计注册进 sweep 的样式表，保证绘图/保存包含全部线条
    designs = list(results.keys())
    for name in designs:
        if name not in sweep.DESIGNS:
            sweep.DESIGNS.append(name)
        if name not in sweep.LABELS:
            label, color, marker = EXTRA_STYLE.get(name, (name, "#7f7f7f", "x"))
            sweep.LABELS[name] = label
            sweep.COLORS[name] = color
            sweep.MARKERS[name] = marker

    # 读取每个设计的网表内容
    contents = {}
    for name in designs:
        p = paths.get(name)
        if name == "dw" or (p and not os.path.exists(p)):
            contents[name] = sweep.DW_RTL
        else:
            with open(p) as f:
                contents[name] = f.read()

    add = [float(x) for x in args.add_delays.replace(",", " ").split() if x.strip()]

    out_dir = args.out_dir or os.path.dirname(merge_json)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Refine DC timing sweep (add delays for ALL designs)")
    print(f"  merge_json : {merge_json}")
    print(f"  designs    : {designs}")
    print(f"  add_delays : {add}")
    print(f"  workers    : {args.workers}")
    print(f"  total      : {len(designs) * len(add)} DC runs")
    print(f"  out_dir    : {out_dir}")
    print("=" * 70, flush=True)

    jobs = [(name, td) for name in designs for td in add]
    flat = sweep.run_jobs(jobs, contents, args.workers, args.bit_width)

    # 按 target_delay merge 回各设计的结果列表
    by_design = {name: {float(e["target_delay"]): e for e in results.get(name, [])} for name in designs}
    for design, td, entry in flat:
        by_design[design][float(td)] = entry
    for name in designs:
        results[name] = sorted(by_design[name].values(), key=lambda r: r["target_delay"])

    # 更新 delays 列表（并集，排序）
    all_delays = sorted(set([float(d) for d in delays]) | set(add))
    data_delays = all_delays

    sweep.save_artifacts(results, out_dir, base_run_dir, data_delays, paths, args.annotate)

    print("\nSummary (good points / total):")
    for name in designs:
        print(f"  {name:6s}: {len(sweep._good(results, name))}/{len(results[name])}")
    print(f"\nAll artifacts in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
