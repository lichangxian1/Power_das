#!/usr/bin/env python3
"""把一个额外设计的网表，以与 dc_timing_sweep_ppa.py 完全相同的标准送远端
DC timing-constraint sweep，合并进一份已有的 dc_timing_sweep.json，并重画
delay_vs_area / delay_vs_power 两张对比图（多出新增设计这一条线）。

复用 dc_timing_sweep_ppa 的评估 (evaluate_single_routing)、并发调度 (run_jobs)
和绘图 (save_artifacts/plot_curves)，因此 delays、目标延迟约束、远端 DC 流程、
绘图风格都与原 sweep 保持一致。

用法:
    python scripts/add_design_to_dc_sweep.py \
        --merge_json outputs/.../dc_timing_sweep_20260614_123637/dc_timing_sweep.json \
        --netlist /home/changxian/Arith-DAS/outputs/2026-05-06/14-50-02/mul_16_and_routing_gene/build_full_ppa/MUL.v \
        --name arith --label "Arith-DAS" --workers 6
"""

import argparse
import json
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/

import run_power_sweep
import dc_timing_sweep_ppa as sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge_json", required=True, help="已有 dc_timing_sweep.json")
    ap.add_argument("--netlist", required=True, help="新增设计的 MUL.v 路径")
    ap.add_argument("--name", default="arith", help="新增设计在 results 里的 key")
    ap.add_argument("--label", default="Arith-DAS", help="图例标签")
    ap.add_argument("--color", default="#9467bd")
    ap.add_argument("--marker", default="D")
    ap.add_argument(
        "--only_delays",
        default=None,
        help=(
            "只跑这些 target delays（空格/逗号分隔），结果按 target_delay merge "
            "进 results[name]，其余点保留。用于重试失败点。默认跑 json 里的全部 delays。"
        ),
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bit_width", type=int, default=16)
    ap.add_argument("--run_timeout", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
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

    name = args.name
    if name not in sweep.DESIGNS:
        sweep.DESIGNS.append(name)
    sweep.LABELS[name] = args.label
    sweep.COLORS[name] = args.color
    sweep.MARKERS[name] = args.marker

    netlist = os.path.abspath(args.netlist)
    if not os.path.exists(netlist):
        raise FileNotFoundError(f"netlist not found: {netlist}")
    with open(netlist) as f:
        contents = {name: f.read()}
    paths[name] = netlist

    out_dir = args.out_dir or (
        os.path.dirname(merge_json)
        + f"_plus_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Add design to DC timing sweep")
    print(f"  merge_json : {merge_json}")
    print(f"  new design : {name}  ({args.label})")
    print(f"  netlist    : {netlist}")
    print(f"  delays     : {delays}")
    print(f"  workers    : {args.workers}")
    print(f"  out_dir    : {out_dir}")
    print("=" * 70, flush=True)

    if args.only_delays:
        only = [
            float(x) for x in args.only_delays.replace(",", " ").split() if x.strip()
        ]
    else:
        only = delays
    jobs = [(name, td) for td in only]
    flat = sweep.run_jobs(jobs, contents, args.workers, args.bit_width)
    by_td = {float(e["target_delay"]): e for e in results.get(name, [])}
    for _design, td, entry in flat:
        by_td[float(td)] = entry
    results[name] = sorted(by_td.values(), key=lambda r: r["target_delay"])

    sweep.save_artifacts(results, out_dir, base_run_dir, delays, paths, args.annotate)

    print("\nSummary (good points / total):")
    for design in sweep.DESIGNS:
        print(f"  {design:6s}: {len(sweep._good(results, design))}/{len(results[design])}")
    print(f"\nAll artifacts in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
