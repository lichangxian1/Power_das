#!/usr/bin/env python3
"""把一次训练 run 的 eda / proxy 网表送远端 DC timing-constraint sweep，并与一份
已有 dc_timing_sweep.json 里 *已经综合好* 的 designware(dw) 和 Arith-DAS(05-11,
即 arith2) 结果合并，产出 4 条对比曲线的 JSON/CSV，再调 replot 画
area_vs_delay.png / power_vs_delay.png（横轴=延迟，纵轴=面积/功耗）。

- 只对新 run 的 eda/proxy 两个网表跑远端 DC（其余复用 base_json，已有综合结果不再重跑）。
- delays / 目标延迟约束 / 远端 DC 流程 / 绘图风格都与原 sweep 一致。

用法:
    python scripts/dc_sweep_routing_gene_compare.py \
        --run_dir outputs/area_budget_sweep/20260615_195403 \
        --base_json outputs/area_budget_sweep/20260613_163000/dc_timing_sweep_20260614_123637_plus_arith/dc_timing_sweep.json \
        --workers 6
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/

import re

import run_power_sweep
import dc_timing_sweep_ppa as sweep

# 复用的、已综合好的设计（直接从 base_json 拷贝，不再跑 DC）
REUSE = ["dw", "arith2"]
# 用新 run 重新综合的设计
FRESH = ["eda", "proxy"]


def adapt_out_width(content, target_msb=30):
    """远端 testbench (src/tb/mult_TB2.v) 把 31bit 线网 m_netlist 接到 DUT 的
    .out 端口，并显式容忍被丢掉的最高位 (m === true_p - high_bit)。AND-array /
    DesignWare / Arith-DAS 等所有对比设计的 out 都是 [30:0] (31bit)。本次 Booth
    网表的顶层 out 是 [31:0] (32bit)，会触发 VCS 的 IOPCWM 端口位宽不匹配致命错误。

    这里把顶层 MUL 重命名为 MUL_core，并外面包一层 31bit 的 MUL 适配壳，只引出低
    31 位 (丢掉的 out[31] 正是 testbench 对所有设计都会忽略的那一位)，从而与其余
    设计在同一 harness 下做公平对比。若 out 本就是 [30:0] 则原样返回。"""
    m = re.search(r"module\s+MUL\s*\(", content)
    if m is None:
        return content
    # 找顶层 out 端口宽度
    port = re.search(r"output\s+(?:wire\s+)?\[(\d+):0\]\s*out\b", content)
    if port is None or int(port.group(1)) == target_msb:
        return content  # 已经是目标位宽
    msb = int(port.group(1))
    core = content[: m.start()] + content[m.start():].replace("module MUL(", "module MUL_core(", 1).replace("module MUL (", "module MUL_core (", 1)
    wrapper = (
        "\nmodule MUL(\n"
        "    input wire clk,\n"
        "    input wire [15:0] a,\n"
        "    input wire [15:0] b,\n"
        f"    output wire [{target_msb}:0] out\n"
        ");\n"
        f"    wire [{msb}:0] out_full;\n"
        "    MUL_core u_core(.clk(clk), .a(a), .b(b), .out(out_full));\n"
        f"    assign out = out_full[{target_msb}:0];\n"
        "endmodule\n"
    )
    return wrapper + "\n" + core


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, help="新训练 run 目录（含 power_source_{eda,proxy}/unconstrained/MUL.v）")
    ap.add_argument("--base_json", required=True, help="含 dw/arith2 综合结果的 dc_timing_sweep.json")
    ap.add_argument("--delays", default=None, help="空格/逗号分隔的 target delays；默认用 base_json 的 delays")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bit_width", type=int, default=16)
    ap.add_argument("--run_timeout", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if args.run_timeout is not None:
        run_power_sweep.EDA_RUN_TIMEOUT = args.run_timeout
        print(f"[override] EDA_RUN_TIMEOUT = {args.run_timeout}s", flush=True)

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(_REPO_ROOT, args.run_dir)
    base_json = args.base_json if os.path.isabs(args.base_json) else os.path.join(_REPO_ROOT, args.base_json)

    with open(base_json) as f:
        base = json.load(f)
    delays = (
        [float(x) for x in args.delays.replace(",", " ").split() if x.strip()]
        if args.delays
        else base["delays"]
    )

    # 新 run 的 eda/proxy 网表
    contents, paths = {}, dict(base.get("paths", {}))
    for name in FRESH:
        p = os.path.join(run_dir, f"power_source_{name}", "unconstrained", "MUL.v")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} netlist not found: {p}")
        with open(p) as fh:
            raw = fh.read()
        adapted = adapt_out_width(raw)
        if adapted != raw:
            print(f"  [{name}] 顶层 out 适配为 31bit ([30:0]) 以匹配远端 testbench", flush=True)
        contents[name] = adapted
        paths[name] = p

    out_dir = args.out_dir or os.path.join(
        run_dir, f"dc_timing_sweep_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("DC timing sweep — routing_gene eda/proxy vs reused dw/arith2")
    print(f"  run_dir   : {run_dir}")
    print(f"  base_json : {base_json}")
    print(f"  fresh     : {FRESH}  reuse: {REUSE}")
    print(f"  delays    : {delays}")
    print(f"  workers   : {args.workers}")
    print(f"  total DC  : {len(FRESH) * len(delays)} runs")
    print(f"  out_dir   : {out_dir}")
    print("=" * 70, flush=True)

    jobs = [(name, td) for name in FRESH for td in delays]
    flat = sweep.run_jobs(jobs, contents, args.workers, args.bit_width)

    results = {}
    for name in FRESH:
        results[name] = []
    for design, _td, entry in flat:
        results[design].append(entry)
    for name in FRESH:
        results[name].sort(key=lambda r: r["target_delay"])

    # 复用已综合的 dw / arith2
    for name in REUSE:
        if name not in base["results"]:
            raise KeyError(f"base_json 缺少已综合设计: {name}")
        results[name] = base["results"][name]

    order = FRESH + REUSE
    json_path = os.path.join(out_dir, "dc_timing_sweep.json")
    with open(json_path, "w") as f:
        json.dump({"run_dir": run_dir, "base_json": base_json, "delays": delays,
                   "paths": paths, "results": results}, f, indent=2)
    print(f"Wrote {json_path}", flush=True)

    csv_path = os.path.join(out_dir, "dc_timing_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "target_delay_ns", "area_um2", "delay_ns", "power_mw"])
        for name in order:
            for r in results[name]:
                w.writerow([name, r["target_delay"], r.get("area"), r.get("delay"), r.get("power_mw")])
    print(f"Wrote {csv_path}", flush=True)

    print("\nSummary (good points / total):")
    for name in order:
        good = [r for r in results[name] if r.get("area") and r.get("delay") and r.get("power_mw")]
        print(f"  {name:6s}: {len(good)}/{len(results[name])}")
    print(f"\nJSON ready for replot: {json_path}", flush=True)


if __name__ == "__main__":
    main()
