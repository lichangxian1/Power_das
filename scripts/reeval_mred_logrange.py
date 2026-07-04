#!/usr/bin/env python3
"""DC+XA reeval for MRED log-range sweeps.

Usage:
  python scripts/reeval_mred_logrange.py outputs/2026-07-04_mred_logrange_trunc_cstar [workers]

The output CSV keeps the real MRED from best_info, then adds DC area + XA power.
Run on the remote machine that can reach the EDA host.  MUL.v is expected in
each k*/ directory; the sync script brings it back locally after training.
"""
import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from run_power_sweep import evaluate_single_routing  # noqa: E402


def wrap_to_31b(src):
    m = re.search(r"output\s+wire\s+\[(\d+):0\]\s+out", src)
    am = re.search(r"input\s+wire\s+\[(\d+):0\]\s+a", src)
    if not (m and am):
        return src
    fw, aw = int(m.group(1)), int(am.group(1))
    if fw <= 30:
        return src
    core = src.replace("module MUL(", "module MUL_core(", 1)
    wrapper = (
        "module MUL(\n    input wire clk,\n"
        f"    input wire [{aw}:0] a,\n    input wire [{aw}:0] b,\n"
        "    output wire [30:0] out\n);\n"
        f"    wire [{fw}:0] full_out;\n"
        "    MUL_core u_core(.clk(clk), .a(a), .b(b), .out(full_out));\n"
        "    assign out = full_out[30:0];\n"
        "endmodule\n\n"
    )
    return wrapper + core


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: reeval_mred_logrange.py <BASE_REL_DIR> [workers]")
    base = sys.argv[1].rstrip("/")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("REEVAL_WORKERS", "6"))

    jobs = []
    for d in sorted(glob.glob(f"{base}/k*/")):
        kk = os.path.basename(d.rstrip("/"))
        bi_path = os.path.join(d, "best_info.json")
        rtl_path = os.path.join(d, "MUL.v")
        if not (os.path.exists(bi_path) and os.path.exists(rtl_path)):
            print(f"skip {kk}: missing best_info.json or MUL.v", flush=True)
            continue
        info = json.load(open(bi_path))
        me = info.get("measured_error") or {}
        n_approx = len(info.get("approx_cells") or {})
        jobs.append({
            "design": kk,
            "mred": me.get("mred"),
            "med": me.get("med"),
            "n_approx": n_approx,
            "rtl": wrap_to_31b(open(rtl_path).read()),
        })

    print(f"{base}: {len(jobs)} designs -> {[j['design'] for j in jobs]}", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(evaluate_single_routing, i, j["rtl"], 16, 1.5): j
                for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            j = futs[fut]
            r = fut.result()
            results[j["design"]] = r
            print(
                f"{j['design']}: success={r.get('success')} area={r.get('area')} "
                f"power_mw={r.get('power_mw')} delay={r.get('delay')}",
                flush=True,
            )

    out = f"{base}/reeval_xa.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "mred", "med", "n_approx", "area_dc", "power_xa_mw", "delay", "success"])
        for j in jobs:
            r = results[j["design"]]
            w.writerow([
                j["design"], j["mred"], j["med"], j["n_approx"],
                r.get("area"), r.get("power_mw"), r.get("delay"), r.get("success"),
            ])
    print("saved ->", out)


if __name__ == "__main__":
    main()
