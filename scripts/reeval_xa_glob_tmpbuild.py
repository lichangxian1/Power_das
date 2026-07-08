#!/usr/bin/env python3
"""XA reeval with a writable temporary local build directory.

Same output schema as reeval_xa_glob.py, but avoids the repo-level build symlink
when it points to a non-writable ramdisk.
"""
import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/lee/Power_das")
sys.path.insert(0, str(ROOT))
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
    base = Path(sys.argv[1]).resolve()
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("REEVAL_WORKERS", "10"))
    tmp = Path(os.environ.get("XA_LOCAL_TMP", "/tmp/power_das_xa_tmpbuild"))
    tmp.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp)

    jobs = []
    for p in sorted(x for x in base.glob("k*") if x.is_dir()):
        kk = p.name
        bi, mv = p / "best_info.json", p / "MUL.v"
        if not (bi.exists() and mv.exists()):
            print("skip (no best_info/MUL.v):", kk)
            continue
        info = json.load(open(bi))
        measured = info.get("measured_error") or {}
        med = measured.get("med")
        mred = measured.get("mred")
        n_approx = measured.get("n_approx")
        wce_mc = measured.get("wce_mc")
        jobs.append((kk, med, mred, n_approx, wce_mc, wrap_to_31b(open(mv).read())))

    print(f"{base}: {len(jobs)} designs -> {[j[0] for j in jobs]}", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(evaluate_single_routing, i, content, 16, 1.5): (kk, med)
            for i, (kk, med, _mred, _n_approx, _wce_mc, content) in enumerate(jobs)
        }
        for fut in as_completed(futs):
            kk, med = futs[fut]
            r = fut.result()
            res[kk] = (med, r)
            print(
                f"{kk}: success={r.get('success')} area={r.get('area')} "
                f"power_mw={r.get('power_mw')} delay={r.get('delay')}",
                flush=True,
            )

    out = base / "reeval_xa.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "med", "mred", "n_approx", "wce_mc", "area_dc", "power_xa_mw", "delay", "success"])
        for kk, med, mred, n_approx, wce_mc, _ in jobs:
            med, r = res[kk]
            w.writerow(
                [
                    kk,
                    med,
                    mred,
                    n_approx,
                    wce_mc,
                    r.get("area"),
                    r.get("power_mw"),
                    r.get("delay"),
                    r.get("success"),
                ]
            )
    print("saved ->", out)


if __name__ == "__main__":
    main()
