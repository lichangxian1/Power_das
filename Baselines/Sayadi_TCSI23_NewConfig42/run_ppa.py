#!/usr/bin/env python3
"""统一口径评测: verilator 16M circular-wrap MED + DC area + XA power @1.5ns.

Uses /home/lee/Baselines/common_eval.py (same pipeline as ELEX4_N / trunc_dadda /
Zhang baselines). Results -> results_ppa.json.

Usage: python3 run_ppa.py [--med-only|--ppa-only]
"""
import json
import os
import sys

sys.path.insert(0, "/home/lee/Baselines")
from common_eval import measure_med, measure_ppa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_ppa.json")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    for idx, mul in enumerate(("mul1", "mul2")):
        rtl = os.path.join(HERE, "rtl", f"sayadi_{mul}_16.v")
        res.setdefault(mul, {})
        if mode != "--ppa-only":
            print(f"[{mul}] verilator 16M wrap-MED ...", flush=True)
            m = measure_med(rtl, 16_000_000)
            print(f"[{mul}] MED={m['med']:.1f} bias={m['bias']:.1f}", flush=True)
            res[mul]["med"] = m
        if mode != "--med-only":
            print(f"[{mul}] DC+XA @1.5ns (remote) ...", flush=True)
            with open(rtl) as f:
                rtl_str = f.read()
            p = measure_ppa(rtl_str, idx=idx, target_delay=1.5)
            print(f"[{mul}] area={p['area']} power_mw={p['power_mw']} "
                  f"delay={p['delay']} success={p['success']}", flush=True)
            res[mul]["ppa"] = {k: p[k] for k in ("area", "power_mw", "delay", "success")}
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
