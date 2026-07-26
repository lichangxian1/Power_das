#!/usr/bin/env python3
"""16-bit Zhang baselines through the project 口径: verilator MED + remote DC area + XA power
@1.5ns. Mirrors trunc_dadda_baseline / reeval_xa_generic. Writes zhang_ppa.csv + zhang_error.csv.

Usage:
  python run_zhang_ppa.py med     # verilator MED only (local, fast)
  python run_zhang_ppa.py ppa     # MED + remote DC PPA (uses DC license)
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/lee/Baselines")
from common_eval import measure_med, measure_ppa  # noqa: E402

DESIGNS = [("Zhang-Proposed-16", "rtl/MUL_proposed_16.v"),
           ("Zhang-ProposedH-16", "rtl/MUL_proposedH_16.v")]
NVEC = int(os.environ.get("NVEC", "16000000"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "med"
    err = {}
    for name, rel in DESIGNS:
        m = measure_med(os.path.join(HERE, rel), NVEC)
        err[name] = m
        print(f"[MED] {name}: {m}", flush=True)
    with open(os.path.join(HERE, "zhang_error.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["design", "med", "bias", "wce_mc"])
        for name, _ in DESIGNS:
            m = err[name] or {}
            w.writerow([name, m.get("med"), m.get("bias"), m.get("wce_mc")])
    if mode != "ppa":
        print("MED done -> zhang_error.csv (run with 'ppa' for DC).")
        return
    rows = []
    for i, (name, rel) in enumerate(DESIGNS):
        rtl = open(os.path.join(HERE, rel)).read()
        r = measure_ppa(rtl, idx=i, target_delay=1.5)
        rows.append((name, err[name], r))
        print(f"[PPA] {name}: success={r.get('success')} area={r.get('area')} "
              f"power_mw={r.get('power_mw')} delay={r.get('delay')}", flush=True)
    with open(os.path.join(HERE, "zhang_ppa.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "med", "bias", "area_dc", "power_xa_mw", "delay", "success"])
        for name, m, r in rows:
            m = m or {}
            w.writerow([name, m.get("med"), m.get("bias"), r.get("area"),
                        r.get("power_mw"), r.get("delay"), r.get("success")])
    print("PPA done -> zhang_ppa.csv")


if __name__ == "__main__":
    main()
