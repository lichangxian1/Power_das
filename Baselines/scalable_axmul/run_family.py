#!/usr/bin/env python3
"""Build a scalable AxM family in the project 口径: emit RTL per knob value, verify golden==RTL
(verilator, sample), measure real MED (verilator 16M circular-wrap) + remote DC area/XA power
@1.5ns. Writes {fam}_results.csv.

Usage:
  python run_family.py drum verify      # golden==RTL spot check (a few knobs)
  python run_family.py drum med         # emit + MED only (local)
  python run_family.py drum ppa         # emit + MED + remote DC (full)
  python run_family.py mitchell ppa
"""
import csv
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/lee/Baselines")
from common_eval import measure_med, measure_ppa  # noqa: E402
import drum
import mitchell

FAM = {"drum": drum, "mitchell": mitchell}
NVEC = int(os.environ.get("NVEC", "16000000"))


def rtl_path(fam, p):
    d = os.path.join(HERE, "rtl"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{fam}_{p:02d}.v")
    open(path, "w").write(FAM[fam].emit(p))
    return path


def verify(fam, params, nvec=400000):
    """golden==RTL on random+corner vectors via verilator (masked 31-bit)."""
    m = FAM[fam]
    rng = random.Random(7)
    P = set((x, y) for x in (0, 1, 2, 3, 255, 256, 65535, 65534, 32768)
            for y in (0, 1, 2, 3, 255, 256, 65535, 65534, 32768))
    while len(P) < nvec:
        P.add((rng.randrange(1 << 16), rng.randrange(1 << 16)))
    ok = True
    for p in params:
        path = rtl_path(fam, p)
        vec = os.path.join(HERE, f"_vec_{fam}_{p}.txt")
        with open(vec, "w") as f:
            for a, b in P:
                f.write(f"{a} {b} {m.golden(a, b, p) & 0x7FFFFFFF}\n")
        obj = os.path.join(HERE, f"_obj_{fam}_{p}")
        subprocess.run(["rm", "-rf", obj])
        bc = subprocess.run(["verilator", "--cc", "--exe", "--build", "-j", "2", "-O3",
                             "-Wno-fatal", "--top-module", "MUL", "--Mdir", obj,
                             path, os.path.join(HERE, "tb_eq.cpp"), "-o", "sim",
                             "-CFLAGS", "-O1"], capture_output=True, text=True)
        if bc.returncode != 0:
            print(f"[{fam} {p}] BUILD FAIL: {bc.stderr[-300:]}"); ok = False; continue
        r = subprocess.run([os.path.join(obj, "sim"), vec], capture_output=True, text=True)
        print(f"[{fam} {p:2d}] {r.stdout.strip()}")
        if "mismatches=0" not in r.stdout:
            ok = False
        subprocess.run(["rm", "-rf", obj]); os.remove(vec)
    print("VERIFY:", "ALL PASS" if ok else "FAILURES")
    return ok


def run(fam, mode):
    m = FAM[fam]
    params = m.SWEEP
    rows = []
    for p in params:
        path = rtl_path(fam, p)
        me = measure_med(path, NVEC)
        med = me["med"] if me else None
        bias = me["bias"] if me else None
        r = {}
        if mode == "ppa":
            r = measure_ppa(open(path).read(), idx=p, target_delay=1.5)
        print(f"[{fam} {p:2d}] MED={med} bias={bias} area={r.get('area')} "
              f"power_mw={r.get('power_mw')} delay={r.get('delay')} ok={r.get('success')}", flush=True)
        rows.append((p, med, bias, r.get("area"), r.get("power_mw"), r.get("delay"), r.get("success")))
    out = os.path.join(HERE, f"{fam}_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "med", "bias", "area_dc", "power_xa_mw", "delay", "success"])
        w.writerows(rows)
    print("saved ->", out)


if __name__ == "__main__":
    fam, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "med")
    if mode == "verify":
        verify(fam, FAM[fam].SWEEP[::3] + [FAM[fam].SWEEP[-1]])
    else:
        run(fam, mode)
