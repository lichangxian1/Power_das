#!/usr/bin/env python3
"""Shared evaluation harness for Power_das third-party baselines (same 口径 as the project).

measure_med(rtl_path, n_vectors)  -> dict(med, bias, wce_mc) | None
    verilator circular-wrap real MED on a 16x16 AND-encoded `module MUL(clk,a,b,out[30:0])`,
    identical to Power_das/verilate/mul_err_wrap.cpp (16M common-random vectors, 31-bit mask).

measure_ppa(rtl_str, target_delay=1.5) -> dict(area, power_mw, delay, success)
    DC area + XA power via run_power_sweep.evaluate_single_routing (remote TSMC, 16-bit @1.5ns),
    the same call the project uses for its own designs and other baselines.

Both reproduce the established baseline pipeline (cf. ELEX4_N, trunc_dadda_baseline).
"""
import os
import shutil
import subprocess
import sys

PD = "/home/lee/Power_das"
HARNESS = os.path.join(PD, "verilate", "mul_err_wrap.cpp")


def measure_med(rtl_path, n_vectors=16_000_000, build_root=None):
    """Verilate `module MUL` + mul_err_wrap.cpp, run, parse 'masked,MED,BIAS,RMSE,ER,WCE'.
    Returns dict(med,bias,wce_mc) or None on failure. Mirrors
    CompressorRouting._measure_error_verilator (-O3 build, top-module MUL)."""
    rtl_abs = os.path.abspath(rtl_path)
    base = build_root or os.path.join(os.path.dirname(rtl_abs), "_verr")
    for attempt in range(2):
        verr = os.path.abspath(os.path.join(base, f"a{attempt}"))
        try:
            shutil.rmtree(verr, ignore_errors=True)
            os.makedirs(verr, exist_ok=True)
            obj = os.path.join(verr, "obj_dir")
            exe = os.path.join(obj, "mul_err")
            bcmd = ["verilator", "--cc", "--exe", "--build", "-j", "1", "-O3",
                    "-Wno-fatal", "--top-module", "MUL", "--Mdir", obj,
                    rtl_abs, HARNESS, "-o", "mul_err"]
            b = subprocess.run(bcmd, cwd=verr, capture_output=True, text=True, timeout=240)
            if b.returncode != 0 or not os.path.exists(exe):
                raise RuntimeError(f"verilator build rc={b.returncode}: {b.stderr[-500:]}")
            r = subprocess.run([exe, str(int(n_vectors))], cwd=verr,
                               capture_output=True, text=True, timeout=240)
            if r.returncode != 0:
                raise RuntimeError(f"verilator run rc={r.returncode}")
            for line in r.stdout.strip().splitlines():
                p = line.split(",")
                if p[0] == "masked":
                    shutil.rmtree(verr, ignore_errors=True)
                    return {"med": float(p[1]), "bias": float(p[2]), "wce_mc": float(p[5])}
            raise RuntimeError("no masked line in: " + r.stdout[-300:])
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[measure_med] attempt {attempt} failed: {e}\n")
            shutil.rmtree(verr, ignore_errors=True)
    return None


def measure_ppa(rtl_str, idx=0, target_delay=1.5, work_dir=None):
    """DC area + XA power @target_delay via the project's remote pipeline.
    Returns dict(area, power_mw, delay, success).

    evaluate_single_routing only uses a local `build/` dir to stage the .v before rsyncing
    to the remote EDA host (everything else is absolute/remote). On this machine PD/build is
    a root-owned ramdisk symlink (not writable), so we run from a writable scratch dir that
    has its own build/ — the import path still points at Power_das for the EDA constants."""
    cwd = os.getcwd()
    sys.path.insert(0, PD)
    wd = work_dir or os.path.join(os.environ.get("BASELINE_DC_WORK", "/tmp/claude-1000"), "dc_stage")
    os.makedirs(os.path.join(wd, "build"), exist_ok=True)
    os.chdir(wd)
    try:
        from run_power_sweep import evaluate_single_routing
        r = evaluate_single_routing(idx, rtl_str, 16, target_delay)
        return {"area": r.get("area"), "power_mw": r.get("power_mw"),
                "delay": r.get("delay"), "success": r.get("success"), "log": r.get("log")}
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    # smoke test on the trunc baseline's k08 RTL if present
    t = "/home/lee/Baselines/trunc_dadda_baseline/rtl/MUL_k08.v"
    if os.path.exists(t):
        print("MED:", measure_med(t, 1_000_000))
