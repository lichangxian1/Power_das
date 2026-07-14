#!/usr/bin/env python3
"""补测统一 substd 菜单里 12 个 22/32 cell 的 dyn power（同 dc_char 口径）。

这些 cell（library.json 里有 LUT）此前只按 standalone 面积 < 原生单元硬标准选型，
dyn 未表征（library.json power_mw=None/0）。本脚本从 LUT 发射独立 module，
远端 dc_char.tcl（compile_ultra, sp=0.25 tr=0.125）表征 area/dyn/tmax。

产物: cellsolver/substd2232_char.json
用法: python -m Appr_Comp.cellsolver.char_substd2232 [--run]
"""
import argparse
import json
import os
import subprocess
import sys
import uuid
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
APPR = os.path.dirname(HERE)
sys.path.insert(0, APPR)
import char_driver as cd  # noqa: E402
from gen_verilog import emit_module  # noqa: E402

CELLS = ["comp22_e4", "comp22_50", "comp22_a0", "comp22_00", "comp22_f0", "comp22_55",
         "comp32_fa50", "comp32_5500", "comp32_aa00", "comp32_5555",
         "comp32_ff00", "comp32_ff55"]
PAT2 = ["".join(p) for p in product("01", repeat=2)]
PAT3 = ["".join(p) for p in product("01", repeat=3)]


def rtl_text(lib):
    out = ["// substd 22/32 cells for dyn characterization (from library.json LUTs)"]
    for n in CELLS:
        c = lib[n]
        if c["type"] == "32":
            src = emit_module(n, ["a", "b", "cin"], PAT3,
                              c["sum_lut"], c["carry_lut"], f"{n} substd")
        else:
            src = emit_module(n, ["a", "cin"], PAT2,
                              c["sum_lut"], c["carry_lut"], f"{n} substd")
        out.append(src)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    lib = json.load(open(os.path.join(APPR, "library.json")))["cells"]

    rtl_path = os.path.join(APPR, "rtl", "comp2232_substd_standalone.v")
    with open(rtl_path, "w") as f:
        f.write(rtl_text(lib))
    print("wrote", rtl_path)
    if not args.run:
        print("(dry-run) modules:", CELLS)
        return

    mlist = os.path.join(HERE, "_substd2232_modules.txt")
    with open(mlist, "w") as f:
        f.write("\n".join(CELLS) + "\n")
    uid = uuid.uuid4().hex[:6]
    remote = f"{cd.EDA_WORK_ROOT.rstrip('/')}/char2232_{uid}"
    setup_cmd, run_cmd = cd.build_remote_cmds(remote, 0.25, 0.125, arcs=False)
    rsyncs = [["rsync", "-az", "-e", f"ssh -p {cd.EDA_PORT}", local,
               f"{cd.EDA_USER}@{cd.EDA_HOST}:{dest}"] for local, dest in [
        (rtl_path, f"{remote}/src/rtl/comp42_lib.v"),   # dc_char analyzes this slot
        (mlist, f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(cd.SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]]
    print(f"[char2232] setup {remote}")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", setup_cmd],
                   check=True, timeout=300)
    for c in rsyncs:
        subprocess.run(c, check=True, timeout=300)
    print("[char2232] running dc_char.tcl ...")
    res = subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", run_cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         timeout=1800)
    ppa = cd.parse_ppa((res.stdout or "").split("====PPA====", 1)[-1])
    if not any(not v.get("error") for v in ppa.values()):
        print("\n".join((res.stdout or "").splitlines()[-40:]))
        sys.exit(1)
    out = {}
    for n in CELLS:
        p = ppa.get(n, {})
        out[n] = {"area": p.get("area"),
                  "dyn_mw": (p["dyn_w"] * 1e3) if p.get("dyn_w") is not None else None,
                  "leak_mw": (p["leak_w"] * 1e3) if p.get("leak_w") is not None else None,
                  "tmax": p.get("tmax")}
        print(f"{n:<16} area={p.get('area')} dyn_mw={out[n]['dyn_mw']} tmax={p.get('tmax')}")
    json.dump({"meta": {"static_prob": 0.25, "toggle_rate": 0.125,
                        "caliber": "dc_char.tcl compile_ultra (同全库口径)"},
               "cells": out}, open(os.path.join(HERE, "substd2232_char.json"), "w"), indent=1)
    print("saved cellsolver/substd2232_char.json")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}",
                    f"rm -rf {remote}"], timeout=120)


if __name__ == "__main__":
    main()
