#!/usr/bin/env python3
"""Standalone-DC characterize the Sayadi TCSI'23 compressor cells with the SAME
caliber as Appr_Comp library cells (dc_char.tcl, sp=0.25 tr=0.125, compile_ultra).

Cells (4:2, no cin/cout, 2 outputs):
  sayadi_ac6g12 / sayadi_ac6g7 : the two AC6G carry isomorphism classes (6 gates)
  sayadi_acfgi                 : sum=1, carry=x4 (gate-free)
  sayadi_acfgii                : sum=x1, carry=x2 (gate-free)

Output: sayadi_cells.json {name: {wae,bias,er,maxe,area,dyn_w,leak_w,tmax}}
"""
import json
import os
import subprocess
import sys
import uuid

sys.path.insert(0, "/home/lee/Power_das/Appr_Comp")
import char_driver as cd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RTL = """\
// Sayadi TCSI'23 approximate 4:2 compressors (no cin/cout)
module sayadi_ac6g12 (a, b, c, d, sum, carry);
    input a, b, c, d; output sum, carry;
    assign sum   = a | b | c | d;
    assign carry = (c & (b | d)) | (a & d);
endmodule
module sayadi_ac6g7 (a, b, c, d, sum, carry);
    input a, b, c, d; output sum, carry;
    assign sum   = a | b | c | d;
    assign carry = (c & (a | b)) | (a & b);
endmodule
module sayadi_acfgi (a, b, c, d, sum, carry);
    input a, b, c, d; output sum, carry;
    assign sum   = 1'b1;
    assign carry = d;
endmodule
module sayadi_acfgii (a, b, c, d, sum, carry);
    input a, b, c, d; output sum, carry;
    assign sum   = a;
    assign carry = b;
endmodule
"""
# P=0.25 error stats (computed by sayadi_common truth tables, see RECON_REPORT)
ERR = {
    "sayadi_ac6g12": {"wae": 0.214844, "bias": -0.003906, "er": 0.214844, "maxe": 1},
    "sayadi_ac6g7":  {"wae": 0.214844, "bias": -0.003906, "er": 0.214844, "maxe": 1},
    "sayadi_acfgi":  {"wae": 0.765625, "bias": 0.5,       "er": 0.648438, "maxe": 2},
    "sayadi_acfgii": {"wae": 0.531250, "bias": -0.25,     "er": 0.484375, "maxe": 2},
}


def main():
    names = sorted(ERR)
    lib_v = os.path.join(HERE, "sayadi_cells.v")
    with open(lib_v, "w") as f:
        f.write(RTL)
    mlist = os.path.join(HERE, "sayadi_module_list.txt")
    with open(mlist, "w") as f:
        f.write("\n".join(names) + "\n")

    uid = uuid.uuid4().hex[:6]
    remote = f"{cd.EDA_WORK_ROOT.rstrip('/')}/charsayadi_{uid}"
    setup_cmd, run_cmd = cd.build_remote_cmds(remote, 0.25, 0.125, arcs=False)
    rsyncs = [["rsync", "-az", "-e", f"ssh -p {cd.EDA_PORT}", local,
               f"{cd.EDA_USER}@{cd.EDA_HOST}:{dest}"] for local, dest in [
        (lib_v, f"{remote}/src/rtl/comp42_lib.v"),
        (mlist, f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(cd.SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]]

    print(f"[char] setup {remote}")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", setup_cmd],
                   check=True, timeout=300)
    for cmd in rsyncs:
        subprocess.run(cmd, check=True, timeout=300)
    print("[char] running dc_char.tcl ...")
    res = subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", run_cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         timeout=1800)
    ppa = cd.parse_ppa((res.stdout or "").split("====PPA====", 1)[-1])
    print("[char] parsed:", {k: v.get("area") for k, v in ppa.items()})
    if not any(not v.get("error") for v in ppa.values()):
        print("\n".join((res.stdout or "").splitlines()[-40:]))
        sys.exit(1)

    out = {}
    for n in names:
        e = dict(ERR[n])
        e.update(ppa.get(n, {"error": "no_ppa"}))
        out[n] = e
    with open(os.path.join(HERE, "sayadi_cells.json"), "w") as f:
        json.dump({"meta": {"static_prob": 0.25, "toggle_rate": 0.125,
                            "caliber": "dc_char.tcl compile_ultra, same as Appr_Comp library"},
                   "cells": out}, f, indent=1)
    print("saved sayadi_cells.json")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}",
                    f"rm -rf {remote}"], timeout=120)


if __name__ == "__main__":
    main()
