#!/usr/bin/env python3
"""comp42s：Sayadi 启发的 sub-std 近似原生 4:2 cell 家族——生成 + DC 表征。

设计来源：enum_substd42.py 门级枚举 (wae, est_area) 帕累托 + P/N/Z 平衡。
两个结构家族（都无 XOR 富集的 parity sum，无 cin）：
  orha  ："OR 配对 + 半加器"  u=(a|b),v=(c|d)，sum=u^v，carry=u&v，
          cout 修正项区分 N/Z/P（0 / a&b / (a&b)|(c&d)），wae 0.117~0.125。
  or4   ：sum=OR4，carry∈{AO22, OA22, c&d, 0}，wae 0.215~0.316（AO/OA=P/N 对，
          与 Sayadi AC6G 同 wae 下界但 ~2 复合单元）。
  thru  ：馈通端点 sum=a,carry=b（= Sayadi ACFGII-1），wae 0.531。

产物：
  Appr_Comp/rtl/comp42s_standalone.v          （表征用独立文件）
  Appr_Comp/cellsolver/substd42_char.json     （library42_native 条目格式）
用法: python -m Appr_Comp.cellsolver.gen_substd42_cells [--run]（无 --run 只生成/打印）
"""
import argparse
import json
import os
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
APPR = os.path.dirname(HERE)
sys.path.insert(0, APPR)
import char_driver as cd  # noqa: E402

# name -> (sum, carry, cout) 的 (verilog expr, python lambda)
CELLS = {
    "comp42s_orha_n": ("(a | b) ^ (c | d)", "(a | b) & (c | d)", "1'b0"),
    "comp42s_orha_z": ("(a | b) ^ (c | d)", "(a | b) & (c | d)", "a & b"),
    "comp42s_orha_p": ("(a | b) ^ (c | d)", "(a | b) & (c | d)", "(a & b) | (c & d)"),
    "comp42s_or4ao_n": ("a | b | c | d", "(a & b) | (c & d)", "1'b0"),
    "comp42s_or4oa_p": ("a | b | c | d", "(a | b) & (c | d)", "1'b0"),
    "comp42s_or4cd_n": ("a | b | c | d", "c & d", "1'b0"),
    "comp42s_or4_n": ("a | b | c | d", "1'b0", "1'b0"),
    "comp42s_thru_n": ("a", "b", "1'b0"),
}


def evl(expr, a, b, c, d):
    if expr == "1'b0":
        return 0
    return int(eval(expr.replace("^", "!=").replace("&", " and ").replace("|", " or "),
                    {}, {"a": a, "b": b, "c": c, "d": d}) in (True, 1))


def stats(exprs):
    s_l, c_l, o_l, v_l = [], [], [], []
    wae = bias = er = 0.0
    maxe = 0
    for p in range(16):
        a, b, c, d = (p >> 0) & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1
        s, c1, c2 = (evl(e, a, b, c, d) for e in exprs)
        v = s + 2 * (c1 + c2)
        e_ = v - (a + b + c + d)
        prob = 1.0
        for bit in (a, b, c, d):
            prob *= 0.25 if bit else 0.75
        wae += prob * abs(e_)
        bias += prob * e_
        er += prob * (e_ != 0)
        maxe = max(maxe, abs(e_))
        s_l.append(s); c_l.append(c1); o_l.append(c2); v_l.append(v)
    return dict(sum_lut=s_l, carry_lut=c_l, cout_lut=o_l, v_lut=v_l,
                wae=round(wae, 6), bias=round(bias, 6), er=round(er, 6), maxe=maxe)


def rtl_text():
    out = ["// comp42s: Sayadi-inspired sub-std approximate native 4:2 cells",
           "// (gen_substd42_cells.py; no cin; cout may be constant 0)"]
    for name, (s, c1, c2) in CELLS.items():
        st = stats((s, c1, c2))
        out.append(f"""
// {name} bias={st['bias']:+.4f} wae={st['wae']:.4f} er={st['er']:.4f} maxe={st['maxe']}
module {name} (a, b, c, d, sum, carry, cout);
    input  a, b, c, d;
    output sum, carry, cout;
    assign sum   = {s};
    assign carry = {c1};
    assign cout  = {c2};
endmodule""")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="远端 DC 表征")
    args = ap.parse_args()

    for name, exprs in CELLS.items():
        st = stats(exprs)
        print(f"{name:<20} wae={st['wae']:.4f} bias={st['bias']:+.4f} "
              f"er={st['er']:.4f} maxe={st['maxe']}")
    rtl_path = os.path.join(APPR, "rtl", "comp42s_standalone.v")
    with open(rtl_path, "w") as f:
        f.write(rtl_text())
    print("wrote", rtl_path)
    if not args.run:
        return

    mlist = os.path.join(HERE, "_substd42_modules.txt")
    with open(mlist, "w") as f:
        f.write("\n".join(CELLS) + "\n")
    uid = uuid.uuid4().hex[:6]
    remote = f"{cd.EDA_WORK_ROOT.rstrip('/')}/char42s_{uid}"
    setup_cmd, run_cmd = cd.build_remote_cmds(remote, 0.25, 0.125, arcs=False)
    rsyncs = [["rsync", "-az", "-e", f"ssh -p {cd.EDA_PORT}", local,
               f"{cd.EDA_USER}@{cd.EDA_HOST}:{dest}"] for local, dest in [
        (rtl_path, f"{remote}/src/rtl/comp42_lib.v"),
        (mlist, f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(cd.SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]]
    print(f"[char42s] setup {remote}")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", setup_cmd],
                   check=True, timeout=300)
    for cmd in rsyncs:
        subprocess.run(cmd, check=True, timeout=300)
    print("[char42s] running dc_char.tcl ...")
    res = subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", run_cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         timeout=1800)
    ppa = cd.parse_ppa((res.stdout or "").split("====PPA====", 1)[-1])
    if not any(not v.get("error") for v in ppa.values()):
        print("\n".join((res.stdout or "").splitlines()[-40:]))
        sys.exit(1)

    out = {}
    for name, exprs in CELLS.items():
        st = stats(exprs)
        p = ppa.get(name, {})
        bias = st["bias"]
        out[name] = {
            "type": "42", "pattern_bits": 4, "family": "sayadi_substd",
            "group": "Z" if abs(bias) < 0.01 else ("P" if bias > 0 else "N"),
            "is_exact": False,
            "v_lut": st["v_lut"], "sum_lut": st["sum_lut"],
            "carry_lut": st["carry_lut"], "cout_lut": st["cout_lut"],
            "bias": bias, "wae": st["wae"], "er": st["er"], "maxe": st["maxe"],
            "area": p.get("area"),
            "dyn_mw": (p["dyn_w"] * 1e3) if p.get("dyn_w") is not None else None,
            "leak_mw": (p["leak_w"] * 1e3) if p.get("leak_w") is not None else None,
            "tmax": p.get("tmax"),
            "Tsum": None, "Tcarry": None, "Tcout": None,
        }
        print(f"{name:<20} area={p.get('area')} dyn_mw={out[name]['dyn_mw']} "
              f"tmax={p.get('tmax')}")
    with open(os.path.join(HERE, "substd42_char.json"), "w") as f:
        json.dump({"meta": {"static_prob": 0.25, "toggle_rate": 0.125,
                            "native_std_area": 5.712,
                            "caliber": "dc_char.tcl compile_ultra (同 comp42n 库口径)"},
                   "cells": out}, f, indent=1)
    print("saved cellsolver/substd42_char.json")
    subprocess.run(["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}",
                    f"rm -rf {remote}"], timeout=120)


if __name__ == "__main__":
    main()
