#!/usr/bin/env python3
"""本地 PPA 流程(yosys + OpenROAD STA, TSMC 28nm)——镜像 Power_das 自带流程，
使本 baseline 的 area/delay/power 与本项目自己的乘法器逐一可比。

每个 design: yosys 综合(synth+dfflibmap+abc -D200 -constr -liberty) -> netlist
            -> OpenROAD STA(虚拟时钟, 组合路径) -> area/delay/power -> PDAP。
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB  = "/home/lee/library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib"
LEF  = "/home/lee/library/t28_official/tcbn28hpcplusbwp12t40p140.lef"
YOSYS = "/home/lee/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys"
ORD   = "/home/lee/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
RTL  = os.path.join(HERE, "rtl")

YOSYS_YS = """read -sv {srcs}
synth -top {top}
dfflibmap -liberty {lib}
abc -D 200 -constr {sdc} -liberty {lib}
write_verilog {netlist}
"""
SDC = "set_driving_cell BUFFD1BWP12T40P140\nset_load 10.0 [all_outputs]\n"

STA_TCL = """read_lef {lef}
read_lib {lib}
read_verilog {netlist}
link_design {top}
create_clock -name vclk -period 5
set clk [lindex [all_clocks] 0]
set_input_delay 0 -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay 0 -clock $clk [all_outputs]
set crit [lindex [find_timing_paths -sort_by_slack] 0]
set d [sta::format_time [[$crit path] arrival] 5]
puts "PPA_DELAY_NS $d"
report_design_area
set_power_activity -input -activity 0.5
report_power
exit
"""

def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode

def synth(top, srcs, wd):
    netlist = os.path.join(wd, f"{top}.netlist.v")
    sdc = os.path.join(wd, "constr.sdc"); open(sdc, "w").write(SDC)
    ys = os.path.join(wd, f"{top}.ys")
    open(ys, "w").write(YOSYS_YS.format(srcs=" ".join(srcs), top=top, lib=LIB, sdc=sdc, netlist=netlist))
    rc = run([YOSYS, "-q", ys], os.path.join(wd, f"{top}.yosys.log"))
    return netlist if (rc == 0 and os.path.exists(netlist)) else None

def sta(top, netlist, wd):
    tcl = os.path.join(wd, f"{top}.sta.tcl")
    open(tcl, "w").write(STA_TCL.format(lef=LEF, lib=LIB, netlist=netlist, top=top))
    log = os.path.join(wd, f"{top}.sta.log")
    run([ORD, "-no_init", "-exit", tcl], log)
    return open(log).read()

def parse(out):
    d = {}
    m = re.search(r"PPA_DELAY_NS\s+([\d.eE+-]+)", out)
    if m: d["delay_ns"] = float(m.group(1))
    m = re.search(r"Design area\s+([\d.eE+-]+)\s*u", out)
    if m: d["area_um2"] = float(m.group(1))
    # report_power Total 行: Internal Switching Leakage Total(W) <pct>%  -> 取倒数第2个为总功耗(W)
    for line in out.splitlines():
        if line.strip().lower().startswith("total"):
            nums = re.findall(r"[\d.]+e[+-]?\d+|\d+\.\d+", line)
            if len(nums) >= 2: d["power_w"] = float(nums[-2])
    return d

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    cells = os.path.join(RTL, "el4_cells.v")
    designs = [
        ("exact_8",  [os.path.join(RTL, "exact_8.v")]),
        ("mul1_8",   [cells, os.path.join(RTL, "mul1_8.v")]),
        ("mul2_8",   [cells, os.path.join(RTL, "mul2_8.v")]),
        ("exact_16", [os.path.join(RTL, "exact_16.v")]),
        ("mul1_16",  [cells, os.path.join(RTL, "mul1_16.v")]),
        ("mul2_16",  [cells, os.path.join(RTL, "mul2_16.v")]),
    ]
    rows = []
    for top, srcs in designs:
        nl = synth(top, srcs, outdir)
        if not nl:
            print(f"[{top}] SYNTH FAIL"); rows.append((top, None)); continue
        out = sta(top, nl, outdir)
        rows.append((top, parse(out)))
        m = rows[-1][1]
        print(f"[{top}] area={m.get('area_um2')} delay_ns={m.get('delay_ns')} power_w={m.get('power_w')}")
    # 汇总表 + PDAP + 相对 exact
    import json
    res = {t: m for t, m in rows}
    json.dump(res, open(os.path.join(outdir, "ppa.json"), "w"), indent=2)
    def pdap(m):
        if not m or any(k not in m for k in ("area_um2","delay_ns","power_w")): return None
        return m["area_um2"] * m["delay_ns"] * m["power_w"] * 1e6  # 任意归一(uW·ns·um2 量纲一致即可比)
    lines = ["design     area(um2)  delay(ns)  power(uW)   PDAP(rel)"]
    for top, m in rows:
        if not m: lines.append(f"{top:10s} FAIL"); continue
        p = pdap(m)
        lines.append(f"{top:10s} {m.get('area_um2',0):9.2f} {m.get('delay_ns',0):9.3f} "
                     f"{m.get('power_w',0)*1e6:9.3f}  {p:.3f}" if p else f"{top:10s} partial {m}")
    table = "\n".join(lines)
    print("\n" + table)
    open(os.path.join(outdir, "ppa_table.txt"), "w").write(table + "\n")

if __name__ == "__main__":
    main()
