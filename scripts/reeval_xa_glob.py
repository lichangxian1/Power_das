#!/usr/bin/env python3
"""XA reeval over all k*/ subdirs of BASE: area(DC)+power(XA)+verilator MED(取自 best_info)。
同口径 reeval_xa_generic.py，但不写死 KS，自动 glob 现有 k*/（适配只跑了部分 k 的 sweep）。
用法: python reeval_xa_glob.py <BASE_REL_DIR> [workers]。输出 {BASE}/reeval_xa.csv。"""
import csv, glob, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(_ROOT); sys.path.insert(0, _ROOT)
from run_power_sweep import evaluate_single_routing


def wrap_to_31b(src):
    """XA testbench DUT 端口写死 [2*W-2:0]（31bit, 与项目 bit31-mask 口径一致）。booth 输出是
    2*W bit（[31:0]）会撞 VCS inout 宽度不匹配。这里给 >31bit 的 MUL 套一层 31bit 顶层 wrapper
    丢掉最高位（DC 会删掉只喂 bit31 的逻辑）→ 与 AND/Zhang 同口径；verilator MED 本就用 out[30:0]。
    AND（out[30:0]，fw=30）不触发，原样返回。"""
    m = re.search(r"output\s+wire\s+\[(\d+):0\]\s+out", src)
    am = re.search(r"input\s+wire\s+\[(\d+):0\]\s+a", src)
    if not (m and am):
        return src
    fw, aw = int(m.group(1)), int(am.group(1))
    if fw <= 30:                      # 已是 <=31bit（AND）→ 不动
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
    BASE = sys.argv[1]
    MAXW = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("REEVAL_WORKERS", "6"))
    jobs = []
    for d in sorted(glob.glob(f"{BASE}/k*/")):
        kk = os.path.basename(d.rstrip("/"))
        bi, mv = os.path.join(d, "best_info.json"), os.path.join(d, "MUL.v")
        if not (os.path.exists(bi) and os.path.exists(mv)):
            print("skip (no best_info/MUL.v):", kk); continue
        med = (json.load(open(bi)).get("measured_error") or {}).get("med")
        jobs.append((kk, med, wrap_to_31b(open(mv).read())))
    print(f"{BASE}: {len(jobs)} designs -> {[j[0] for j in jobs]}", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=MAXW) as ex:
        futs = {ex.submit(evaluate_single_routing, i, c, 16, 1.5): (kk, med) for i, (kk, med, c) in enumerate(jobs)}
        for fut in as_completed(futs):
            kk, med = futs[fut]; r = fut.result(); res[kk] = (med, r)
            print(f"{kk}: success={r.get('success')} area={r.get('area')} power_mw={r.get('power_mw')} delay={r.get('delay')}", flush=True)
    out = f"{BASE}/reeval_xa.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["design", "med", "area_dc", "power_xa_mw", "delay", "success"])
        for kk, _, _ in jobs:
            med, r = res[kk]; w.writerow([kk, med, r.get("area"), r.get("power_mw"), r.get("delay"), r.get("success")])
    print("saved ->", out)


if __name__ == "__main__":
    main()
