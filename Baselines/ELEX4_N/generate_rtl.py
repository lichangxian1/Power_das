#!/usr/bin/env python3
"""生成 ELEX2024 N-4 近似乘法器的可综合 Verilog —— **结构性复刻**(2026-06-29 重写)。

与旧版(纯 satN4 饱和)不同: 直接 inline 论文 Fig 2/3 的真实 N-4 压缩器网表 + Table II 近似 4-2,
全部经由 el4_common 的 ROps(dual-mode) 生成, 与 golden(GOps) 走**同一份 cell 逻辑**,
故 golden==RTL 由构造保证(再用 verilator 穷举/采样复核)。

产出(rtl/):
  el4_cells.v   —— 占位(本版顶层自包含, 不再实例化外部 cell; 保留供旧流程 source 列表引用)
  mul1_<N>.v / mul2_<N>.v —— 自包含顶层(端口 a,b,p; p 为 2N 位精确位宽)
数据通路: P = trunc_const + Σ_k (列贡献 << k)
  · 截断列: 跳过(并入常数)。
  · MUL1 近似列: N-4 压缩器 -> 4 位, stage2 精确 -> 贡献 = 四位之和。
  · MUL2 近似列(仅列 N,N+1): N-4 压缩器 -> 4 位 -> 近似 4-2 -> 贡献 = 2C+S。
  · 精确列: 部分积比特精确求和。
"""
import os
from el4_common import ROps, regions, trunc_const, col_pps, n4_bits, apx42, height

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")

CELLS_STUB = """// el4_cells.v — 占位文件。
// 结构性复刻版顶层(mul1_*/mul2_*)已自包含(inline 真实压缩器网表), 不再实例化外部 cell。
// 保留此文件仅为兼容旧 source 列表 [el4_cells.v, mulX.v]。
"""


def emit_mul(N, design):
    name = f"{design}_{N}"
    W = 2 * N
    trunc, sat = regions(N, design)
    const = trunc_const(N, design)
    ops = ROps()

    pp_decls = []
    for i in range(N):
        for j in range(N):
            pp_decls.append(f"    wire pp_{i}_{j} = a[{j}] & b[{i}];")

    terms = [f"({const})"]
    for k in range(W - 1):
        if k in trunc:
            continue
        ppk = [f"pp_{i}_{j}" for (i, j) in col_pps(N, k)]   # 规范配对顺序(按行 i 升序)
        if k in sat:
            w = n4_bits(ops, ppk)                          # 真实 N-4 -> 4 位(emit 进 ops.lines)
            v = ops._w()
            if design == "mul1":
                ops.lines.append(f"    wire [2:0] {v} = {w[0]} + {w[1]} + {w[2]} + {w[3]};")
            else:
                C, S = apx42(ops, w)                        # stage2 近似 4-2: value = 2C+S
                ops.lines.append(f"    wire [2:0] {v} = ({C} * 2) + {S};")
            terms.append(f"({v} << {k})")
        else:
            if len(ppk) == 1:
                terms.append(f"({ppk[0]} << {k})")
            else:
                v = ops._w()
                cw = max(1, len(ppk).bit_length())
                ops.lines.append(f"    wire [{cw-1}:0] {v} = {' + '.join(ppk)};")
                terms.append(f"({v} << {k})")

    L = []
    L.append(f"// {name}.v — ELEX2024 N-4 近似乘法器 **结构性复刻** (N={N}). 自动生成, 勿手改。")
    L.append(f"// 截断列={sorted(trunc)} 常数={const}(0x{const:x}) 近似列(真实N-4)={sorted(sat)}")
    L.append(f"// golden==RTL 由 el4_common dual-mode(GOps/ROps) 同源保证。")
    L.append(f"module {name} (input [{N-1}:0] a, input [{N-1}:0] b, output [{W-1}:0] p);")
    L.extend(pp_decls)
    L.extend(ops.lines)
    L.append("    assign p = " + " + ".join(terms) + ";")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def emit_wrapper(N, design):
    """项目口径包装: MUL(clk,a,b,out[30:0]), 31 位掩码(丢 bit31), 与 evo/v2022 同口径。"""
    name = f"{design}_{N}"
    return (f"// MUL wrapper for ELEX2024 {name} (31-bit masked, drops bit31 — 与 evo/v2022 同口径)\n"
            f"module MUL(input wire clk, input wire [{N-1}:0] a, input wire [{N-1}:0] b, output wire [30:0] out);\n"
            f"  wire [{2*N-1}:0] p_full;\n"
            f"  {name} dut(.a(a), .b(b), .p(p_full));\n"
            f"  assign out = p_full[30:0];\n"
            f"endmodule\n")


def main():
    os.makedirs(RTL, exist_ok=True)
    with open(os.path.join(RTL, "el4_cells.v"), "w") as f:
        f.write(CELLS_STUB)
    print("generated rtl/el4_cells.v (stub)")
    for N in (8, 16):
        for design in ("mul1", "mul2"):
            name = f"{design}_{N}"
            with open(os.path.join(RTL, f"{name}.v"), "w") as f:
                f.write(emit_mul(N, design))
            trunc, sat = regions(N, design)
            print(f"generated rtl/{name}.v (const={trunc_const(N,design)} 0x{trunc_const(N,design):x}, "
                  f"sat={len(sat)} cols)")
    # 项目口径包装(16-bit, 供 verilator MED + 远端 DC)
    wdir = os.path.join(RTL, "wrappers")
    os.makedirs(wdir, exist_ok=True)
    for design in ("mul1", "mul2"):
        with open(os.path.join(wdir, f"MUL_{design}_16.v"), "w") as f:
            f.write(emit_wrapper(16, design) + "\n" + emit_mul(16, design))
    print("generated rtl/wrappers/MUL_mul{1,2}_16.v")


if __name__ == "__main__":
    main()
