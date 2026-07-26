#!/usr/bin/env python3
"""Emit synthesizable structural Verilog for the Zhang TCAS-II'23 multipliers.

Top module is the project 口径 form `MUL(clk, a, b, out[OB-1:0])` (clk unused, combinational):
AND partial products -> explicit gate-level reduction via zhang_core (proposed/Esposito/FA/HA
cells) -> final two-row CPA `out = const + Σ colval·2^k`. Because the netlist is the actual
cell structure, DC synthesises the paper's design (not a re-optimised tree).

For N=16 the 口径 uses OB=31 (out[30:0], masked), matching verilate/mul_err_wrap.cpp and DC.
For N=8 equivalence checks OB=16 (full product).
"""
from zhang_core import RtlOps, reduce_columns
from calibrate import base


def config(N, hybrid):
    """Calibrated against paper Table III/IV. Hybrid (Fig.2): Esposito @ level 1 + one
    proposed compressor @ level 2 on the tall inexact columns; the lowest inexact column
    (C1, next to truncation) stays proposed-greedy; AND gate(s) on the top column(s).
    8-bit finishes exact @ level 3; 16-bit re-applies Esposito @ level 3 (paper: "Esposito's
    compressors are used at the first and third levels")."""
    c = base(N)
    inex = sorted(c["inexact"])
    if not hybrid:
        c.update(hybrid=False, and_all=False, espo_cols=set(), and_cols=set(), post_l2="exact")
    elif N == 8:
        c.update(hybrid=True, and_all=False, espo_cols={5, 6, 7}, and_cols={7}, post_l2="exact")
    elif N == 16:
        c.update(hybrid=True, and_all=False, espo_cols=set(inex[1:]),
                 and_cols={inex[-1], inex[-2]}, post_l2="esposito")
    return c


def emit_mul(N, hybrid, ob=None):
    cfg = config(N, hybrid)
    OB = ob if ob else (2 * N - 1)          # N=16 -> 31 (masked 口径); N=8 -> 16 (full)
    WW = 2 * N + 2
    L = [f"// Zhang TCAS-II'23 {'ProposedH' if hybrid else 'Proposed'} {N}x{N} unsigned "
         f"approximate multiplier (auto-generated, structural).",
         f"// trunc={sorted(cfg['trunc'])} const={cfg['const']} inexact={sorted(cfg['inexact'])} "
         f"espo={sorted(cfg['espo_cols'])} and={sorted(cfg['and_cols'])}",
         f"module MUL(input wire clk, input wire [{N-1}:0] a, input wire [{N-1}:0] b,",
         f"           output wire [{OB-1}:0] out);"]
    # partial products
    cols = [[] for _ in range(2 * N)]
    for i in range(N):
        for j in range(N):
            w = f"p_{i}_{j}"
            L.append(f"    wire {w} = a[{j}] & b[{i}];")
            cols[i + j].append(w)
    for k in cfg["trunc"]:
        cols[k] = []
    # gate-level reduction (RtlOps collects the assign lines)
    ops = RtlOps()
    fin = reduce_columns(ops, cols, cfg)
    L += ops.lines
    # final CPA: out = const + Σ_k (colval_k << k)
    terms = [f"{WW}'d{cfg['const']}"]
    for k in range(len(fin)):
        bits = fin[k]
        if not bits:
            continue
        cv = f"cv_{k}"
        L.append(f"    wire [1:0] {cv} = " + " + ".join(bits) + ";")
        terms.append(f"({cv} << {k})")
    L.append(f"    wire [{WW-1}:0] _full = " + " + ".join(terms) + ";")
    L.append(f"    assign out = _full[{OB-1}:0];")
    L.append("endmodule")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    rtl = os.path.join(here, "rtl"); os.makedirs(rtl, exist_ok=True)
    for hyb, tag in [(False, "proposed"), (True, "proposedH")]:
        # 16-bit 口径 RTL (masked out[30:0])
        open(os.path.join(rtl, f"MUL_{tag}_16.v"), "w").write(emit_mul(16, hyb))
        # 8-bit full-product RTL (for exhaustive equivalence check)
        open(os.path.join(rtl, f"MUL_{tag}_8.v"), "w").write(emit_mul(8, hyb, ob=16))
        print(f"wrote rtl/MUL_{tag}_16.v  rtl/MUL_{tag}_8.v")
