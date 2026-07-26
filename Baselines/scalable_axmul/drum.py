#!/usr/bin/env python3
"""DRUM — Dynamic Range Unbiased Multiplier (Hashemi, Bahar, Reda, ICCAD 2015).

Scalable family, knob k = number of significant bits kept per operand.
Each operand is rounded to its k leading bits (from the leading one), with the lowest kept
bit forced to 1 (the unbiasing trick); the two k-bit mantissas are multiplied (k×k) and the
result is shifted back. k=16 -> exact. Lower k -> smaller k×k multiplier, higher error.

Golden and RTL are the same computation (LOD + extract + k×k mult + shift), so golden==RTL.
Top module is the project 口径: MUL(clk,a,b,out[30:0]) (31-bit masked, combinational).
"""
N = 16


def drum_operand(x, k):
    """Return (mantissa, shift) for operand x at precision k."""
    if x == 0:
        return 0, 0
    l = x.bit_length() - 1
    if l < k:                       # already fits in k bits -> exact
        return x, 0
    s = l - (k - 1)
    return (x >> s) | 1, s          # top k bits, LSB forced to 1 (unbias)


def golden(a, b, k):
    ma, sa = drum_operand(a, k)
    mb, sb = drum_operand(b, k)
    return (ma * mb) << (sa + sb)


def emit(k):
    return f"""// DRUM (Hashemi ICCAD'15) {N}x{N} unsigned, k={k}  (auto-generated, behavioural).
module MUL(input wire clk, input wire [{N-1}:0] a, input wire [{N-1}:0] b,
           output wire [30:0] out);
    localparam K = {k};
    function [4:0] lod(input [{N-1}:0] x);
        integer i; begin lod = 5'd0;
            for (i = 0; i < {N}; i = i + 1) if (x[i]) lod = i[4:0]; end
    endfunction
    wire [4:0] la = lod(a), lb = lod(b);
    wire ta = (la >= K), tb = (lb >= K);                 // operand doesn't fit in K bits
    wire [4:0] sa = ta ? (la - (K-1)) : 5'd0;
    wire [4:0] sb = tb ? (lb - (K-1)) : 5'd0;
    wire [{N-1}:0] sha = a >> sa, shb = b >> sb;
    wire [K-1:0] ma = sha[K-1:0] | (ta ? {{{{(K-1){{1'b0}}}}, 1'b1}} : {{K{{1'b0}}}});
    wire [K-1:0] mb = shb[K-1:0] | (tb ? {{{{(K-1){{1'b0}}}}, 1'b1}} : {{K{{1'b0}}}});
    wire [2*K-1:0] pp = ma * mb;
    wire [31:0] full = ({{{{(32-2*K){{1'b0}}}}, pp}}) << (sa + sb);
    assign out = full[30:0];
endmodule
"""


SWEEP = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _selftest():
    import random
    rng = random.Random(0)
    for k in SWEEP:
        ae = se = 0; n = 200000; mx = 0
        for _ in range(n):
            a = rng.randrange(1 << N); b = rng.randrange(1 << N)
            e = golden(a, b, k) - a * b
            ae += abs(e); se += e; mx = max(mx, abs(e))
        D = (1 << (2 * N))
        print(f"  k={k:2d}: NMED={ae/n/D:.3e}  MRED~{'-':>4}  bias/MED={se/ae if ae else 0:+.3f}  "
              f"MED={ae/n:.1f}  maxrel~{mx/((1<<(2*N))-1):.4f}")
    # exactness at k=16
    assert all(golden(a, b, 16) == a * b for a in range(0, 1 << N, 997) for b in range(0, 1 << N, 1009))
    print("  k=16 exact: OK  (DRUM is near-unbiased: bias/MED small)")


if __name__ == "__main__":
    print("DRUM self-test (16-bit, 200k random):")
    _selftest()
