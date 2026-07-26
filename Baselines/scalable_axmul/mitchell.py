#!/usr/bin/env python3
"""Mitchell logarithmic multiplier (J. N. Mitchell, IRE Trans. 1962) + truncated-mantissa family.

log2(a·b) ≈ log2(a)+log2(b) with the piecewise-linear approx log2(1+f)≈f. With leading-one
positions ka,kb and fractional values fa,fb (bits below the leading one):
    cross = fa·2^kb + fb·2^ka ,  s = ka+kb
    P = 2^s + cross           if cross <  2^s   (log fraction < 1)
    P = 2·cross               if cross >= 2^s   (log fraction ≥ 1)
Always underestimates; the inherent log error floor is ~ -11%.

Scalable knob W = number of fractional mantissa bits kept (fa,fb truncated to the top W bits):
smaller W -> smaller fraction adders, higher error; large W -> the pure-Mitchell log floor.
Golden and RTL are the same computation (behavioural), so golden==RTL.
Top module: project 口径 MUL(clk,a,b,out[30:0]).
"""
N = 16


def mitch_operand(x, W):
    if x == 0:
        return 0, 0
    k = x.bit_length() - 1
    f = x - (1 << k)
    if k > W:                       # keep only the top W fractional bits
        f = (f >> (k - W)) << (k - W)
    return k, f


def golden(a, b, W):
    if a == 0 or b == 0:
        return 0
    ka, fa = mitch_operand(a, W)
    kb, fb = mitch_operand(b, W)
    s = ka + kb
    cross = (fa << kb) + (fb << ka)
    return ((1 << s) + cross) if cross < (1 << s) else (cross << 1)


def emit(W):
    return f"""// Mitchell log multiplier (IRE'62) {N}x{N} unsigned, W={W} mantissa bits (auto-gen, behavioural).
module MUL(input wire clk, input wire [{N-1}:0] a, input wire [{N-1}:0] b,
           output wire [30:0] out);
    localparam W = {W};
    function [4:0] lod(input [{N-1}:0] x);
        integer i; begin lod = 5'd0;
            for (i = 0; i < {N}; i = i + 1) if (x[i]) lod = i[4:0]; end
    endfunction
    wire [4:0] ka = lod(a), kb = lod(b);
    wire [{N-1}:0] far = a & ~({N}'d1 << ka);            // bits below the leading one
    wire [{N-1}:0] fbr = b & ~({N}'d1 << kb);
    wire [4:0] tta = (ka > W) ? (ka - W) : 5'd0;         // truncate mantissa to W bits
    wire [4:0] ttb = (kb > W) ? (kb - W) : 5'd0;
    wire [{N-1}:0] fa = (far >> tta) << tta;
    wire [{N-1}:0] fb = (fbr >> ttb) << ttb;
    wire [31:0] xsum = ({{16'd0, fa}} << kb) + ({{16'd0, fb}} << ka);   // 'cross' is a SV keyword
    wire [4:0] s = ka + kb;
    wire [31:0] base = 32'd1 << s;
    wire [32:0] res = (xsum < base) ? ({{1'b0, base}} + {{1'b0, xsum}}) : {{xsum, 1'b0}};
    assign out = (a == 0 || b == 0) ? 31'd0 : res[30:0];
endmodule
"""


SWEEP = [2, 3, 4, 5, 6, 8, 11]


def _selftest():
    import random
    rng = random.Random(0)
    for W in SWEEP + [15]:
        ae = se = 0; n = 200000; mx = 0
        for _ in range(n):
            a = rng.randrange(1, 1 << N); b = rng.randrange(1, 1 << N)
            e = golden(a, b, W) - a * b
            ae += abs(e); se += e; mx = max(mx, abs(e))
        D = 1 << (2 * N)
        print(f"  W={W:2d}: NMED={ae/n/D:.3e}  bias/MED={se/ae:+.3f}  MED={ae/n:.1f}  maxrel~{mx/((1<<32)-1):.4f}")
    print("  (W=15 ≈ pure Mitchell: NMED floor from the log approximation, error always negative)")


if __name__ == "__main__":
    print("Mitchell self-test (16-bit, 200k random):")
    _selftest()
