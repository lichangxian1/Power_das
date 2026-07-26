#!/usr/bin/env python3
"""ELEX2024 N-4 近似乘法器 golden 模型 + 精度评估(穷举8-bit / 采样16-bit)。
逻辑全部来自 el4_common(与 RTL 生成器同源, 防漂移)。"""
import random
from el4_common import approx_mul, trunc_const, regions

def metrics(N, design, sample=None, seed=1):
    c = trunc_const(N, design)
    Dmax = ((1 << N) - 1) ** 2
    if sample is None and N <= 8:
        pairs = ((a, b) for a in range(1 << N) for b in range(1 << N))
        total = (1 << N) * (1 << N)
    else:
        rng = random.Random(seed)
        n = sample or 2_000_000
        pairs = ((rng.randrange(1 << N), rng.randrange(1 << N)) for _ in range(n))
        total = n
    ae = 0; sw = 0; rs = 0.0; rn = 0; se = 0; mx = 0
    for (a, b) in pairs:
        ex = a * b; ap = approx_mul(a, b, N, design, c); e = ap - ex
        if e: sw += 1
        ae += abs(e); se += e; mx = max(mx, abs(e))
        if ex: rs += abs(e) / ex; rn += 1
    return dict(const=c, ER=sw / total * 100, NMED=(ae / total) / Dmax,
                MRED=rs / rn, MED=ae / total, bias=se / total, maxerr=mx, total=total)

def show(N, design, **kw):
    m = metrics(N, design, **kw)
    mode = "exhaustive" if (N <= 8 and kw.get("sample") is None) else f"sample {m['total']}"
    print(f"== {design.upper()}-{N}  const={m['const']}(0x{m['const']:x})  [{mode}] ==")
    print(f"   ER={m['ER']:.2f}%  NMED={m['NMED']*1e3:.3f}e-3  MRED={m['MRED']*1e2:.3f}e-2  "
          f"MED={m['MED']:.2f}  bias={m['bias']:+.2f}  maxerr={m['maxerr']}")
    return m

if __name__ == "__main__":
    print("paper Table III: MUL1-8 ER88.58 NMED0.722e-3 MRED0.568e-2 | "
          "MUL2-8 ER99.82 NMED5.884e-3 MRED8.175e-2\n")
    show(8, "mul1"); show(8, "mul2")
    print()
    show(16, "mul1", sample=2_000_000); show(16, "mul2", sample=2_000_000)
