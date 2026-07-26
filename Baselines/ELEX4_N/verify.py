#!/usr/bin/env python3
"""golden==RTL 等价验证: el4_common.approx_mul (golden) vs 生成的 RTL (verilator)。
8-bit 穷举 65536; 16-bit 随机+角点。比对全 2N 位精确积(非掩码)。"""
import os, random, subprocess, sys
from el4_common import approx_mul, trunc_const

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")


def vectors(N):
    if N == 8:
        return [(a, b) for a in range(256) for b in range(256)]
    rng = random.Random(7)
    P = set((x, y) for x in (0, 1, 2, 3, 255, 256, 65535, 65534, 32768, 32767)
            for y in (0, 1, 2, 3, 255, 256, 65535, 65534, 32768, 32767))
    while len(P) < 300000:
        P.add((rng.randrange(1 << 16), rng.randrange(1 << 16)))
    return list(P)


def verify(design, N):
    name = f"{design}_{N}"
    src = os.path.join(RTL, f"{name}.v")
    const = trunc_const(N, design)
    vec = os.path.join(HERE, f"_vec_{name}.txt")
    with open(vec, "w") as f:
        for a, b in vectors(N):
            f.write(f"{a} {b} {approx_mul(a, b, N, design, const)}\n")
    obj = os.path.join(HERE, f"_obj_{name}")
    subprocess.run(["rm", "-rf", obj])
    bc = subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-j", "2", "-O3", "-Wno-fatal",
         "--top-module", name, "--Mdir", obj,
         "-CFLAGS", f"-O1 -DDUT_HEADER=V{name}.h -DDUT_CLASS=V{name} -DVEC_FILE={vec}",
         src, os.path.join(HERE, "tb.cpp"), "-o", "sim"],
        capture_output=True, text=True)
    if bc.returncode != 0:
        print(f"[{name}] BUILD FAIL:\n{bc.stderr[-800:]}"); return False
    r = subprocess.run([os.path.join(obj, "sim")], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    line = out[-1] if out else "(no output)"
    ok = "PASS" in line
    print(f"[{name}] {line}")
    subprocess.run(["rm", "-rf", obj]); os.remove(vec)
    return ok


if __name__ == "__main__":
    targets = [("mul1", 8), ("mul2", 8), ("mul1", 16), ("mul2", 16)]
    allok = all(verify(d, n) for d, n in targets)
    print("VERIFY:", "ALL PASS" if allok else "FAILURES")
    sys.exit(0 if allok else 1)
