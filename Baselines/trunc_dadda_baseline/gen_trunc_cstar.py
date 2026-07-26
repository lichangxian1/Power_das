#!/usr/bin/env python3
"""纯 Dadda + 截断 baseline 的 C* 版:常数从 round(E[Δ])(MED最优) 换成 argmin E[|C−Δ|/p](MRED最优)。
C* 计算与 trainer/arith_das.py _setup_truncation 的 mred 分支逐字同源(全宽MC seed1 1M, grid81+refine41)。
生成 RTL 后直接用同一 harness(verilate/mul_err_wrap.cpp, 16M 向量)实测 med/bias/mred。
输出: rtl_cstar/MUL_cstar_kXX.v + error_trunc_cstar.csv"""
import os, sys, random, shutil, subprocess
sys.path.insert(0, "/home/lee/Power_das")
os.chdir("/home/lee/Power_das")
import numpy as np
random.seed(0); np.random.seed(0)
from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorGraph
from run_power_sweep import VerilogEmitter, generate_legal_random_routing

BIT, ENC = 16, "and"
BASE = "/home/lee/Baselines/trunc_dadda_baseline"
OUT = f"{BASE}/rtl_cstar"
os.makedirs(OUT, exist_ok=True)

pp = get_initial_partial_product(BIT, ENC)
ct0 = CompressorTree.dadda(pp)
assignment = ct0.compressor_assignment_fused()
comp_graph = CompressorGraph(pp, assignment)
emitter = VerilogEmitter(comp_graph)
routing = generate_legal_random_routing(comp_graph)        # 同 gen_trunc.py: 固定 seed 同一布线
routing_assignment = emitter.emit_assignment(routing)

def cstar(k):
    """与 _setup_truncation mred 分支同源: 全宽 MC argmin E[|C−Δ|/p]。"""
    ppi = [int(x) for x in pp]
    e_delta = sum(0.25 * ppi[c] * (1 << c) for c in range(k))
    c_target = int(round(e_delta))
    if c_target <= 0:
        return 0
    rng_f = np.random.default_rng(1)
    W = BIT; Nf = 1_000_000
    af = rng_f.integers(0, 1 << W, size=Nf, dtype=np.int64)
    bf = rng_f.integers(0, 1 << W, size=Nf, dtype=np.int64)
    dl = np.zeros(Nf, dtype=np.int64)
    for i in range(min(k, W)):
        ai = (af >> i) & 1
        for j in range(min(k - i, W)):
            dl += (ai & ((bf >> j) & 1)) << (i + j)
    pm = af * bf; nz = pm > 0
    pw = pm[nz].astype(np.float64); dw = dl[nz].astype(np.float64)
    def _mred_of(cc): return float(np.mean(np.abs(cc - dw) / pw))
    grid = np.linspace(0.0, 1.2 * c_target, 81)
    i0 = int(np.argmin([_mred_of(cc) for cc in grid]))
    step = grid[1] - grid[0]
    fine = np.linspace(max(0.0, grid[i0] - step), grid[i0] + step, 41)
    vals = [_mred_of(cc) for cc in fine]
    return int(round(fine[int(np.argmin(vals))]))

def greedy_bits(k, c):
    ppi = [int(x) for x in pp]
    bits, rem = {}, c
    for col in range(k - 1, -1, -1):
        w = 1 << col; m = min(ppi[col], rem // w)
        if m > 0: bits[col] = m; rem -= m * w
    return bits, c - rem

def measure(rtl_abs, tag, n_vectors=16_000_000):
    """同 trainer._measure_error_verilator 流程(本地串行, -j 4)。"""
    harness = "/home/lee/Power_das/verilate/mul_err_wrap.cpp"
    verr = f"{OUT}/verr_{tag}"
    shutil.rmtree(verr, ignore_errors=True); os.makedirs(verr, exist_ok=True)
    obj = os.path.join(verr, "obj_dir"); exe = os.path.join(obj, "mul_err")
    bcmd = ["verilator", "--cc", "--exe", "--build", "-j", "4", "-O3",
            "-Wno-fatal", "--top-module", "MUL", "--Mdir", obj,
            rtl_abs, harness, "-o", "mul_err"]
    b = subprocess.run(bcmd, cwd=verr, capture_output=True, text=True, timeout=300)
    if b.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(f"build fail rc={b.returncode}: {b.stderr[-300:]}")
    r = subprocess.run([exe, str(n_vectors)], cwd=verr, capture_output=True, text=True, timeout=600)
    for line in r.stdout.strip().splitlines():
        p = line.split(",")
        if p[0] == "masked":
            shutil.rmtree(verr, ignore_errors=True)
            return float(p[1]), float(p[2]), (float(p[6]) if len(p) > 6 else None)
    raise RuntimeError(f"no masked line: {r.stdout[-200:]}")

KS = [2, 4, 6, 8, 10, 12, 14, 16]
rows = ["design,cstar,med,bias,mred"]
for k in KS:
    c = cstar(k)
    bits, c_act = greedy_bits(k, c)
    ct = CompressorTree.dadda(pp)
    ct.trunc_cols = k
    ct.trunc_bits = bits
    mul = Mul(BIT, ENC, ct)
    path = f"{OUT}/MUL_cstar_k{k:02d}.v"
    mul.emit_verilog(path, assignment=routing_assignment)
    med, bias, mred = measure(os.path.abspath(path), f"k{k:02d}")
    row = f"k{k:02d},{c_act},{med:.6f},{bias:.6f},{mred:.8f}"
    rows.append(row)
    print(f"k={k:2d} C*={c_act:6d}  med={med:12.1f} bias={bias:+12.1f} mred={mred*100:.4f}%", flush=True)
with open(f"{BASE}/error_trunc_cstar.csv", "w") as f:
    f.write("\n".join(rows) + "\n")
print("WROTE", f"{BASE}/error_trunc_cstar.csv")
