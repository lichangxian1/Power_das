#!/usr/bin/env python3
"""纯 Dadda 压缩树 + 低位截断(常数补偿) baseline 生成器。
完全复用本项目 utils 生成器(同源/同流程),0 近似 cell。
顶层 module MUL(clk,a,b,out[30:0]) 直接过 verilator MED harness 与 DC。"""
import os, sys, random
sys.path.insert(0, "/home/lee/Power_das")
os.chdir("/home/lee/Power_das")
import numpy as np
random.seed(0); np.random.seed(0)
from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorGraph
from run_power_sweep import VerilogEmitter, generate_legal_random_routing

BIT, ENC = 16, "and"
OUT = "/home/lee/Baselines/trunc_dadda_baseline/rtl"
os.makedirs(OUT, exist_ok=True)

pp = get_initial_partial_product(BIT, ENC)
ct0 = CompressorTree.dadda(pp)
assignment = ct0.compressor_assignment_fused()
comp_graph = CompressorGraph(pp, assignment)
emitter = VerilogEmitter(comp_graph)
routing = generate_legal_random_routing(comp_graph)        # 固定 seed -> 所有 k 复用同一布线
routing_assignment = emitter.emit_assignment(routing)

def trunc_bits(k):
    """复现 _setup_truncation: C=round(E[Δ]),贪心用低列槽位的常数1位表示。"""
    ppi = [int(x) for x in pp]
    e = sum(0.25 * ppi[c] * (1 << c) for c in range(k))
    ctgt = int(round(e)); bits = {}; rem = ctgt
    for c in range(k - 1, -1, -1):
        w = 1 << c; m = min(ppi[c], rem // w)
        if m > 0: bits[c] = m; rem -= m * w
    return bits

KS = list(range(1, 26))
for k in KS:
    ct = CompressorTree.dadda(pp)        # 同一 Dadda 结构,只改截断
    ct.trunc_cols = k
    ct.trunc_bits = trunc_bits(k)
    mul = Mul(BIT, ENC, ct)
    path = f"{OUT}/MUL_k{k:02d}.v"
    mul.emit_verilog(path, assignment=routing_assignment)
    top = next(l for l in open(path) if l.strip().startswith("module"))
    print(f"k={k:2d}: {os.path.basename(path)}  trunc_bits={ct.trunc_bits}  top={top.strip()}")
print("DONE", len(KS), "RTL ->", OUT)
