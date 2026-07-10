#!/usr/bin/env python3
"""独立梯度求解跨 k 扫描：填补 batch_solve 里缺的④梯度自身数字。

梯度 solver 不给贪心温启动、自己从零解 steps 步（对偶上升 + STE），离散化后
sim 16M 测（已证 =verilator）。与 summary.csv 里的贪心/GA 并排,给完整第三列。
用法: python -m Appr_Comp.cellsolver.grad_sweep [--ks 2,4,..] [--steps 300]
"""
import argparse
import copy
import csv
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from trainer.arith_das import CompressorGraph  # noqa: E402
from utils.compressor_tree import CompressorTree  # noqa: E402
from utils.mul import Mul  # noqa: E402
from Appr_Comp.cellsolver import sim as S  # noqa: E402
from Appr_Comp.cellsolver.solver import GradientCellSolver  # noqa: E402
from Appr_Comp.cellsolver.demo_solve import build_trainer, massage  # noqa: E402

DIR_RE = re.compile(r"k(\d+)_b([\d.eE+-]+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun",
                    default="outputs/2026-07-09_06_mred_outer_rerun_np5")
    ap.add_argument("--ks", default="2,4,6,8,10,12,14")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--final_vectors", type=int, default=16_000_000)
    ap.add_argument("--root",
                    default=os.path.join(os.environ.get("SCRATCH", "/tmp"),
                                         "cellsolver_batch"))
    args = ap.parse_args()
    cache = os.path.join(args.root, "cache")
    # 读回贪心/GA 汇总以并排
    summ = {}
    sp = os.path.join(args.root, "summary.csv")
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp)):
            summ[int(r["k"])] = r
    budg = {int(m.group(1)): float(m.group(2))
            for d in os.listdir(args.rerun) if (m := DIR_RE.match(d))}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for k in [int(x) for x in args.ks.split(",")]:
        d = [x for x in os.listdir(args.rerun)
             if DIR_RE.match(x) and int(DIR_RE.match(x).group(1)) == k][0]
        bi = massage(json.load(open(f"{args.rerun}/{d}/best_info.json")))
        exp = build_trainer(k, budg[k], os.path.join(args.root, f"grad_k{k:02d}"))
        cg = CompressorGraph(exp.initial_pp, copy.deepcopy(bi["assignment"]),
                             num_node_types=exp.num_node_types)
        ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                            bi["ct"].get("ct42"))
        ct.trunc_cols = exp.trunc_cols
        ct.trunc_bits = exp._trunc_bits
        specs = S.parse_pp_specs(Mul(exp.bit_width, exp.encode_type, ct)
                                 .emit_pp_encoder())
        tree = S.TreeSim(cg, bi["connection"], specs, device)
        solver = GradientCellSolver(exp, tree, specs, budg[k], device=device,
                                    pool_vectors=args.final_vectors,
                                    cache_dir=cache)
        cfg, _hist = solver.solve(steps=args.steps, log=lambda *a, **k: None)
        me = solver.measure_full(cfg)
        util = me["mred"] / budg[k]
        sv = solver.area_saving(cfg)
        gr = summ.get(k, {})
        row = dict(k=k, budget=budg[k], grad_n=len(cfg), grad_save=sv,
                   grad_util=util, grad_mred=me["mred"], grad_med=me["med"],
                   greedy_save=float(gr.get("greedy_save", 0)),
                   greedy_n=int(gr.get("greedy_n", 0)),
                   ga_save=float(gr.get("ga_save", 0)))
        rows.append(row)
        print(f"[k{k:02d}] 独立梯度 n={len(cfg)} save={sv:.1f} util={util:.0%} "
              f"mred={me['mred']:.3e} | 贪心 n={row['greedy_n']} "
              f"save={row['greedy_save']:.1f} | GA save={row['ga_save']:.1f}")

    print("\n" + "=" * 84)
    print("④独立梯度 vs ③贪心 vs GA（面积节省 µm², sim 16M=verilator）")
    print("=" * 84)
    print(f"{'k':>3}{'预算':>10}{'GA省':>8}{'梯度cell':>9}{'梯度省':>9}"
          f"{'梯度利用':>9}{'贪心cell':>9}{'贪心省':>9}{'梯度/贪心':>10}")
    for r in rows:
        frac = r["grad_save"] / r["greedy_save"] if r["greedy_save"] > 0 else 0
        print(f"{r['k']:>3}{r['budget']:>10.1e}{r['ga_save']:>8.1f}"
              f"{r['grad_n']:>9}{r['grad_save']:>9.1f}{r['grad_util']:>8.0%}"
              f"{r['greedy_n']:>9}{r['greedy_save']:>9.1f}{frac:>9.0%}")
    json.dump(rows, open(os.path.join(args.root, "grad_summary.json"), "w"),
              indent=2, default=str)
    print(f"\n-> {os.path.join(args.root, 'grad_summary.json')}")


if __name__ == "__main__":
    main()
