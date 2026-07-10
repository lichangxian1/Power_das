#!/usr/bin/env python3
"""20+ 架构 GA vs greedy 公平对比：从 rerun 的 save_iter 检查点取多样结构,
每个结构在同一 slot 菜单上解 greedy,发射 exact/GA/greedy 三变体 RTL 到暂存目录,
供 reeval_xa_glob_tmpbuild.py 送真实 DC+XA。

检查点 = rerun(max_col=16 同配置)不同训练 iter 的 best_info → 不同布线/树结构 +
各自 GA cell 包。greedy 用 exp._is_approx_col_allowed 同资格判定,严格公平。

用法: python -m Appr_Comp.cellsolver.arch_compare <staging_dir>
输出: staging/k{kk}_i{it}_{exact|ga|greedy}/{MUL.v,best_info.json} + manifest.json
"""
import copy
import json
import os
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
from Appr_Comp.cellsolver.demo_solve import (  # noqa: E402
    build_trainer, massage, emit_variant, cfg_from_cell_types)

RER = "outputs/2026-07-09_06_mred_outer_rerun_np5"
BUDGET = {2: 1e-7, 4: 4e-7, 6: 2.5e-6, 8: 1.3e-5, 10: 6e-5, 12: 2.8e-4, 14: 1e-3}
DIRK = {2: "k02_b1.000e-07", 4: "k04_b4.000e-07", 6: "k06_b2.500e-06",
        8: "k08_b1.300e-05", 10: "k10_b6.000e-05", 12: "k12_b2.800e-04",
        14: "k14_b1.000e-03"}
# 每 k 取 3 个中段检查点（不同结构,均有 GA cell）。可用环境变量 CELLSOLVER_SPECS
# 覆盖,格式 "k:it,k:it,..." 例 "8:59,10:39"（便于错峰/只补跑未完成的）。
if os.environ.get("CELLSOLVER_SPECS"):
    SPECS = [(int(a), int(b)) for a, b in
             (s.split(":") for s in os.environ["CELLSOLVER_SPECS"].split(","))]
else:
    SPECS = [(k, it) for k in (2, 4, 6, 8, 10, 12, 14) for it in (39, 59, 99)]


def ckpt_path(k, it):
    return f"{RER}/{DIRK[k]}/logs/save_iter{it}/best_info.json"


def main():
    stage = sys.argv[1]
    cache = os.path.join(os.path.dirname(stage.rstrip("/")), "arch_cache")
    os.makedirs(stage, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = []
    for k, it in SPECS:
        p = ckpt_path(k, it)
        if not os.path.exists(p):
            print(f"[k{k:02d}_i{it}] 检查点缺失,跳过"); continue
        bi = massage(json.load(open(p)))
        ga_cts = bi.get("cell_types") or {}
        if not cfg_from_cell_types(ga_cts):
            print(f"[k{k:02d}_i{it}] GA 无 cell,跳过"); continue
        budget = BUDGET[k]
        work = os.path.join(stage, f"_work_k{k:02d}_i{it}")
        exp = build_trainer(k, budget, work)
        cg = CompressorGraph(exp.initial_pp, copy.deepcopy(bi["assignment"]),
                             num_node_types=exp.num_node_types)
        ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                            bi["ct"].get("ct42"))
        ct.trunc_cols = exp.trunc_cols
        ct.trunc_bits = exp._trunc_bits
        specs = S.parse_pp_specs(Mul(exp.bit_width, exp.encode_type, ct)
                                 .emit_pp_encoder())
        tree = S.TreeSim(cg, bi["connection"], specs, device)
        solver = GradientCellSolver(exp, tree, specs, budget, device=device,
                                    pool_vectors=16_000_000, cache_dir=cache)
        if not solver.space.slots:
            print(f"[k{k:02d}_i{it}] 无 slot,跳过"); continue
        cfg_greedy = solver.greedy_add(log=lambda *a, **k: None)
        tag = f"k{k:02d}_i{it}"
        variants = {
            "exact": {},
            "ga": ga_cts,
            "greedy": {str(n): [t, kk] for n, (t, kk) in cfg_greedy.items()},
        }
        for v, cts in variants.items():
            d = os.path.join(stage, f"{tag}_{v}")
            emit_variant(exp, bi, cts, d)
        ga_sv = solver.area_saving(cfg_from_cell_types(ga_cts))
        gr_sv = solver.area_saving(cfg_greedy)
        m = dict(k=k, it=it, budget=budget, n_slots=len(solver.space.slots),
                 ga_n=len(cfg_from_cell_types(ga_cts)), greedy_n=len(cfg_greedy),
                 ga_proxy=ga_sv, greedy_proxy=gr_sv,
                 ga_mred=solver.measure_full(cfg_from_cell_types(ga_cts))["mred"],
                 greedy_mred=solver.measure_full(cfg_greedy)["mred"])
        manifest.append(m)
        print(f"[{tag}] slots={m['n_slots']} GA n={m['ga_n']} proxy={ga_sv:.1f} | "
              f"greedy n={m['greedy_n']} proxy={gr_sv:.1f} "
              f"(greedy mred={m['greedy_mred']:.2e})", flush=True)
    json.dump(manifest, open(os.path.join(stage, "manifest.json"), "w"),
              indent=2, default=str)
    print(f"\n{len(manifest)} 架构就绪 -> {stage}")


if __name__ == "__main__":
    main()
