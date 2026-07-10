#!/usr/bin/env python3
"""端到端 demo：仿真器对拍 verilator → 梯度求解 cell 包 → 终验对比 GA 包。

用法（arith_das env）:
  python -m Appr_Comp.cellsolver.demo_solve \
      --run outputs/2026-07-09_06_mred_outer_rerun_np5/k12_b2.800e-04 \
      --k 12 --budget 2.8e-4 [--steps 300] [--final_vectors 16000000]

阶段:
  A 对拍：exact 配置 + GA cell 配置，各发射 RTL → verilator 2M vs 仿真器 2M
    （同 xorshift 流 → med/bias 必须打印精度内一致，mred 容差 1e-8）
  B 求解：per-slot logits，~steps 步，MRED hinge + 面积项
  C 终验：solver 包发射 RTL → verilator final_vectors；三配置同口径对比表
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from omegaconf import OmegaConf  # noqa: E402

from trainer.arith_das import CompressorGraph, CompressorRouting  # noqa: E402
from utils.compressor_tree import CompressorTree, get_initial_partial_product  # noqa: E402
from utils.mul import Mul  # noqa: E402

from Appr_Comp.cellsolver import sim as S  # noqa: E402
from Appr_Comp.cellsolver.solver import GradientCellSolver  # noqa: E402

CONFIG = "configs/config_groups/mul_16_approx_error_obj.yaml"


def build_trainer(k, mred_budget, work):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(work, "logs"),
        "build_dir": os.path.join(work, "build"),
        "experiment_prefix": "cellsolver_demo",
        "trunc_cols": int(k),
        "num_episodes": 1, "num_samples": 1, "n_processing": 1,
        "outer_cell_search": True,
        # 与 07-09_06 rerun 同菜单：T32=10/T22=5/T42=13（原生 4:2 库）
        "use_ct42": True,
        "approx42_library_path": "Appr_Comp/library42_native.json",
        "approx42_rtl_path": "Appr_Comp/rtl/comp42n_lib.v",
    })
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = float(mred_budget)
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()
    exp.initial_pp = get_initial_partial_product(
        exp.bit_width, exp.encode_type).astype(int)
    return exp


def massage(bi):
    for kk in ("ct32", "ct22", "ct42"):
        if isinstance(bi.get("ct"), dict) and bi["ct"].get(kk) is not None:
            bi["ct"][kk] = np.array(bi["ct"][kk])
    if isinstance(bi.get("assignment"), list):
        bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                            for stage in bi["assignment"]]
    return bi


def emit_variant(exp, bi, cell_types, out_dir):
    exp.found_best_info = {**bi, "cell_types": cell_types}
    return exp.export_best_candidate(out_dir)


def verilator_measure(rtl, build, n):
    me = CompressorRouting._measure_error_verilator(rtl, build, n)
    if not me or me.get("mred") is None:
        raise RuntimeError(f"verilator measure failed: {me}")
    return me


def cfg_from_cell_types(cell_types):
    return {int(n): (int(tk[0]), int(tk[1]))
            for n, tk in (cell_types or {}).items() if int(tk[1]) != 0}


def sim_measure_prefix(solver, tree, specs, exp, config, n, chunk=500_000):
    """池前缀 n 个向量的精确测量（与 verilator 同流对拍用）。"""
    luts = solver.space.cell_luts_of(config)
    a, b = solver.pool_a[:n], solver.pool_b[:n]
    outs = []
    for i in range(0, n, chunk):
        pp = S.compute_pp_bits(specs, a[i:i + chunk], b[i:i + chunk],
                               exp.bit_width, solver.device)
        outs.append(tree.eval_exact(pp, luts))
    return S.error_stats(torch.cat(outs), a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/2026-07-09_06_mred_outer_rerun_np5/"
                    "k12_b2.800e-04")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--budget", type=float, default=2.8e-4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val_vectors", type=int, default=2_000_000)
    ap.add_argument("--final_vectors", type=int, default=16_000_000)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    work = args.work or os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "cellsolver_demo",
        os.path.basename(args.run.rstrip("/")))
    os.makedirs(work, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo] run={args.run} k={args.k} budget={args.budget:g} "
          f"device={device} work={work}")

    exp = build_trainer(args.k, args.budget, work)
    bi = massage(json.load(open(os.path.join(args.run, "best_info.json"))))
    ga_cell_types = bi.get("cell_types") or {}

    # ---- 仿真器构建（与 export 同源的图重建）----
    assignment = copy.deepcopy(bi["assignment"])
    comp_graph = CompressorGraph(exp.initial_pp, assignment,
                                 num_node_types=exp.num_node_types)
    ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                        bi["ct"].get("ct42"))
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    specs = S.parse_pp_specs(mul.emit_pp_encoder())
    n_pp = sum(int(h) for h in exp.initial_pp)
    assert len(specs) == n_pp, f"pp 解析不全: {len(specs)}/{n_pp}"
    tree = S.TreeSim(comp_graph, bi["connection"], specs, device)

    print("[demo] 生成/加载 16M 向量池 + 分层索引 ...")
    solver = GradientCellSolver(
        exp, tree, specs, args.budget, device=device,
        pool_vectors=args.final_vectors,
        cache_dir=os.path.join(work, "..", "cache"))
    print(f"[demo] 分层: S12(g<2^22)={len(solver.est.a12)} "
          f"S3_sub={len(solver.est.a3s)} w3={solver.est.w3:.1f} "
          f"n_rel={solver.est.n_rel}")
    print(f"[demo] slots={len(solver.space.slots)} "
          f"tables: T32={len(exp.type_table_32)} T22={len(exp.type_table_22)} "
          f"T42={len(exp.type_table_42 or [])}")

    # ---- A. 对拍 ----
    print("\n=== A. 仿真器 vs verilator 对拍（同向量流,", args.val_vectors, "vectors）===")
    for tag, cts in (("exact", {}), ("ga", ga_cell_types)):
        rtl = emit_variant(exp, bi, cts, os.path.join(work, f"rtl_{tag}"))
        t0 = time.time()
        mv = verilator_measure(rtl, os.path.join(work, f"vbuild_{tag}"),
                               args.val_vectors)
        t1 = time.time()
        ms = sim_measure_prefix(solver, tree, specs, exp,
                                cfg_from_cell_types(cts), args.val_vectors)
        t2 = time.time()
        dm = abs(ms["med"] - mv["med"])
        dmr = abs(ms["mred"] - mv["mred"])
        ok = dm < 1e-5 and dmr < 1e-7
        print(f"[{tag:5s}] verilator med={mv['med']:.6f} mred={mv['mred']:.8f} "
              f"({t1-t0:.0f}s) | sim med={ms['med']:.6f} mred={ms['mred']:.8f} "
              f"({t2-t1:.0f}s) | Δmed={dm:.2e} Δmred={dmr:.2e} "
              f"{'OK' if ok else '**MISMATCH**'}")
        if not ok:
            raise SystemExit("对拍失败——仿真器语义与 RTL 不一致,停止")

    # ---- B1. ③ 贪心加法基线（实测口径 lazy greedy）----
    print("\n=== B1. 贪心加法基线（分层实测打分）===")
    t0 = time.time()
    cfg_greedy = solver.greedy_add(log=print)
    print(f"[demo] 贪心耗时 {time.time()-t0:.0f}s")

    # ---- B2. ④ 梯度求解（hybrid：从贪心解温启动精调）----
    print(f"\n=== B2. 梯度求解（{args.steps} 步, 贪心温启动）===")
    t0 = time.time()
    cfg, hist = solver.solve(steps=args.steps, init_config=dict(cfg_greedy),
                             log=print)
    print(f"[demo] 求解耗时 {time.time()-t0:.0f}s, n_cells={len(cfg)}")
    if solver.area_saving(cfg) <= solver.area_saving(cfg_greedy):
        print("[demo] 梯度精调未超过贪心解(验收基线) —— ④相对③无增益")

    # ---- C. 终验对比 ----
    print(f"\n=== C. verilator 终验（{args.final_vectors} vectors）===")
    solver_cell_types = {str(n): [t, k] for n, (t, k) in cfg.items()}
    greedy_cell_types = {str(n): [t, k] for n, (t, k) in cfg_greedy.items()}
    rows = []
    for tag, cts in (("exact", {}), ("ga", ga_cell_types),
                     ("greedy", greedy_cell_types),
                     ("solver", solver_cell_types)):
        rtl = emit_variant(exp, bi, cts, os.path.join(work, f"final_{tag}"))
        mv = verilator_measure(rtl, os.path.join(work, f"vfinal_{tag}"),
                               args.final_vectors)
        cc = cfg_from_cell_types(cts)
        rows.append((tag, len(cc), mv["med"], mv["mred"],
                     mv["mred"] / args.budget, solver.area_saving(cc)))
    print(f"\n{'配置':<8}{'n_cells':>8}{'med':>14}{'mred':>14}"
          f"{'预算利用率':>10}{'Σcell面积节省':>14}")
    for tag, nc, med, mred, util, sv in rows:
        print(f"{tag:<8}{nc:>8}{med:>14.4f}{mred:>14.3e}{util:>9.1%}{sv:>13.2f}")
    ga_me = bi.get("measured_error") or {}
    if ga_me.get("mred") is not None:
        print(f"\n[ref] GA best_info 远端实测 mred={ga_me['mred']:.3e} "
              f"(本地重测应一致)")
    out = {"config": {str(n): list(tk) for n, tk in cfg.items()},
           "rows": rows, "hist": hist}
    json.dump(out, open(os.path.join(work, "solve_result.json"), "w"), indent=2)
    print(f"[demo] 结果 -> {work}/solve_result.json")


if __name__ == "__main__":
    main()
