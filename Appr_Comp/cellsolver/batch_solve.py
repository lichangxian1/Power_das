#!/usr/bin/env python3
"""跨 k 点批量：在每个 k 的 GA 最优结构上,同预算重解 cell 包,汇总对比。

公平性：每个 k 用该 k 自身的 best_info 结构/布线为固定底座；可近似列范围沿用
GA 当时的 approx_max_col=16（stock config），三方比的纯粹是选 cell 的能力。
k≥16 时资格带 [k,16) 为空 → 三方皆 0 cell（配置限制，非 cell 无潜力），短路记零。

每 k：
  ③ 贪心（lazy greedy 实测打分 + 升级扫描）
  ④ 梯度（从贪心解温启动精调,仅验证"能否超过③"）
  终验：verilator 16M 对 exact/ga/greedy 三配置（sim 同步交叉核对,Δmed 必须=0）

产出：summary.csv + 打印汇总表。16M 向量池全 k 共享（cache_dir 固定）。

用法（arith_das env）:
  python -m Appr_Comp.cellsolver.batch_solve \
      [--rerun outputs/2026-07-09_06_mred_outer_rerun_np5] \
      [--ks 2,4,6,8,10,12,14] [--steps 150] [--final_vectors 16000000]
"""
import argparse
import csv
import os
import re
import sys
import time
import json
import copy

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
    build_trainer, massage, emit_variant, verilator_measure,
    cfg_from_cell_types, sim_measure_prefix,
)

DIR_RE = re.compile(r"k(\d+)_b([\d.eE+-]+)$")


def run_one(rerun, k, budget, work, cache, steps, val_vec, final_vec, log):
    run_dir = None
    for d in os.listdir(rerun):
        m = DIR_RE.match(d)
        if m and int(m.group(1)) == k:
            run_dir = os.path.join(rerun, d)
            break
    if run_dir is None:
        log(f"[k{k:02d}] 找不到 run 目录,跳过")
        return None
    bi = massage(json.load(open(os.path.join(run_dir, "best_info.json"))))
    ga_cts = bi.get("cell_types") or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp = build_trainer(k, budget, work)
    assignment = copy.deepcopy(bi["assignment"])
    comp_graph = CompressorGraph(exp.initial_pp, assignment,
                                 num_node_types=exp.num_node_types)
    ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                        bi["ct"].get("ct42"))
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    specs = S.parse_pp_specs(mul.emit_pp_encoder())
    tree = S.TreeSim(comp_graph, bi["connection"], specs, device)
    solver = GradientCellSolver(exp, tree, specs, budget, device=device,
                                pool_vectors=final_vec, cache_dir=cache)

    n_slots = len(solver.space.slots)
    log(f"\n[k{k:02d}] budget={budget:g} slots={n_slots} "
        f"GA_cells={len(cfg_from_cell_types(ga_cts))} device={device}")

    # 短路：无合法 slot（高 k 资格带为空）→ 三方皆纯截断
    if n_slots == 0:
        rtl = emit_variant(exp, bi, {}, os.path.join(work, "final_exact"))
        me = verilator_measure(rtl, os.path.join(work, "vfinal_exact"), final_vec)
        log(f"[k{k:02d}] 资格带空,三方=纯截断 floor mred={me['mred']:.3e}")
        row = dict(n_slots=0, budget=budget, floor_mred=me["mred"],
                   grad_beats_greedy=False)
        for tag in ("exact", "ga", "greedy"):
            row[f"{tag}_n"] = 0
            row[f"{tag}_mred"] = me["mred"]
            row[f"{tag}_med"] = me["med"]
            row[f"{tag}_util"] = me["mred"] / budget
            row[f"{tag}_save"] = 0.0
        return row

    # ③ 贪心
    t0 = time.time()
    cfg_greedy = solver.greedy_add(log=log)
    t_greedy = time.time() - t0
    # ④ 梯度（温启动精调,只验证能否超过③）
    t0 = time.time()
    cfg_grad, _hist = solver.solve(steps=steps, init_config=dict(cfg_greedy),
                                   log=lambda *a, **k: None)
    t_grad = time.time() - t0
    grad_beats = solver.area_saving(cfg_grad) > solver.area_saving(cfg_greedy) + 1e-6

    # 终验 verilator 16M + sim 交叉核对
    row = dict(n_slots=n_slots, budget=budget)
    variants = [("exact", {}), ("ga", ga_cts),
                ("greedy", {str(n): [t, kk] for n, (t, kk) in cfg_greedy.items()})]
    for tag, cts in variants:
        rtl = emit_variant(exp, bi, cts, os.path.join(work, f"final_{tag}"))
        me = verilator_measure(rtl, os.path.join(work, f"vfinal_{tag}"), final_vec)
        cc = cfg_from_cell_types(cts)
        ms = solver.measure_full(cc)   # 同 16M 全量,与 verilator 逐位可比
        dmed = abs(me["med"] - ms["med"])
        if dmed > 1e-4:
            log(f"[k{k:02d}] **{tag} 对拍失配 Δmed={dmed:.2e} "
                f"(sim {ms['med']:.4f} vs veri {me['med']:.4f})**")
        row[f"{tag}_n"] = len(cc)
        row[f"{tag}_mred"] = me["mred"]
        row[f"{tag}_med"] = me["med"]
        row[f"{tag}_util"] = me["mred"] / budget
        row[f"{tag}_save"] = solver.area_saving(cc)
    row["grad_beats_greedy"] = grad_beats
    row["floor_mred"] = row["exact_mred"]
    log(f"[k{k:02d}] floor={row['exact_mred']:.3e} | "
        f"GA n={row['ga_n']} save={row['ga_save']:.1f} util={row['ga_util']:.0%} | "
        f"greedy n={row['greedy_n']} save={row['greedy_save']:.1f} "
        f"util={row['greedy_util']:.0%} | ④>③={grad_beats} "
        f"(greedy {t_greedy:.0f}s grad {t_grad:.0f}s)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun",
                    default="outputs/2026-07-09_06_mred_outer_rerun_np5")
    ap.add_argument("--ks", default="2,4,6,8,10,12,14,16,18,20")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--val_vectors", type=int, default=2_000_000)
    ap.add_argument("--final_vectors", type=int, default=16_000_000)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = args.root or os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "cellsolver_batch")
    cache = os.path.join(root, "cache")
    os.makedirs(cache, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]
    print(f"[batch] rerun={args.rerun} ks={ks} steps={args.steps} root={root}")

    rows = []
    for k in ks:
        try:
            r = run_one(args.rerun, k, budget_of(args.rerun, k),
                        os.path.join(root, f"k{k:02d}"), cache,
                        args.steps, args.val_vectors, args.final_vectors, print)
            if r:
                r["k"] = k
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[k{k:02d}] FAIL: {e}")

    # 汇总
    cols = ["k", "budget", "n_slots", "floor_mred",
            "ga_n", "ga_save", "ga_util", "ga_med",
            "greedy_n", "greedy_save", "greedy_util", "greedy_med",
            "grad_beats_greedy"]
    csv_path = os.path.join(root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n" + "=" * 96)
    print("汇总（verilator 16M；面积节省=standalone cell 面积口径,µm²,未过 DC）")
    print("=" * 96)
    print(f"{'k':>3}{'预算':>11}{'GAcell':>7}{'GA省':>8}{'GA利用':>8}"
          f"{'贪心cell':>9}{'贪心省':>9}{'贪心利用':>9}{'倍数':>7}{'④>③':>6}"
          f"{'贪心MED':>11}")
    for r in rows:
        mult = (r["greedy_save"] / r["ga_save"]) if r.get("ga_save", 0) > 0 else \
            (float("inf") if r.get("greedy_save", 0) > 0 else 0)
        print(f"{r['k']:>3}{r['budget']:>11.1e}{r.get('ga_n',0):>7}"
              f"{r.get('ga_save',0):>8.1f}{r.get('ga_util',0):>7.0%}"
              f"{r.get('greedy_n',0):>9}{r.get('greedy_save',0):>9.1f}"
              f"{r.get('greedy_util',0):>8.0%}"
              f"{('∞' if mult==float('inf') else f'{mult:.1f}x'):>7}"
              f"{str(r.get('grad_beats_greedy','-')):>6}"
              f"{r.get('greedy_med',0):>11.0f}")
    print(f"\n[batch] summary -> {csv_path}")
    json.dump(rows, open(os.path.join(root, "summary.json"), "w"),
              indent=2, default=str)


def budget_of(rerun, k):
    for d in os.listdir(rerun):
        m = DIR_RE.match(d)
        if m and int(m.group(1)) == k:
            return float(m.group(2))
    raise ValueError(f"no dir for k={k}")


if __name__ == "__main__":
    main()
