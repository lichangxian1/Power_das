#!/usr/bin/env python3
"""Step-2b 多目标预算扫描：每个 MRED 误差档给一条 面积×功耗 帕累托前沿。

单目标版(budget_sweep)每档只留面积最小点;此版在**松预算档**额外扫"锥体清扫偏好"
pref_const∈{1(面积最优), N(功耗偏好)} × 最深 2 k,发射候选后 verilator 16M 门控,
DC+XA 揭示真实 (area,power),每档保留**非支配集**=面积↔功耗前沿。功耗贪心打不了
分(Q1),旋钮=const_kind cell 偏好,真实功耗差由 DC+XA 裁决。误差只约束 MRED。

菜单固定用统一 substd 库 selected_compressors_all_substd.json(22/32/42 全 PPA<std)。

用法(求解按约定跑远端 gpu0/gpu2):
  python -m Appr_Comp.cellsolver.pareto_sweep [--staging outputs/2026-07-12_pareto_sweep]
  # DC+XA: python3 scripts/reeval_xa_glob_tmpbuild.py <staging> 10
  python -m Appr_Comp.cellsolver.pareto_sweep --analyze <staging>
"""
import argparse
import copy
import csv
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

from trainer.arith_das import CompressorGraph  # noqa: E402
from utils.compressor_tree import CompressorTree, get_initial_partial_product  # noqa: E402
from utils.mul import Mul  # noqa: E402

from Appr_Comp.cellsolver import sim as S  # noqa: E402
from Appr_Comp.cellsolver.solver import GradientCellSolver  # noqa: E402
from Appr_Comp.cellsolver.demo_solve import (  # noqa: E402
    massage, emit_variant, verilator_measure, CONFIG,
)

MENU = "Appr_Comp/selected_compressors_all_substd.json"
FLOORS = {2: 5e-08, 4: 3e-07, 6: 1.7e-06, 8: 8.92e-06, 10: 4.095e-05,
          12: 0.00019085, 14: 0.00079652, 16: 0.00345191,
          18: 0.0118332, 20: 0.0394763}
# (预算, 候选k列表, 该档变体)：tight=单点面积最优；loose=2k×2pref 面积功耗前沿
TIGHT, LOOSE = "tight", "loose"
BANDS = [
    (1e-7, [2], TIGHT), (1e-6, [2], TIGHT), (1e-5, [8, 6], TIGHT),
    (1e-4, [10, 8], LOOSE), (1e-3, [12, 10], LOOSE),
    (1e-2, [14, 12], LOOSE), (1e-1, [20, 18], LOOSE),
]
PREF_HI = 12.0     # 功耗偏好档的锥体清扫权重


def build_exp(k, budget, work):
    from omegaconf import OmegaConf
    from trainer.arith_das import CompressorRouting
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
        "experiment_prefix": "pareto_sweep",
        "trunc_cols": int(k), "num_episodes": 1, "num_samples": 1,
        "n_processing": 1, "outer_cell_search": True, "use_ct42": True,
        "approx_lib_path": MENU,
        "approx42_library_path": "Appr_Comp/library42_native.json",
        "approx42_rtl_path": "Appr_Comp/rtl/comp42n_lib.v",
        "approx42_max_types": None,
    })
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = float(budget)
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()
    exp.initial_pp = get_initial_partial_product(
        exp.bit_width, exp.encode_type).astype(int)
    exp.approx_col_window = 6
    exp.approx_max_col = 30
    return exp


def base_info(base_run, k):
    for d in sorted(os.listdir(base_run)):
        p = os.path.join(base_run, d, "best_info.json")
        if d.startswith(f"k{k:02d}_b") and os.path.exists(p):
            return p
    raise FileNotFoundError(f"{base_run} 无 k{k:02d}")


def make_solver(k, budget, base_run, work, cache, device, est):
    exp = build_exp(k, budget, work)
    bi = massage(json.load(open(base_info(base_run, k))))
    comp_graph = CompressorGraph(exp.initial_pp, copy.deepcopy(bi["assignment"]),
                                 num_node_types=exp.num_node_types)
    ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                        bi["ct"].get("ct42"))
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    specs = S.parse_pp_specs(mul.emit_pp_encoder())
    tree = S.TreeSim(comp_graph, bi["connection"], specs, device)
    solver = GradientCellSolver(exp, tree, specs, budget, device=device,
                                cache_dir=cache, est=est)
    return exp, bi, solver


def gate_stage(exp, bi, solver, cfg, budget, d, vectors, log, max_repairs=8):
    colmap = {n: c for n, _t, c in solver.space.slots}
    cfg = dict(cfg)
    for it in range(max_repairs + 1):
        cts = {str(n): [t, kk] for n, (t, kk) in cfg.items()}
        rtl = emit_variant(exp, bi, cts, d)
        me = verilator_measure(rtl, os.path.join(d, f"vbuild{it}"), vectors)
        if me["mred"] <= budget or not cfg:
            return cfg, me, it
        worst = max(cfg, key=lambda n: solver.space.wae_of(*cfg[n])
                    * (2 ** colmap.get(n, 0)))
        cfg.pop(worst)
    return cfg, me, max_repairs


def run(staging, base_run, vectors):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(os.environ.get("SCRATCH", "/tmp"), "pareto_sweep")
    cache = os.path.join(root, "cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(staging, exist_ok=True)
    print(f"[pareto] device={device} staging={staging} menu={MENU}")
    est = None
    ex_done = set()
    for budget, ks, kind in BANDS:
        variants = ([(k, 1.0, "a"), (k, PREF_HI, "p")] for k in ks) if kind == LOOSE \
            else ([(k, 1.0, "a")] for k in ks)
        jobs = [v for sub in variants for v in sub]
        for k, pref, tag in jobs:
            tb = f"b{budget:.0e}".replace("-", "m")
            work = os.path.join(root, f"{tb}_k{k:02d}")
            os.makedirs(work, exist_ok=True)
            try:
                exp, bi, solver = make_solver(k, budget, base_run, work, cache,
                                              device, est)
                est = solver.est
                floor = solver.gate_mred({})
                if floor > budget:
                    print(f"[{tb} k{k}] floor {floor:.2e}>b 跳过")
                    continue
                t0 = time.time()
                cfg = solver.greedy_add(log=lambda *a, **kw: None,
                                        upgrade=False, pref_const=pref)
                d = os.path.join(staging, f"k{k:02d}_{tb}_gr{tag}")
                cfg, me, rep = gate_stage(exp, bi, solver, cfg, budget, d,
                                          vectors, print)
                nconst = sum(1 for n, (t, kk) in cfg.items()
                             if solver.space.const_kind.get((t, kk)))
                json.dump({"measured_error": me,
                           "cell_types": {str(n): [t, kk]
                                          for n, (t, kk) in cfg.items()},
                           "sweep": {"budget": budget, "k": k, "pref": pref,
                                     "variant": "gr" + tag, "kind": kind,
                                     "n_cells": len(cfg), "n_const": nconst,
                                     "repairs": rep,
                                     "proxy_saving": solver.area_saving(cfg)}},
                          open(os.path.join(d, "best_info.json"), "w"))
                print(f"[{tb} k{k} pref{pref:g}] n={len(cfg)} const={nconst} "
                      f"mred={me['mred']:.3e} rep={rep} ({time.time()-t0:.0f}s)")
                if k not in ex_done:
                    ex_done.add(k)
                    dx = os.path.join(staging, f"k{k:02d}_{tb}_ex")
                    rtl = emit_variant(exp, bi, {}, dx)
                    mex = verilator_measure(rtl, os.path.join(dx, "vbuild"),
                                            vectors)
                    json.dump({"measured_error": mex, "cell_types": {},
                               "sweep": {"budget": budget, "k": k,
                                         "variant": "ex", "kind": kind,
                                         "n_cells": 0}},
                              open(os.path.join(dx, "best_info.json"), "w"))
                    print(f"[{tb} k{k}] exact mred={mex['mred']:.3e}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[{tb} k{k} pref{pref:g}] FAIL: {e}")
    print(f"\n[pareto] 完成 → DC+XA: python3 scripts/reeval_xa_glob_tmpbuild.py "
          f"{staging} 10 && python -m Appr_Comp.cellsolver.pareto_sweep "
          f"--analyze {staging}")


def analyze(staging):
    rows = {}
    for r in csv.DictReader(open(os.path.join(staging, "reeval_xa.csv"))):
        if r.get("success") == "True":
            rows[r["design"]] = r
    metas = {}
    for d in os.listdir(staging):
        p = os.path.join(staging, d, "best_info.json")
        if os.path.exists(p):
            metas[d] = json.load(open(p)).get("sweep") or {}
    per_b = {}
    print(f"{'budget':>9}{'k':>4}{'var':>5}{'pref':>6} | {'n':>3}{'const':>6}"
          f"{'mred':>11}{'area':>8}{'power':>9}")
    for d, m in sorted(metas.items(), key=lambda x: (x[1].get("budget", 0),
                                                     -x[1].get("k", 0),
                                                     x[1].get("variant", ""))):
        if not m or d not in rows:
            continue
        a = float(rows[d]["area_dc"]); p = float(rows[d]["power_xa_mw"])
        mr = (json.load(open(os.path.join(staging, d, "best_info.json")))
              ["measured_error"]["mred"])
        print(f"{m['budget']:>9.0e}{m.get('k',0):>4}{m.get('variant',''):>5}"
              f"{m.get('pref',0):>6.0f} | {m.get('n_cells',0):>3}"
              f"{m.get('n_const',0):>6}{mr:>11.3e}{a:>8.1f}{p:>9.4f}")
        if m.get("variant", "").startswith("gr"):
            per_b.setdefault(m["budget"], []).append((a, p, m["k"],
                                                       m.get("pref", 1), d))
    print("\n=== 每档 面积×功耗 非支配前沿（Pareto，越左下越好）===")
    for b, items in sorted(per_b.items()):
        nd = [x for x in items if not any(
            (o[0] <= x[0] and o[1] <= x[1] and o != x) for o in items)]
        nd.sort()
        s = "  ".join(f"k{k}pref{pf:g}:{a:.1f}/{p:.4f}" for a, p, k, pf, _ in nd)
        dom = len(items) - len(nd)
        print(f"b={b:.0e}: 前沿 {len(nd)} 点 (被支配 {dom}): {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="outputs/2026-07-12_pareto_sweep")
    ap.add_argument("--base_run",
                    default="outputs/2026-07-09_21_mred_warm240eg_np4")
    ap.add_argument("--vectors", type=int, default=16_000_000)
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        run(args.staging, args.base_run, args.vectors)


if __name__ == "__main__":
    main()
