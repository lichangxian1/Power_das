#!/usr/bin/env python3
"""Step-2 预算扫描驱动器：k 枚举 × fast 贪心求解 × verilator 门控 → DC 探针裁决。

每个 MRED 预算 b∈{1e-7..1e-1}：取最深的 3 个可行 k（纯截断 floor ≤ b，
floor 表来自 2026-07-09_mred_trunc_baseline 16M 实测），每个 k 以 warm240eg
该 k 的 best_info 结构/布线为底座，fast 贪心（upgrade=False）解 cell 包，
发射 RTL → verilator 16M 门控（超预算按 wae·2^col 摘 cell 重发）→ staging。
DC+XA 后每个预算点按实测面积选赢家（跨 k 排序不信代理，spike 已定罪）。

42 菜单 = selected_compressors42_native 精选 13 + 新增 comp42s_*（substd42）
+ comp42n_zero，经运行时生成的 sweep 库文件注入（approx42_max_types=None）。

用法（GPU 求解按约定跑远端 gpu0/gpu2）:
  python -m Appr_Comp.cellsolver.budget_sweep [--staging outputs/2026-07-12_budget_sweep]
      [--base_run outputs/2026-07-09_21_mred_warm240eg_np4] [--vectors 16000000]
  # DC+XA: python3 scripts/reeval_xa_glob_tmpbuild.py <staging> 10
  python -m Appr_Comp.cellsolver.budget_sweep --analyze <staging>
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
from utils.compressor_tree import CompressorTree  # noqa: E402
from utils.mul import Mul  # noqa: E402

from Appr_Comp.cellsolver import sim as S  # noqa: E402
from Appr_Comp.cellsolver.solver import GradientCellSolver  # noqa: E402
from Appr_Comp.cellsolver.demo_solve import (  # noqa: E402
    build_trainer, massage, emit_variant, verilator_measure,
)

# 纯截断 mred-C* floor（16M 实测,outputs/2026-07-09_mred_trunc_baseline）
FLOORS = {2: 5e-08, 4: 3e-07, 6: 1.7e-06, 8: 8.92e-06, 10: 4.095e-05,
          12: 0.00019085, 14: 0.00079652, 16: 0.00345191,
          18: 0.0118332, 20: 0.0394763}
BUDGETS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_CAND = 3          # 每预算取最深 3 个可行 k
# 已有同底座 exact DC 数据的 k（spike 07-11: 788.9/726.1/628.2）→ 不重复探针
EX_HAVE_DC = {10, 12, 14}


def candidates(budget):
    ks = sorted((k for k, f in FLOORS.items() if f <= budget), reverse=True)
    return ks[:N_CAND]


def make_menu42(work):
    """精选 13 + comp42s_* + comp42n_zero → sweep 专用 42 库文件（全字段）。"""
    sel = json.load(open(os.path.join(REPO, "Appr_Comp/selected_compressors42_native.json")))
    lib = json.load(open(os.path.join(REPO, "Appr_Comp/library42_native.json")))
    names = list(sel["cells"].keys())
    names += [n for n in lib["cells"]
              if n.startswith("comp42s_") and n not in names]
    if "comp42n_zero" not in names:
        names.append("comp42n_zero")
    cells = {}
    for n in names:
        cells[n] = dict(lib["cells"][n])
        cells[n].update(sel["cells"].get(n, {}))
    path = os.path.join(work, "sweep_lib42.json")
    json.dump({"meta": sel.get("meta") or {}, "cells": cells}, open(path, "w"))
    return path, names


def build_exp(k, budget, work, menu42):
    """同 demo_solve.build_trainer，但 42 菜单换 sweep 库（max_types=None 不截断）。"""
    from omegaconf import OmegaConf
    from trainer.arith_das import CompressorRouting
    from utils.compressor_tree import get_initial_partial_product
    import Appr_Comp.cellsolver.demo_solve as D
    cfg = OmegaConf.to_container(OmegaConf.load(D.CONFIG), resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(work, "logs"),
        "build_dir": os.path.join(work, "build"),
        "experiment_prefix": "budget_sweep",
        "trunc_cols": int(k),
        "num_episodes": 1, "num_samples": 1, "n_processing": 1,
        "outer_cell_search": True,
        "use_ct42": True,
        "approx42_library_path": menu42,
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
    raise FileNotFoundError(f"{base_run} 无 k{k:02d} best_info")


def solve_one(budget, k, base_run, work, cache, device, est=None, log=print):
    menu42, menu_names = make_menu42(work)
    exp = build_exp(k, budget, work, menu42)
    bi = massage(json.load(open(base_info(base_run, k))))
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
                                cache_dir=cache, est=est)
    floor = solver.gate_mred({})
    log(f"[b={budget:g} k={k}] slots={len(solver.space.slots)} "
        f"T42={len(exp.type_table_42)} floor={floor:.3e} util0={floor/budget:.0%}")
    if floor > budget:
        log(f"[b={budget:g} k={k}] floor 超预算，跳过")
        return None
    t0 = time.time()
    cfg = solver.greedy_add(log=lambda *a, **kw: None, upgrade=False)  # fast 模式
    gm = solver.gate_mred(cfg)
    from collections import Counter
    used = Counter(f"T{ {0:'32',1:'22',4:'42'}[t] }:"
                   + solver.space.tables[t][kk]["name"]
                   for _n, (t, kk) in cfg.items())
    log(f"[b={budget:g} k={k}] fast greedy n={len(cfg)} mred={gm:.3e} "
        f"util={gm/budget:.0%} saving={solver.area_saving(cfg):.2f} "
        f"({time.time()-t0:.0f}s) 用型: {dict(used)}")
    return dict(exp=exp, bi=bi, cfg=cfg, solver=solver, floor=floor, gm=gm,
                est=solver.est)


def gate_and_stage(budget, k, r, d, vectors, log=print, max_repairs=8):
    """发射 → verilator 16M 门控 → 超预算摘 wae·2^col 最大 cell 重发。"""
    solver, exp, bi = r["solver"], r["exp"], r["bi"]
    cfg = dict(r["cfg"])
    colmap = {n: c for n, _t, c in solver.space.slots}
    for it in range(max_repairs + 1):
        cts = {str(n): [t, kk] for n, (t, kk) in cfg.items()}
        rtl = emit_variant(exp, bi, cts, d)
        me = verilator_measure(rtl, os.path.join(d, f"vbuild{it}"), vectors)
        if me["mred"] <= budget or not cfg:
            return cfg, me, it
        worst = max(cfg, key=lambda n: solver.space.wae_of(*cfg[n])
                    * (2 ** colmap.get(n, 0)))
        log(f"  [gate] mred={me['mred']:.3e} > b，摘 node {worst} "
            f"(repair {it + 1}/{max_repairs})")
        cfg.pop(worst)
    return cfg, me, max_repairs


def stage_meta(d, budget, k, variant, me, cfg, solver, repairs, floor):
    cts = {str(n): [t, kk] for n, (t, kk) in cfg.items()}
    json.dump({
        "measured_error": me, "cell_types": cts,
        "sweep": {"budget": budget, "k": k, "variant": variant,
                  "n_cells": len(cfg), "repairs": repairs, "floor": floor,
                  "proxy_saving": solver.area_saving(cfg) if cfg else 0.0,
                  "menu42": "native13+substd8+zero"},
    }, open(os.path.join(d, "best_info.json"), "w"))


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
    print(f"{'budget':>9}{'k':>4}{'var':>4} | {'n':>3}{'mred实测':>11}"
          f"{'DC面积':>9}{'XA功耗':>9}")
    per_b = {}
    for d, m in sorted(metas.items(), key=lambda x: (x[1].get("budget", 0),
                                                     -x[1].get("k", 0))):
        if not m or d not in rows:
            continue
        a = float(rows[d]["area_dc"]); p = float(rows[d]["power_xa_mw"])
        mr = (json.load(open(os.path.join(staging, d, "best_info.json")))
              ["measured_error"]["mred"])
        print(f"{m['budget']:>9.0e}{m['k']:>4}{m['variant']:>4} | "
              f"{m.get('n_cells', 0):>3}{mr:>11.3e}{a:>9.1f}{p:>9.4f}")
        if m["variant"] == "gr":
            per_b.setdefault(m["budget"], []).append((a, m["k"], p, d))
    print("\n=== 每预算点赢家（DC 面积裁决）===")
    for b, items in sorted(per_b.items()):
        a, k, p, d = min(items)
        alts = ", ".join(f"k{kk}:{aa:.1f}" for aa, kk, _p, _d in sorted(items))
        print(f"b={b:.0e}: 赢家 k={k} area={a:.1f} power={p:.4f}mW  ({alts})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="outputs/2026-07-12_budget_sweep")
    ap.add_argument("--base_run",
                    default="outputs/2026-07-09_21_mred_warm240eg_np4")
    ap.add_argument("--vectors", type=int, default=16_000_000)
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(os.environ.get("SCRATCH", "/tmp"), "budget_sweep")
    cache = os.path.join(root, "cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)
    plan = {b: candidates(b) for b in BUDGETS}
    print(f"[sweep] device={device} staging={args.staging}")
    print("[sweep] 候选表:", {f"{b:g}": ks for b, ks in plan.items()})

    est = None            # 16M 池/分层跨 (b,k) 复用
    ex_done = set(EX_HAVE_DC)
    for budget in BUDGETS:
        for k in plan[budget]:
            tag_b = f"b{budget:.0e}".replace("-", "m")
            work = os.path.join(root, f"{tag_b}_k{k:02d}")
            os.makedirs(work, exist_ok=True)
            try:
                r = solve_one(budget, k, args.base_run, work, cache, device,
                              est=est)
                if r is None:
                    continue
                est = r["est"]
                d = os.path.join(args.staging, f"k{k:02d}_{tag_b}_gr")
                cfg, me, repairs = gate_and_stage(budget, k, r, d,
                                                  args.vectors)
                stage_meta(d, budget, k, "gr", me, cfg, r["solver"],
                           repairs, r["floor"])
                print(f"  staged {os.path.basename(d)} n={len(cfg)} "
                      f"mred={me['mred']:.3e} repairs={repairs}")
                if k not in ex_done:      # 同底座 exact 参照（每 k 一次）
                    ex_done.add(k)
                    dx = os.path.join(args.staging, f"k{k:02d}_ex")
                    rtl = emit_variant(r["exp"], r["bi"], {}, dx)
                    mex = verilator_measure(rtl, os.path.join(dx, "vbuild"),
                                            args.vectors)
                    stage_meta(dx, budget, k, "ex", mex, {}, r["solver"],
                               0, r["floor"])
                    print(f"  staged {os.path.basename(dx)} "
                          f"mred={mex['mred']:.3e}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[b={budget:g} k={k}] FAIL: {e}")
    print(f"\n[sweep] 完成 → 本地跑: python3 scripts/reeval_xa_glob_tmpbuild.py "
          f"{args.staging} 10 && python -m Appr_Comp.cellsolver.budget_sweep "
          f"--analyze {args.staging}")


if __name__ == "__main__":
    main()
