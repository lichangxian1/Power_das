#!/usr/bin/env python3
"""Step-0 spike：k 选择器的成本模型验证——同一预算下扫多个 k，
比较「加法代理排序」vs「整网表 DC 实测排序」是否一致。

每 (budget, k)：以 warm240eg 该 k 的 best_info 结构/布线为底座，
贪心求解 cell 包（现库，无 ZERO），发射 exact / greedy 两个变体。
代理总面积(k) = DC(exact_k) − Σ standalone cell 面积节省
实测总面积(k) = DC(greedy_k)
若两种口径给出的最优 k 不一致 → 求解器选 k 必须加 DC 探针（成本模型问题前置暴露）。

用法（arith_das env，先跑本脚本再对 staging 目录跑 reeval）:
  python -m Appr_Comp.cellsolver.spike_kselect [--staging outputs/2026-07-11_spike_kselect]
  python3 scripts/reeval_xa_glob_tmpbuild.py <staging> 6
  python -m Appr_Comp.cellsolver.spike_kselect --analyze <staging>
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

BASE_RUN = "outputs/2026-07-09_21_mred_warm240eg_np4"
# 每个预算取最深的 3 个可行 k（floor 见 mred_trunc_baseline）
SWEEP = [(2.8e-4, [8, 10, 12]), (1e-3, [10, 12, 14])]


def base_info(k):
    for d in os.listdir(BASE_RUN):
        if d.startswith(f"k{k:02d}_b") and \
                os.path.exists(os.path.join(BASE_RUN, d, "best_info.json")):
            return os.path.join(BASE_RUN, d, "best_info.json")
    raise FileNotFoundError(f"warm240eg 无 k{k:02d} best_info")


def solve_one(budget, k, work, cache, device, log=print):
    exp = build_trainer(k, budget, work)
    # 对齐 warm240eg 约定：窗口带随 k 走
    exp.approx_col_window = 6
    exp.approx_max_col = 30
    bi = massage(json.load(open(base_info(k))))
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
                                cache_dir=cache)
    floor = solver.gate_mred({})
    log(f"[b={budget:g} k={k}] slots={len(solver.space.slots)} "
        f"floor={floor:.3e} util0={floor/budget:.0%}")
    if floor > budget:
        log(f"[b={budget:g} k={k}] floor 超预算，跳过")
        return None
    t0 = time.time()
    cfg = solver.greedy_add(log=lambda *a, **kw: None)
    gm = solver.gate_mred(cfg)
    log(f"[b={budget:g} k={k}] greedy n={len(cfg)} mred={gm:.3e} "
        f"util={gm/budget:.0%} proxy_saving={solver.area_saving(cfg):.2f} "
        f"({time.time()-t0:.0f}s)")
    return dict(exp=exp, bi=bi, cfg=cfg, solver=solver, floor=floor, gm=gm)


def stage(budget, k, r, staging, vectors, log=print):
    tag_b = f"b{budget:g}".replace("-", "m").replace(".", "p")
    out = {}
    for variant, cts in (("ex", {}),
                         ("gr", {str(n): [t, kk] for n, (t, kk) in r["cfg"].items()})):
        d = os.path.join(staging, f"k{k:02d}_{tag_b}_{variant}")
        rtl = emit_variant(r["exp"], r["bi"], cts, d)
        me = verilator_measure(rtl, os.path.join(d, "vbuild"), vectors)
        json.dump({"measured_error": me, "cell_types": cts,
                   "spike": {"budget": budget, "k": k, "variant": variant,
                             "proxy_saving": r["solver"].area_saving(
                                 {int(n): (t[0], t[1]) for n, t in cts.items()})
                             if cts else 0.0}},
                  open(os.path.join(d, "best_info.json"), "w"))
        log(f"  staged {os.path.basename(d)} mred={me['mred']:.3e} med={me['med']:.0f}")
        out[variant] = d
    return out


def analyze(staging):
    rows = {}
    for r in csv.DictReader(open(os.path.join(staging, "reeval_xa.csv"))):
        if r.get("success") != "True":
            continue
        rows[r["design"]] = r
    metas = {}
    for d in os.listdir(staging):
        p = os.path.join(staging, d, "best_info.json")
        if os.path.exists(p):
            metas[d] = json.load(open(p)).get("spike") or {}
    print(f"{'k':>4}{'budget':>10} | {'DC(exact)':>10}{'代理省':>8}{'代理总':>9}"
          f"{'DC(greedy)':>11}{'实测省':>8}{'兑现率':>8}")
    verdict = {}
    for d, m in sorted(metas.items()):
        if m.get("variant") != "gr":
            continue
        k, b = m["k"], m["budget"]
        ex = d.replace("_gr", "_ex")
        if d not in rows or ex not in rows:
            continue
        a_gr, a_ex = float(rows[d]["area_dc"]), float(rows[ex]["area_dc"])
        proxy_sv = float(m["proxy_saving"])
        proxy_total = a_ex - proxy_sv
        real_sv = a_ex - a_gr
        cash = real_sv / proxy_sv if proxy_sv > 0 else float("nan")
        verdict.setdefault(b, []).append((k, proxy_total, a_gr))
        print(f"{k:>4}{b:>10.1e} | {a_ex:>10.1f}{proxy_sv:>8.1f}{proxy_total:>9.1f}"
              f"{a_gr:>11.1f}{real_sv:>8.1f}{cash:>8.0%}")
    for b, items in sorted(verdict.items()):
        by_proxy = min(items, key=lambda x: x[1])[0]
        by_real = min(items, key=lambda x: x[2])[0]
        ok = "一致 ✓" if by_proxy == by_real else "**不一致 ✗ → k 选择必须加 DC 探针**"
        print(f"\nbudget={b:g}: 代理选 k={by_proxy}, DC 实测选 k={by_real} → {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="outputs/2026-07-11_spike_kselect")
    ap.add_argument("--vectors", type=int, default=16_000_000)
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(os.environ.get("SCRATCH", "/tmp"), "spike_kselect")
    cache = os.path.join(root, "cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)
    print(f"[spike] device={device} staging={args.staging}")
    for budget, ks in SWEEP:
        for k in ks:
            work = os.path.join(root, f"b{budget:g}_k{k:02d}")
            try:
                r = solve_one(budget, k, work, cache, device)
                if r:
                    stage(budget, k, r, args.staging, args.vectors)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[b={budget:g} k={k}] FAIL: {e}")
    print(f"\n[spike] 求解+发射完成 → 跑: python3 scripts/reeval_xa_glob_tmpbuild.py "
          f"{args.staging} 6 && python -m Appr_Comp.cellsolver.spike_kselect "
          f"--analyze {args.staging}")


if __name__ == "__main__":
    main()
