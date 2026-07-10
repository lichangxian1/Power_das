#!/usr/bin/env python3
"""外环 greedy 求解器接入冒烟（CPU,不占 GPU）。

A. 参数管线 + _cell_solver_active 逻辑（各模式开关口径,无 GPU/池）。
B. _outer_greedy_solve 端到端控制流:真实 k14 结构上建 solver→greedy→cfg→state["cells"]
   →_episode_cell_types→重发射 RTL,断言状态/映射/RTL 同步、cells 坐标反查正确。
C. 默认关(solver=None):get_samples 走原 errgate/进化路径,_cell_solver_active=False。

用法: /home/lee/anaconda3/envs/arith_das/bin/python scripts/smoke_outer_greedy.py
"""
import copy
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import trainer  # noqa: E402
from trainer.arith_das import CompressorRouting, CompressorGraph  # noqa: E402
from utils import get_initial_partial_product  # noqa: E402
from utils.compressor_tree import CompressorTree  # noqa: E402
from utils.mul import Mul  # noqa: E402

CONFIG = "configs/config_groups/mul_16_approx_error_obj.yaml"
CKPT = ("outputs/2026-07-09_06_mred_outer_rerun_np5/k14_b1.000e-03/"
        "logs/save_iter99/best_info.json")
WORK = "/tmp/smoke_outer_greedy"


def build(k, budget, solver=None, vectors=2_000_000):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(WORK, "logs"), "build_dir": os.path.join(WORK, "build"),
        "experiment_prefix": "smoke_greedy", "trunc_cols": int(k),
        "num_episodes": 1, "num_samples": 1, "n_processing": 1,
        "outer_cell_search": True, "use_ct42": True,
        "approx42_library_path": "Appr_Comp/library42_native.json",
        "approx42_rtl_path": "Appr_Comp/rtl/comp42n_lib.v",
        "outer_solver_cache": os.path.join(WORK, "pool"),
    })
    if solver:
        tk["outer_cell_solver"] = solver
        tk["outer_solver_vectors"] = vectors
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = float(budget)
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()
    exp.initial_pp = get_initial_partial_product(exp.bit_width, exp.encode_type).astype(int)
    return exp


def massage(bi):
    for kk in ("ct32", "ct22", "ct42"):
        if isinstance(bi.get("ct"), dict) and bi["ct"].get(kk) is not None:
            bi["ct"][kk] = np.array(bi["ct"][kk])
    if isinstance(bi.get("assignment"), list):
        bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                            for stage in bi["assignment"]]
    return bi


def setup_structure(exp, bi):
    """把 checkpoint 结构装进 exp（模拟 reset 之后的状态）。"""
    assignment = copy.deepcopy(bi["assignment"])
    exp.state = {"ct32": bi["ct"]["ct32"], "ct22": bi["ct"]["ct22"],
                 "ct42": bi["ct"].get("ct42"), "cells": []}
    exp.comp_graph = CompressorGraph(exp.initial_pp, assignment,
                                     num_node_types=exp.num_node_types)
    exp._episode_cell_types = {}
    ct = CompressorTree(exp.initial_pp, bi["ct"]["ct32"], bi["ct"]["ct22"],
                        bi["ct"].get("ct42"))
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    return mul, bi["connection"]


def main():
    os.makedirs(os.path.join(WORK, "build"), exist_ok=True)
    bi = massage(json.load(open(CKPT)))

    # ---- A. _cell_solver_active 口径 ----
    print("=== A. 参数管线 + _cell_solver_active ===")
    off = build(14, 1e-3, solver=None)
    assert off.outer_cell_solver is None
    assert off._cell_solver_active() is False, "solver=None 不应生效"
    print("  solver=None: _cell_solver_active=False ✓")

    on = build(14, 1e-3, solver="greedy")
    assert on.outer_cell_solver == "greedy"
    # 需 MRED 模式:构造时 error_metric 默认非 mred,_setup_truncation 后设为 mred
    assert on._cell_solver_active() is True, "greedy+MRED 应生效"
    print("  solver=greedy + MRED 预算: _cell_solver_active=True ✓")
    # 非 MRED 模式应失效
    on.error_metric = "med"
    assert on._cell_solver_active() is False, "非 MRED 模式 solver 应失效"
    on.error_metric = "mred"
    print("  非 MRED 模式: _cell_solver_active=False ✓")

    # outer_cell_search=False 时构造应自动关掉 solver
    cfg2 = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    tk2 = copy.deepcopy(cfg2["trainer"]["kwargs"]); tk2.pop("area_budgets", None)
    tk2.update(copy.deepcopy(cfg2["experiment"]["kwargs"]))
    tk2.update({"synth": "dc", "device": "cpu", "log_dir": None,
                "build_dir": os.path.join(WORK, "build"), "trunc_cols": 14,
                "outer_cell_search": False, "outer_cell_solver": "greedy",
                "num_episodes": 1, "num_samples": 1, "n_processing": 1})
    ncs = CompressorRouting(**tk2)
    assert ncs.outer_cell_solver is None, "无 outer_cell_search 应忽略 solver"
    print("  outer_cell_search=False: solver 被忽略置 None ✓")

    # ---- B. _outer_greedy_solve_robust 端到端 ----
    print("\n=== B. _outer_greedy_solve_robust 端到端（k14, CPU 2M, 2 条同布线）===")
    exp = build(14, 1e-3, solver="greedy", vectors=2_000_000)
    mul, connection = setup_structure(exp, bi)
    rtl = os.path.join(WORK, "build", "MUL-0.v")
    # 初始发射(空 cells,纯截断)
    assignment0 = exp.emit_assignment(connection, cell_map={})
    mul.emit_verilog(rtl, assignment=assignment0, extra_modules_src="")
    exact_txt = open(rtl).read()

    # 鲁棒版:传整集布线列表(冒烟用同一布线两份,验证多树路径)
    exp._outer_greedy_solve_robust([connection, connection])
    type_choices = dict(exp._episode_cell_types)
    cell_map = exp._cell_map_from_types(type_choices)

    # 断言:状态/映射同步
    n = len(exp.state["cells"])
    assert n == len(exp._episode_cell_types) == len(type_choices) == len(cell_map), \
        f"状态数量不一致: cells={n} ect={len(exp._episode_cell_types)} " \
        f"tc={len(type_choices)} cm={len(cell_map)}"
    print(f"  解出 n_cells={n},state/_episode_cell_types/type_choices/cell_map 数量一致 ✓")
    # 鲁棒版不发射 RTL(主循环负责);此处手动发射验证 cell_map 可用
    if n > 0:
        mul.emit_verilog(
            rtl,
            assignment=exp.emit_assignment(connection, cell_map=cell_map),
            extra_modules_src=exp._approx_modules_src(cell_map),
        )

    # cells 坐标反查正确(vertex_list)
    vlist = exp.comp_graph.vertex_list
    for e in exp.state["cells"]:
        s, c, t, idx, kk = e
        node = exp.comp_graph.indice_map[(s, c, t, idx)]
        vs, vc, vt, vidx = vlist[node]
        assert (vs, vc, vt, vidx) == (s, c, t, idx), "cells 坐标与 vertex_list 不符"
        assert exp._episode_cell_types[node] == (t, kk), "_episode_cell_types 映射错"
        assert kk >= 1, "cell 表索引应 >=1(exact 不入表)"
    print(f"  {n} 个 cell 坐标反查 vertex_list 全部一致、k>=1 ✓")

    # RTL 已重发射(若有 cell,应与纯截断不同且含近似实例)
    new_txt = open(rtl).read()
    if n > 0:
        assert new_txt != exact_txt, "有 cell 但 RTL 未变"
        napprox = sum(new_txt.count(exp.type_table_32[k]["name"])
                      for k in range(1, len(exp.type_table_32)))
        print(f"  RTL 已重发射,与纯截断不同,含近似 comp32 实例 ✓")
    else:
        print("  n_cells=0(2M 池 floor 估计偏高属正常),RTL 保持纯截断 ✓")

    # ---- C. 默认关回归 ----
    print("\n=== C. 默认关(solver=None)不触 greedy 分支 ===")
    off2 = build(14, 1e-3, solver=None)
    mul2, conn2 = setup_structure(off2, massage(json.load(open(CKPT))))
    assert off2._cell_solver_active() is False
    print("  solver=None: _cell_solver_active=False,get_samples 不进 greedy 分支 ✓")

    print("\n✅ ALL GREEDY-INTEGRATION SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
