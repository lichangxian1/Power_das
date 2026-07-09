#!/usr/bin/env python3
"""外环实测误差预筛门（outer_errgate）控制流冒烟 — EDA/verilator-free。
monkeypatch _measure_error_verilator 为脚本化假测量，断言：
  A. 超预算 → 贪心摘 wae·2^col 最大 cell → 重发射 → 复测通过；映射/cell_map/RTL 同步
  B. 修复步数耗尽仍超 → 清空 cells 保底，且清空后不再复测
  C. verilator 探测失败(None) → 放行，cells 不动
  D. 开关口径：默认关不激活；MRED 模式看 mred_budget；error_as_metric 无预算不激活
用法: SMOKE_DIR=/tmp/xxx python scripts/smoke_outer_errgate.py
"""
import copy
import os
import random
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.arith_das import CompressorGraph, CompressorRouting
from utils.compressor_tree import CompressorTree, get_initial_partial_product
from utils.mul import Mul

SP = os.environ.get("SMOKE_DIR", "/tmp/power_das_outer_errgate_smoke")
PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, MRED_BUDGET = 12, 2.8e-4


def build_trainer(extra):
    cfg = OmegaConf.to_container(
        OmegaConf.load(f"{PD}/configs/config_groups/mul_16_approx_error_obj.yaml"),
        resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    os.makedirs(SP, exist_ok=True)
    run_dir = os.path.join(SP, "gate_run")
    os.makedirs(run_dir, exist_ok=True)
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(run_dir, "logs"),
        "build_dir": os.path.join(run_dir, "build"),
        "experiment_prefix": "gate_smoke",
        "trunc_cols": K,
        "num_episodes": 3, "num_samples": 2, "n_processing": 1,
        "outer_cell_search": True,
    })
    tk.update(extra)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = MRED_BUDGET
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()
    return exp


def setup_episode(exp, n_cells=3):
    """reset() 的确定性替身：取池首状态、种 n_cells 个 cell、建图、发射 sample-0 RTL。
    返回 (mul, conn, rtl_path, cells)。"""
    exp.state = copy.deepcopy(exp.pool.get_pool()[0][1])
    exp.state.setdefault("cells", [])
    pp = get_initial_partial_product(exp.bit_width, exp.encode_type)
    ct = CompressorTree(pp, exp.state["ct32"], exp.state["ct22"], exp.state.get("ct42"))
    exp.assignment = ct.compressor_assignment_fused()
    exp.comp_graph = CompressorGraph(pp, exp.assignment,
                                     num_node_types=exp.num_node_types)
    slots = exp._enumerate_type_slots(exp.assignment)
    assert len(slots) >= n_cells, f"可用 slot 不足: {len(slots)}"
    # 取不同列的前 n 个 slot，k 从 1 递增（保证 wae·2^col 贡献可区分）
    picked, used_cols = [], set()
    for s in slots:
        if s[1] in used_cols:
            continue
        used_cols.add(s[1])
        _h, table = exp._type_head_and_table(s[2])
        k = min(1 + len(picked), len(table) - 1)
        picked.append([s[0], s[1], s[2], s[3], k])
        if len(picked) == n_cells:
            break
    assert len(picked) == n_cells
    exp.state["cells"] = [list(e) for e in picked]
    assert exp._refresh_episode_cell_types() == 0
    with torch.no_grad():
        Z = exp.get_Z_mat()
        conn, _lp = exp.sample_from_logits(Z)
    cell_map = exp._cell_map_from_types(exp._episode_cell_types)
    ct.trunc_cols, ct.trunc_bits = exp.trunc_cols, exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    rtl_path = os.path.join(SP, "gate_run", "MUL-0.v")
    mul.emit_verilog(rtl_path,
                     assignment=exp.emit_assignment(conn, cell_map=cell_map),
                     extra_modules_src=exp._approx_modules_src(cell_map))
    return mul, conn, rtl_path, picked


def patch_measure(script):
    """script: list，每次调用 pop(0)；元素 None=探测失败，float=返回该 mred。"""
    calls = []

    def fake(rtl_path, build_path, n_vectors):
        calls.append(n_vectors)
        v = script.pop(0)
        if v is None:
            return None
        return {"med": v * 1e5, "bias": 0.0, "wce_mc": 0.0,
                "mred": v, "source": "fake"}

    CompressorRouting._measure_error_verilator = staticmethod(fake)
    return calls


def cell_contrib(exp, e):
    _h, table = exp._type_head_and_table(int(e[2]))
    return float(table[int(e[4])]["wae"]) * float(1 << int(e[1]))


# ============ A. 超预算 → 摘最坏 cell → 复测通过 ============
exp = build_trainer({"outer_errgate": True})
assert exp._outer_gate_active(), "MRED 模式 + mred_budget 应激活预筛门"
mul, conn, rtl_path, cells0 = setup_episode(exp, 3)
worst = max(cells0, key=lambda e: cell_contrib(exp, e))
worst_name = exp._type_head_and_table(int(worst[2]))[1][int(worst[4])]["name"]
calls = patch_measure([MRED_BUDGET * 3, MRED_BUDGET * 0.5])  # 超→修复→过
tc, cm = exp._outer_errgate_screen(mul, conn, rtl_path)
assert len(calls) == 2, f"应探测 2 次, 实际 {len(calls)}"
assert len(exp.state["cells"]) == 2 and worst[:4] not in \
    [e[:4] for e in exp.state["cells"]], "应摘除 wae·2^col 最大的 cell"
assert tc == dict(exp._episode_cell_types) and len(tc) == 2
assert set(cm.values()) == {
    exp._type_head_and_table(int(e[2]))[1][int(e[4])]["name"]
    for e in exp.state["cells"]}, "cell_map 应与修复后 state 同步"
src = open(rtl_path).read()
kept_names = set(cm.values())
if worst_name not in kept_names:  # 同型同 k 的 cell 可能复用模块名
    assert not re.search(rf"\b{worst_name}\b +\w+_\d+ \(", src), \
        "被摘 cell 的实例不应再出现在重发射的 RTL 中"
print(f"A. OK: 超budget摘除最坏cell {worst_name}@col{worst[1]} (贡献 "
      f"{cell_contrib(exp, worst):.1f} LSB), 2次探测, RTL/映射/cell_map 同步")

# ============ B. 修复耗尽 → 清空保底, 清空后不复测 ============
exp_b = build_trainer({"outer_errgate": True, "outer_errgate_max_repairs": 2})
mul, conn, rtl_path, _ = setup_episode(exp_b, 3)
calls = patch_measure([MRED_BUDGET * 3] * 10)  # 永远超
tc, cm = exp_b._outer_errgate_screen(mul, conn, rtl_path)
assert exp_b.state["cells"] == [] and tc == {} and cm == {}
assert len(calls) == 3, f"3→2→1(清空即停): 应 3 次探测, 实际 {len(calls)}"
src = open(rtl_path).read()
assert "comp32_" not in src and "comp22_" not in src, "清空后 RTL 不应含近似模块"
print(f"B. OK: 修复耗尽清空保底, {len(calls)} 次探测后停")

# ============ C. 探测失败 → 放行不动 ============
exp_c = build_trainer({"outer_errgate": True})
mul, conn, rtl_path, cells0 = setup_episode(exp_c, 2)
calls = patch_measure([None])
tc, cm = exp_c._outer_errgate_screen(mul, conn, rtl_path)
assert len(calls) == 1 and len(exp_c.state["cells"]) == 2 and len(tc) == 2
print("C. OK: verilator 失败放行, cells 不动")

# ============ D. 开关口径 ============
exp_d = build_trainer({})  # outer_errgate 默认 False
assert not exp_d._outer_gate_active(), "默认关"
exp_d.outer_errgate = True
assert exp_d._outer_gate_active(), "MRED+budget 应激活"
exp_d.mred_budget = None
assert not exp_d._outer_gate_active(), "error_as_metric(无 mred_budget) 不激活"
exp_d.error_metric = "med"
assert not exp_d._outer_gate_active(), "med 模式无 med_budget 不激活"
exp_d.med_budget = 1e4
assert exp_d._outer_gate_active(), "med_budget 应激活"
over, d = exp_d._gate_budget_exceeded({"med": 2e4})
assert over, "med 超预算应判超"
over, d = exp_d._gate_budget_exceeded({"med": 5e3})
assert not over
print("D. OK: 开关与预算口径 (默认关/mred/med/error_as_metric)")

print("\n全部通过 ✔")
