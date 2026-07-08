#!/usr/bin/env python3
"""approx-CT42 × 原生 4:2 库 端到端集成冒烟（EDA-free，本地 verilator）。
复刻 train_dc.py 构造序列（MRED 接线 + C* 重算），断言：
  A. loader/类型表  B. 内环发射+解析误差对拍+PPO/可微梯度
  C. 外环提议/过滤/映射/get_samples  D. verilator MC |e|<=解析 WCE 上界
用法: SMOKE_DIR=/tmp/xxx python scripts/smoke_approx42_integration.py"""
import copy
import json
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainer as trainer_pkg
from trainer.arith_das import CompressorRouting
from utils.compressor_tree import CompressorTree, get_initial_partial_product
from utils.mul import Mul

SP = os.environ.get("SMOKE_DIR", "/tmp/power_das_approx42_smoke")
PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL42 = f"{PD}/Appr_Comp/selected_compressors42_native.json"
RTL42 = f"{PD}/Appr_Comp/rtl/comp42n_lib.v"
K, MRED_BUDGET = 12, 2.8e-4  # Scheme A 的 k12 点


def build_trainer(extra):
    cfg = OmegaConf.to_container(
        OmegaConf.load(f"{PD}/configs/config_groups/mul_16_approx_error_obj.yaml"),
        resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    os.makedirs(SP, exist_ok=True)
    run_dir = os.path.join(SP, "integ42n_run")
    os.makedirs(run_dir, exist_ok=True)
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(run_dir, "logs"),
        "build_dir": os.path.join(run_dir, "build"),
        "experiment_prefix": "integ42n",
        "use_ct42": True,
        "approx42_library_path": SEL42,
        "approx42_rtl_path": RTL42,
        "approx42_max_types": 13,
        "trunc_cols": K,
        "num_episodes": 3, "num_samples": 2, "n_processing": 1,
    })
    tk.update(extra)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = MRED_BUDGET
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()          # 走 mred 分支取 C*（train_dc 同序）
    return exp


def promote_ct42(exp, n_target=3):
    """在近似窗口内 promote 若干 FA+HA -> CT42，重建图（reset 的确定性替身）。"""
    if exp.state is None:
        exp.state = copy.deepcopy(exp.pool.get_pool()[0][1])
    done = 0
    for _ in range(64):
        if done >= n_target:
            break
        mask = exp.get_action_mask()
        legal = [a for a in np.where(mask)[0]
                 if a % 6 == 4 and exp._is_approx_col_allowed(a // 6)]
        if not legal:
            break
        exp.transition(int(random.choice(legal)))
        done += 1
    pp = get_initial_partial_product(exp.bit_width, exp.encode_type)
    ct = CompressorTree(pp, exp.state["ct32"], exp.state["ct22"], exp.state.get("ct42"))
    exp.assignment = ct.compressor_assignment_fused()
    from trainer.arith_das import CompressorGraph
    exp.comp_graph = CompressorGraph(pp, exp.assignment,
                                     num_node_types=exp.num_node_types)
    return done


# ============ A. loader ============
exp = build_trainer({})
t42 = exp.type_table_42
assert len(t42) == 13, f"T42 size {len(t42)}"
assert t42[0]["group"] == "exact" and t42[0]["name"] == "CT42"
assert abs(t42[0]["area"] - 17.304) < 1e-6, f"exact anchor area {t42[0]['area']}"
assert all(e["name"].startswith("comp42n_") for e in t42[1:])
assert exp._ct42_native4_names.issuperset({e["name"] for e in t42[1:]})
waes = [e["wae"] for e in t42[1:]]
assert waes == sorted(waes), "approx42 应按 wae 升序"
print(f"A. loader OK: 13 types, exact area={t42[0]['area']}, wae∈[{waes[0]:.3f},{waes[-1]:.3f}]")

# ============ B. 内环 ============
n42 = promote_ct42(exp, 3)
ct42_nodes = [i for i, v in enumerate(exp.comp_graph.vertex_list)
              if v[2] == 4 and exp._is_approx_col_allowed(v[1])]
assert n42 >= 2 and len(ct42_nodes) >= 2, f"promote {n42}, nodes {len(ct42_nodes)}"
t32_nodes = [i for i, v in enumerate(exp.comp_graph.vertex_list)
             if v[2] == 0 and exp._is_approx_col_allowed(v[1])]

with torch.no_grad():
    Z = exp.get_Z_mat()
    conn, _lp = exp.sample_from_logits(Z)

type_choices = {ct42_nodes[0]: (4, 1), ct42_nodes[1]: (4, 6), t32_nodes[0]: (0, 1)}
cell_map = exp._cell_map_from_types(type_choices)
assignment = exp.emit_assignment(conn, cell_map=cell_map)
pp = get_initial_partial_product(exp.bit_width, exp.encode_type)
ct = CompressorTree(pp, exp.state["ct32"], exp.state["ct22"], exp.state.get("ct42"))
ct.trunc_cols, ct.trunc_bits = exp.trunc_cols, exp._trunc_bits
mul = Mul(exp.bit_width, exp.encode_type, ct)
rtl_path = os.path.join(SP, "integ42n_run", "MUL_inner.v")
mul.emit_verilog(rtl_path, assignment=assignment,
                 extra_modules_src=exp._approx_modules_src(cell_map))
src = open(rtl_path).read()
for nid, (t, k) in type_choices.items():
    if t != 4:
        continue
    name = t42[k]["name"]
    m = re.search(rf"{name} ct42_{nid} \((.*?)\);", src, re.S)
    assert m, f"{name} 未实例化"
    assert ".cin" not in m.group(1), f"{name} 不应接 .cin"
    assert f"module {name} " in src, f"{name} 模块体未注入"
assert src.count("CT42 ct42_") == len(ct42_nodes) + \
    sum(1 for v in exp.comp_graph.vertex_list if v[2] == 4) - len(ct42_nodes) - 2, \
    "其余 ct42 节点应保持 exact CT42"
comp32_name = exp.type_table_32[1]["name"]
assert f"module {comp32_name}" in src, "3:2 近似模块体未注入"

med, abias, nmed, wce = exp._analytic_error(type_choices)
cols = {n: exp.comp_graph.vertex_list[n][1] for n in type_choices}
exp_med = exp._trunc_med + sum(
    (t42[k]["wae"] if t == 4 else exp.type_table_32[k]["wae"]) * (1 << cols[n])
    for n, (t, k) in type_choices.items())
exp_wce = exp._trunc_wce + sum(
    (t42[k]["maxe"] if t == 4 else exp.type_table_32[k]["maxe"]) * (1 << cols[n])
    for n, (t, k) in type_choices.items())
assert abs(med - exp_med) < 1e-6 and abs(wce - exp_wce) < 1e-6, (med, exp_med, wce, exp_wce)
print(f"B1. 内环发射+解析误差 OK: med={med:.1f} wce={wce:.0f} "
      f"(trunc {exp._trunc_med:.1f}/{exp._trunc_wce})")

# PPO 梯度到 type_head_42
exp.optim.zero_grad()
_ = exp.get_Z_mat()  # 有梯度地重建 _node_emb
lp = exp._independent_cell_type_log_prob({ct42_nodes[0]: (4, 1)})
lp.backward()
g = exp.type_head_42.weight.grad
assert g is not None and float(g.abs().sum()) > 0, "type_head_42 无梯度"
# cardinality 重算路径（含 T42 节点）
exp.optim.zero_grad()
_ = exp.get_Z_mat()
info = {"cell_types": {ct42_nodes[0]: (4, 2)},
        "cell_type_info": {"mode": "cardinality", "cardinality_choice_idx": 1,
                            "selected_order": [ct42_nodes[0]]}}
lp2 = exp._cell_type_log_prob(info)
lp2.backward()
assert float(exp.type_head_42.weight.grad.abs().sum()) > 0
assert exp.approx_cardinality_logits.grad is not None
print("B2. PPO 梯度 OK: independent+cardinality 两路都回传 type_head_42/cardinality logits")

# ============ C. 外环 ============
exp2 = build_trainer({"outer_cell_search": True, "outer_p_struct": 0.0,
                      "outer_p_cell": 0.5, "outer_p_resample": 0.5})
promote_ct42(exp2, 3)
exp2.state["cells"] = []
slots = exp2._enumerate_type_slots(exp2._current_assignment())
assert any(s[2] == 4 for s in slots), "外环 slot 枚举不含 T42"
rng = np.random.default_rng(0)
adds = [exp2._propose_cell_add([], slots, rng) for _ in range(200)]
adds = [a for a in adds if a]
t42_adds = [a for a in adds if a[2] == 4]
assert t42_adds, "200 次提议无一 T42（打分/过滤有问题）"
assert all(exp2._cells_budget_ok([a]) for a in adds[:50])
# 过滤器负样本：最大 maxe cell 堆到高列应被拒
kmax = max(range(1, len(exp2.type_table_42)), key=lambda k: exp2.type_table_42[k]["wae"])
hi_col = max(s[1] for s in slots)
bad_slot = [s for s in slots if s[1] == hi_col][0]
bad = [[bad_slot[0], bad_slot[1], bad_slot[2], bad_slot[3], kmax]] if bad_slot[2] == 4 else None
# 人造：足够多 cell 必然超 slack
many = []
for s in slots:
    _h, tb = exp2._type_head_and_table(s[2])
    many.append([s[0], s[1], s[2], s[3], len(tb) - 1])
assert not exp2._cells_budget_ok(many), "满配 cell 竟然过滤通过（slack 未生效）"
resample = exp2._op_resample_k(slots, np.random.default_rng(1))
assert exp2._cells_budget_ok(resample)
print(f"C1. 外环提议 OK: {len(adds)}/200 可行、T42 提议 {len(t42_adds)}、"
      f"resample-K={len(resample)}、满配被拒")

# 外环 reset 映射 + get_samples 共用 cell_map + all-exact 配对
exp2.state["cells"] = resample if resample else t42_adds[:1]
exp2.update_pool(1.0, exp2.state)
random.seed(11)
exp2.reset()
assert isinstance(exp2._episode_cell_types, dict)
os.makedirs(exp2.build_dir, exist_ok=True)
sample_info = exp2.get_samples()
kinds = [s.get("candidate_kind") for s in sample_info]
assert "all_exact" in kinds, "inject_exact_candidate 未生效"
non_base = [s for s in sample_info if not s.get("baseline_only")]
maps = {json.dumps(sorted((s.get("cell_types") or {}).items())) for s in non_base}
assert len(maps) == 1, "外环模式下多样本 cell 配置不一致"
if exp2._episode_cell_types:
    s0 = non_base[0]
    rtl = open(s0["rtl_path"]).read()
    for nid, (t, k) in exp2._episode_cell_types.items():
        if t == 4 and k != 0:
            assert exp2.type_table_42[k]["name"] in rtl
assert exp2._cell_type_log_prob(non_base[0]) is None, "外环类型不应进 PPO ratio"
print(f"C2. 外环 reset/get_samples OK: cells={len(exp2._episode_cell_types)}, "
      f"样本 {len(sample_info)}（含 all-exact 配对），类型不进 ratio")

# ============ D. verilator MC 校验（内环 RTL，300k 向量） ============
d = os.path.join(SP, "integ42n_run", "vsim")
os.makedirs(d, exist_ok=True)
bound = int(round(wce))
tb = f'''
#include "VMUL.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    VMUL* top = new VMUL;
    srand(12345);
    long maxabs = 0; long nerr = 0;
    for (int i = 0; i < 300000; i++) {{
        unsigned a = ((unsigned)rand() << 16 | rand()) & 0xFFFF;
        unsigned b = ((unsigned)rand() << 16 | rand()) & 0xFFFF;
        top->a = a; top->b = b; top->clk = 0; top->eval();
        long golden = ((long)a * (long)b) & 0x7FFFFFFFL;
        long e = (long)top->out - golden;
        e = ((e + (1L<<30)) & 0x7FFFFFFFL) - (1L<<30);
        if (e) nerr++;
        if (labs(e) > maxabs) maxabs = labs(e);
    }}
    printf("maxabs=%ld nerr=%ld\\n", maxabs, nerr);
    return 0;
}}
'''
open(os.path.join(d, "tb.cpp"), "w").write(tb)
subprocess.run(["cp", rtl_path, os.path.join(d, "MUL.v")], check=True)
subprocess.run(["verilator", "--cc", "MUL.v", "--top-module", "MUL",
                "--exe", "tb.cpp", "--build", "-o", "sim", "-Wno-fatal"],
               cwd=d, check=True, capture_output=True)
out = subprocess.run(["./obj_dir/sim"], cwd=d, capture_output=True, text=True,
                     check=True).stdout
maxabs = int(re.search(r"maxabs=(\d+)", out).group(1))
nerr = int(re.search(r"nerr=(\d+)", out).group(1))
assert 0 < maxabs <= bound, f"max|e|={maxabs} vs bound={bound}"
print(f"D. verilator OK: 300k 向量 max|e|={maxabs} <= WCE 上界 {bound}, 误差样本 {nerr}")
print("\nALL INTEGRATION CHECKS PASSED")
