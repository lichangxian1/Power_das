#!/usr/bin/env python3
"""统一 substd 菜单端到端冒烟（EDA-free）：一份 json 管 22/32/42。

断言：
  A. loader 从统一菜单读出 T22/T32/T42 三张类型表（全非 exact cell 都是 substd/comp42s）
  B. 每类近似 cell 的 Verilog module 都能发射（22/32 LUT emit + 42 从 comp42n_lib.v 抽取）
  C. 采样布线后整树 emit_verilog 可综合（含 42 cell 免 cin 发射）+ verilator 编译过
  D. approx42_max_types 被 Branch A 忽略（菜单 7 个 42 cell 全在，不截断）

用法: SMOKE_DIR=/tmp/xxx python scripts/smoke_unified_substd_menu.py
"""
import copy
import os
import random
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PD)
from trainer.arith_das import CompressorRouting  # noqa: E402

SP = os.environ.get("SMOKE_DIR", "/tmp/power_das_unified_smoke")
MENU = f"{PD}/Appr_Comp/selected_compressors_all_substd.json"
LIB22 = f"{PD}/Appr_Comp/library.json"
RTL42 = f"{PD}/Appr_Comp/rtl/comp42n_lib.v"
K = 12


def build():
    cfg = OmegaConf.to_container(
        OmegaConf.load(f"{PD}/configs/config_groups/mul_16_approx_error_obj.yaml"),
        resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    run_dir = os.path.join(SP, "run")
    os.makedirs(run_dir, exist_ok=True)
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(run_dir, "logs"),
        "build_dir": os.path.join(run_dir, "build"),
        "experiment_prefix": "unified",
        # === 统一菜单：一份文件管全部 ===
        "approx_lib_path": MENU,             # 22/32/42 都从这里选
        "approx_library_path": LIB22,        # 22/32 LUT backing
        "use_ct42": True,
        "approx42_library_path": MENU,       # 42 LUT + native-4 识别 + anchor
        "approx42_rtl_path": RTL42,          # 42 结构化 RTL
        "approx42_max_types": 3,             # 故意设小，验证 Branch A 忽略它
        "trunc_cols": K,
        "num_episodes": 2, "num_samples": 2, "n_processing": 1,
    })
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = 2.8e-4
    return exp


def main():
    os.makedirs(SP, exist_ok=True)
    exp = build()

    # A. 三张类型表
    t22, t32, t42 = exp.type_table_22, exp.type_table_32, exp.type_table_42
    print(f"A. T22={len(t22)} T32={len(t32)} T42={len(t42)}")
    assert t22[0]["group"] == "exact" and t32[0]["group"] == "exact"
    assert t42[0]["group"] == "exact" and t42[0]["name"] == "CT42", t42[0]
    apx22 = [e["name"] for e in t22[1:]]
    apx32 = [e["name"] for e in t32[1:]]
    apx42 = [e["name"] for e in t42[1:]]
    print(f"   T22 approx: {apx22}")
    print(f"   T32 approx: {apx32}")
    print(f"   T42 approx: {apx42}")
    # D. Branch A 不截断：max_types=3 但 42 仍 7 个
    assert len(apx42) == 7, f"Branch A 应无视 max_types，得 7 个，实际 {len(apx42)}"
    assert all(n.startswith("comp42s") for n in apx42), "T42 应全是 comp42s"
    assert exp._ct42_native4_names.issuperset(set(apx42)), "comp42s 须被认作 native-4(免 cin)"
    print("D. Branch A 忽略 approx42_max_types=3 → 全 7 个 comp42s 保留 ✓")

    # B. 每类近似 cell 都有可发射 module
    for n in apx22 + apx32 + apx42:
        assert n in exp.approx_module_src_by_name, f"{n} 无 module 源"
        src = exp.approx_module_src_by_name[n]
        assert f"module {n}" in src, n
    # 42 免 cin：module 端口无 cin
    for n in apx42:
        head = exp.approx_module_src_by_name[n].split(";")[0]
        assert "cin" not in head, f"{n} 不该有 cin 端口: {head}"
    print(f"B. {len(apx22)+len(apx32)+len(apx42)} 个近似 module 全部可发射（42 免 cin）✓")

    # C. 采样布线 → 整树 emit（含 22/32/42 三类近似 cell）→ verilator 编译
    from utils.compressor_tree import CompressorTree, get_initial_partial_product
    from utils.mul import Mul
    from trainer.arith_das import CompressorGraph
    # 初始化状态 + promote 若干 CT42 节点（确定性构图）
    if exp.state is None:
        exp.state = copy.deepcopy(exp.pool.get_pool()[0][1])
    done = 0
    for _ in range(64):
        if done >= 3:
            break
        mask = exp.get_action_mask()
        legal = [a for a in np.where(mask)[0]
                 if a % 6 == 4 and exp._is_approx_col_allowed(a // 6)]
        if not legal:
            break
        exp.transition(int(random.choice(legal)))
        done += 1
    pp0 = get_initial_partial_product(exp.bit_width, exp.encode_type)
    ct0 = CompressorTree(pp0, exp.state["ct32"], exp.state["ct22"], exp.state.get("ct42"))
    exp.assignment = ct0.compressor_assignment_fused()
    exp.comp_graph = CompressorGraph(pp0, exp.assignment, num_node_types=exp.num_node_types)
    vl = exp.comp_graph.vertex_list
    n42 = [i for i, v in enumerate(vl) if v[2] == 4 and exp._is_approx_col_allowed(v[1])]
    n32 = [i for i, v in enumerate(vl) if v[2] == 0 and exp._is_approx_col_allowed(v[1])]
    n22 = [i for i, v in enumerate(vl) if v[2] == 1 and exp._is_approx_col_allowed(v[1])]
    assert n42 and n32 and n22, f"可近似节点 42={len(n42)} 32={len(n32)} 22={len(n22)}"
    with torch.no_grad():
        conn, _lp = exp.sample_from_logits(exp.get_Z_mat())
    # 三类各挑一个近似 cell（42 用 orha_n=第1个近似, 32/22 用第1个近似）
    type_choices = {n42[0]: (4, 1), n32[0]: (0, 1), n22[0]: (1, 1)}
    cell_map = exp._cell_map_from_types(type_choices)
    assignment = exp.emit_assignment(conn, cell_map=cell_map)
    pp = get_initial_partial_product(exp.bit_width, exp.encode_type)
    ct = CompressorTree(pp, exp.state["ct32"], exp.state["ct22"], exp.state.get("ct42"))
    ct.trunc_cols, ct.trunc_bits = exp.trunc_cols, exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    vpath = os.path.join(SP, "unified_tree.v")
    mul.emit_verilog(vpath, assignment=assignment,
                     extra_modules_src=exp._approx_modules_src(cell_map))
    src = open(vpath).read()
    for nid, (t, k) in type_choices.items():
        name = (t42 if t == 4 else t32 if t == 0 else t22)[k]["name"]
        assert f"module {name}" in src, f"{name} 模块体未注入"
    print(f"C. 整树 emit：注入 42={t42[1]['name']} 32={t32[1]['name']} 22={t22[1]['name']}")
    objd = os.path.join(SP, "obj")
    r = subprocess.run(
        ["verilator", "--cc", "-Wno-fatal", "--top-module", "MUL", "--Mdir", objd, vpath],
        capture_output=True, text=True)
    assert r.returncode == 0, "verilator 编译失败:\n" + r.stderr[-1200:]
    print(f"   整树 RTL verilator 编译通过 ({vpath})")

    print("\nALL UNIFIED-MENU SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
