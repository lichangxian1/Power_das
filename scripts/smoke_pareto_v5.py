#!/usr/bin/env python3
"""v5 多目标（混合 k 种群 + 非支配档案）冒烟，EDA-free。

  T0 ParetoArchive 单元：分箱/支配/ε_power/去重/拥挤淘汰/最近非空箱
  T1 旧行为回归：不 enable → found_best_info property 走 legacy、reset 走池、
     update_found_best_info 走标量 rank
  T2 M0 k 线程化：per-k TruncProfile lazy 缓存，C* 跨 k 不同、缓存命中不重算
  T3 v5 流程：Dadda 种子集（免变异）→ 档案准入 → 亲代继承 k → 变异后 k 存活
     → 代表 property → export_front 逐 k RTL 发射（截断常数随 k 变）
  T4 M2 zero 算子：_zero_entry_of 找 Z 型、zero-col 整列填/逐列爬升/unzero 反向、
     骰子接入、跳过闭式过滤（紧 slack 下仍能加）、锚点口径
  T5 M2 TT oracle：小池（200k）实测 mred、宽限放行、紧限二分修剪+重发射
用法: SMOKE_DIR=/tmp/xxx ~/anaconda3/envs/arith_das/bin/python scripts/smoke_pareto_v5.py
"""
import copy
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.arith_das import CompressorRouting  # noqa: E402
from utils.common import ParetoArchive  # noqa: E402
from utils.compressor_tree import CompressorTree  # noqa: E402

PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("SMOKE_DIR", "/tmp/power_das_v5_smoke")
os.makedirs(SP, exist_ok=True)
FAILS = []


def check(name, cond, msg=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  {msg}" if msg else ""))
    if not cond:
        FAILS.append(name)


# ────────────────────────── T0 archive 单元 ──────────────────────────
def t0():
    a = ParetoArchive(mred_lo=1e-7, mred_hi=2e-1, bin_ratio=2.0, bin_cap=3,
                      eps_power=0.01)
    check("T0.bins", a.n_bins == 21, f"n_bins={a.n_bins}")
    check("T0.bin_of", a.bin_of(1e-7) == 0 and a.bin_of(1.9e-7) == 0
          and a.bin_of(2.1e-7) == 1 and a.bin_of(0.19) == 20
          and a.bin_of(0.21) is None and a.bin_of(None) is None)
    ok1, b1 = a.add(1e-4, 700.0, 9.0, {"ct": {"k": 12}})
    check("T0.admit_first", ok1 and b1 == 9)
    ok2, _ = a.add(1e-4, 710.0, 9.5, {"ct": {"k": 12}})     # 被支配
    check("T0.reject_dominated", not ok2 and len(a) == 1)
    ok3, _ = a.add(1e-4, 710.0, 8.0, {"ct": {"k": 12}})     # 面积大功耗小 → 非支配
    check("T0.keep_tradeoff", ok3 and len(a) == 2)
    ok4, _ = a.add(1e-4, 700.0, 9.05, {"ct": {"k": 12}})    # ε_power 内重复
    check("T0.reject_eps_dup", not ok4)
    ok5, _ = a.add(1e-4, 650.0, 8.9, {"ct": {"k": 10}})     # 支配 700/9.0 → 顶掉
    ents = a.bins[9]
    check("T0.dominate_evicts", ok5 and len(ents) == 2
          and not any(e["area"] == 700.0 for e in ents))
    a.add(1e-4, 680.0, 8.4, {"ct": {"k": 11}})
    a.add(1e-4, 660.0, 8.7, {"ct": {"k": 11}})              # cap=3 → 拥挤淘汰
    ents = sorted(a.bins[9], key=lambda e: e["area"])
    check("T0.crowding_cap", len(ents) == 3
          and ents[0]["area"] == 650.0 and ents[-1]["area"] == 710.0,
          f"areas={[e['area'] for e in ents]}")
    check("T0.nearest", a.nearest_nonempty(9) == 9 and a.nearest_nonempty(3) == 9
          and a.nearest_nonempty(20) == 9)
    check("T0.parent", a.sample_parent(5, random) is not None
          and a.sample_parent(5, random)["payload"]["ct"]["k"] in (10, 11, 12))
    snap = a.snapshot()
    check("T0.snapshot", len(snap) == 3 and all(s["bin"] == 9 for s in snap))
    empty = ParetoArchive()
    check("T0.empty", empty.sample_parent(0, random) is None
          and empty.global_min_area() is None and len(empty) == 0)


# ────────────────────────── trainer 构造 ──────────────────────────
def build_trainer(extra, tag):
    cfg = OmegaConf.to_container(
        OmegaConf.load(f"{PD}/configs/config_groups/mul_16_approx_error_obj.yaml"),
        resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    run_dir = os.path.join(SP, tag)
    os.makedirs(run_dir, exist_ok=True)
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "device": "cpu",
        "log_dir": os.path.join(run_dir, "logs"),
        "build_dir": os.path.join(run_dir, "build"),
        "experiment_prefix": tag,
        "trunc_cols": 12,
        "approx_max_col": 30, "approx_col_window": 6,
        "num_episodes": 3, "num_samples": 2, "n_processing": 1,
    })
    tk.update(extra)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    exp = CompressorRouting(**tk)
    exp.error_metric = "mred"
    exp.mred_budget = 2.8e-4
    exp.mred_scale = 1e-3
    exp._trunc_bits = {}
    exp._setup_truncation()          # train_dc 同序：mred 分支重算 C*
    return exp


def fake_sample(exp, mred, area, power):
    """离线伪样本：真实采样 connection（发射可用），PPA/误差喂假值。"""
    Z = exp.get_Z_mat()
    conn, _lp = exp.sample_from_logits(Z)
    res = [{"delay": 1.4, "area": float(area), "power": float(power)}]
    me = {"med": 1.0, "bias": 0.0, "mred": float(mred), "source": "smoke"}
    return {
        "connection": conn, "result": res, "measured_error": me,
        "cell_types": {}, "cell_type_info": {"mode": "smoke"},
        "objective": exp.get_objective(res, cell_types={}, measured_error=me),
    }


# ────────────────────────── T1 旧行为回归 ──────────────────────────
def t1():
    exp = build_trainer({"outer_cell_search": False}, "t1_legacy")
    check("T1.flag_off", not getattr(exp, "pareto_v5", False))
    check("T1.property_legacy",
          exp.found_best_info is exp._found_best_info
          and exp.found_best_info["objective"] == float("inf"))
    exp.reset()
    check("T1.reset_no_k", "k" not in exp.state)
    s = fake_sample(exp, 2.0e-4, 700.0, 0.0095)
    exp.update_found_best_info([s])
    check("T1.legacy_update", exp.found_best_info["objective"] == s["objective"]
          and exp.found_best_info["connection"] is not None)


# ────────────────────────── T2 M0 档缓存 ──────────────────────────
def t2():
    exp = build_trainer({"outer_cell_search": False}, "t2_profiles")
    consts, floors = {}, {}
    for k in (12, 8, 2):
        exp._activate_trunc_profile(k)
        # C*=0（如 k=2 的 MRED 最优）时常数位表合法为空
        check(f"T2.k{k}_active", exp.trunc_cols == k
              and (bool(exp._trunc_bits) or exp._trunc_const == 0))
        consts[k] = exp._trunc_const
        floors[k] = exp._trunc_model_mred
    check("T2.const_diff", consts[12] > consts[8] > consts[2] >= 0,
          f"C*={consts}")
    check("T2.floor_order", (floors[12] or 0) > (floors[8] or 0),
          f"floors={floors}")
    c12 = consts[12]
    import time
    t = time.time()
    exp._activate_trunc_profile(12)   # 缓存命中应瞬间
    dt = time.time() - t
    check("T2.cache_hit", exp._trunc_const == c12 and dt < 0.05,
          f"dt={dt*1000:.1f}ms")
    check("T2.cache_keys", sorted(exp._trunc_profiles) == [2, 8, 12])
    return exp


# ────────────────────────── T3 v5 流程 ──────────────────────────
def t3():
    exp = build_trainer({"outer_cell_search": True}, "t3_v5")
    exp.enable_pareto_v5(mred_lo=1e-7, mred_hi=2e-1, bin_ratio=2.0,
                         bin_cap=4, eps_power=0.01, seed_ks=[2, 8, 12])
    dadda32 = CompressorTree.dadda(exp.initial_pp).ct32.astype(int)
    seen_seed_k = []
    for ep in range(5):
        exp._v5_begin_episode(ep)
        exp.reset()
        if exp._v5_seeding:
            seen_seed_k.append(exp._v5_seed_k)
            check(f"T3.seed_ep{ep}_dadda",
                  np.array_equal(exp.state["ct32"], dadda32)
                  and exp.state["k"] == exp._v5_seed_k
                  and exp.trunc_cols == exp._v5_seed_k
                  and exp.state.get("cells") == [])
        else:
            check(f"T3.evo_ep{ep}_k_inherited",
                  exp.state.get("k") in (2, 8, 12)
                  and exp.trunc_cols == exp.state["k"],
                  f"k={exp.state.get('k')}")
        # 伪评估：种子集在 floor×1.05 入箱；进化集在伪预算 0.9 处入箱
        floor = exp._trunc_model_mred or 1e-7
        mred = min(floor * 1.05, exp.mred_budget * 0.9)
        base = 900.0 - 15.0 * exp.trunc_cols
        s1 = fake_sample(exp, mred, base, base * 1.3e-5)
        s2 = fake_sample(exp, mred * 1.1, base - 5.0, base * 1.32e-5)
        exp.update_found_best_info([s1, s2])
    check("T3.seed_order", seen_seed_k == [2, 8, 12], f"{seen_seed_k}")
    arch = exp._v5_archive
    check("T3.archive_filled", len(arch) >= 4 and arch.n_nonempty() >= 3,
          f"pts={len(arch)} bins={arch.n_nonempty()}")
    rep = exp.found_best_info
    check("T3.representative", rep is not None and rep.get("connection") is not None
          and "k" in rep["ct"])
    # 支配语义再验：同箱塞一个被支配点，档案不该收
    b_used = next(b for b in range(arch.n_bins) if arch.bins[b])
    e0 = arch.bins[b_used][0]
    ok, _ = arch.add(e0["mred"], e0["area"] + 50, e0["power"] + 1e-4,
                     {"ct": {"k": 99}})
    check("T3.no_dominated_entry", not ok)
    # front 导出：逐 k RTL，截断常数应随 k 不同
    front_dir = os.path.join(SP, "t3_v5", "front")
    n = exp.export_front(front_dir)
    dirs = sorted(d for d in os.listdir(front_dir)
                  if os.path.isdir(os.path.join(front_dir, d)))
    check("T3.front_export", n == len(arch) and n == len(dirs) and n >= 4,
          f"n={n} dirs={dirs}")
    check("T3.front_glob_k", all(d.startswith("k") for d in dirs))
    rtls = {}
    for d in dirs:
        kk = int(d[1:3])
        p = os.path.join(front_dir, d, "MUL.v")
        if os.path.exists(p):
            rtls.setdefault(kk, open(p).read())
    ks = sorted(rtls)
    check("T3.rtl_per_k", len(ks) >= 2 and any(
        rtls[ks[0]] != rtls[k2] for k2 in ks[1:]), f"ks={ks}")
    check("T3.front_json", os.path.exists(os.path.join(front_dir, "front.json")))


# ────────────────────────── T4 M2 zero 算子 ──────────────────────────
MENU_SUBSTD = {
    "approx_lib_path": "Appr_Comp/selected_compressors_all_substd.json",
    "approx_library_path": "Appr_Comp/library.json",
}


def t4():
    exp = build_trainer({"outer_cell_search": True, "outer_zero_ops": True,
                         **MENU_SUBSTD}, "t4_zero")
    rng = np.random.default_rng(3)
    check("T4.anchor", exp._EXACT_AREA_INCTX == {0: 2.856, 1: 2.184, 4: 5.712})
    kz32, kz22 = exp._zero_entry_of(0), exp._zero_entry_of(1)
    e32, e22 = exp.type_table_32[kz32], exp.type_table_22[kz22]
    check("T4.zero_entry_const", kz32 is not None and kz22 is not None
          and e32.get("const_zero") and e22.get("const_zero")
          and e32["name"] == "comp32_zero" and e22["name"] == "comp22_zero",
          f"kz32={e32.get('name')} kz22={e22.get('name')}")
    # 语义回归：绝不能再挑到 Z 组的零偏置功能 cell（N/Z/P 是偏置符号分组）
    check("T4.zero_not_biasZ", e32.get("group") == "N" and e22.get("group") == "N")
    exp.reset()   # 初始化 state（legacy 池路径）
    slots = exp._enumerate_type_slots(exp._current_assignment())
    cols = sorted({sl[1] for sl in slots})
    check("T4.slots_window", cols and cols[0] == exp.trunc_cols, f"cols={cols}")
    c1 = exp._op_zero_col([], slots, rng)
    n_c0 = sum(1 for sl in slots
               if sl[1] == cols[0] and exp._zero_entry_of(sl[2]) is not None)
    check("T4.zero_col_fills_lowest", c1 is not None
          and len(c1) == n_c0 and all(e[1] == cols[0] for e in c1),
          f"n={len(c1) if c1 else 0}/{n_c0}")
    c2 = exp._op_zero_col(c1, slots, rng)
    check("T4.zero_col_climbs", c2 is not None
          and {e[1] for e in c2} == {cols[0], cols[1]},
          f"cols={sorted({e[1] for e in c2}) if c2 else None}")
    c3 = exp._op_unzero_col(c2, rng)
    check("T4.unzero_lowest", c3 is not None
          and {e[1] for e in c3} == {cols[1]})
    # 骰子接入 + 跳过闭式过滤：slack 压到 ~0，zero 算子仍应放 cell
    exp.mred_budget = (exp._trunc_model_mred or 2e-4) * 1.001
    exp.outer_p_struct, exp.outer_p_cell = 0.0, 0.0
    exp.outer_p_resample, exp.outer_p_zero = 0.0, 1.0
    exp.reset()
    zc = exp.state.get("cells") or []
    check("T4.dice_bypass_filter", len(zc) >= max(1, n_c0 - 1)
          and all((exp.type_table_32 if int(e[2]) == 0 else
                   exp.type_table_22)[int(e[4])].get("const_zero")
                  for e in zc),
          f"n_cells={len(zc)}")
    return exp


# ────────────────────────── T5 M2 TT oracle ──────────────────────────
def t5():
    exp = build_trainer({"outer_cell_search": True, "outer_zero_ops": True,
                         "outer_tt_oracle": True,
                         "outer_solver_vectors": 200_000, **MENU_SUBSTD},
                        "t5_oracle")
    exp.enable_pareto_v5(mred_lo=1e-7, mred_hi=2e-1, bin_ratio=2.0,
                         bin_cap=4, eps_power=0.01, seed_ks=[12])
    exp._v5_begin_episode(0)
    exp.reset()
    # 手工装一列 ZERO + 采样 sample-0 布线，模拟 get_samples 预筛点
    slots = exp._enumerate_type_slots(exp._current_assignment())
    rng = np.random.default_rng(5)
    exp.state["cells"] = exp._op_zero_col([], slots, rng)
    n0 = len(exp.state["cells"])
    exp._refresh_episode_cell_types()
    conn, _lp = exp.sample_from_logits(exp.get_Z_mat())
    from utils.mul import Mul  # noqa: E402
    ct = CompressorTree(exp.initial_pp, exp.state["ct32"], exp.state["ct22"],
                        exp.state.get("ct42"))
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    rtl = os.path.join(SP, "t5_oracle", "MUL-0.v")
    cm0 = exp._cell_map_from_types(exp._episode_cell_types)
    mul.emit_verilog(rtl, assignment=exp.emit_assignment(conn, cell_map=cm0),
                     extra_modules_src=exp._approx_modules_src(cm0))
    # 宽限（v5 档案上限 2e-1）：应实测后原样放行
    tc, _cm = exp._outer_tt_oracle_screen(mul, conn, rtl)
    check("T5.pass_loose", len(tc) == n0 and len(exp.state["cells"]) == n0,
          f"n={len(tc)}/{n0}")
    check("T5.est_cached", getattr(exp, "_cell_solver_est", None) is not None)
    # 紧限（预算模式 floor×1.02）：应二分修剪 + 重发射
    exp.pareto_v5 = False
    exp.mred_budget = (exp._trunc_model_mred or 2e-4) * 1.02
    mt0 = os.path.getmtime(rtl)
    tc2, _cm2 = exp._outer_tt_oracle_screen(mul, conn, rtl)
    check("T5.trim_tight", len(tc2) < n0
          and len(exp.state["cells"]) == len(tc2),
          f"n={n0}->{len(tc2)}")
    check("T5.reemit", os.path.getmtime(rtl) >= mt0)
    exp.pareto_v5 = True


if __name__ == "__main__":
    t0()
    t1()
    t2()
    t3()
    t4()
    t5()
    print("\n" + ("ALL PASS ✅" if not FAILS else f"FAILED ❌: {FAILS}"))
    sys.exit(1 if FAILS else 0)
