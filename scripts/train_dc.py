#!/usr/bin/env python3
"""用远端 DC 直出 PPA 训练 ARITH-DAS（PPA 源 = 远端 DC，功耗取 DC report_power，
不走 VCS/XA）。沿用 unconstrained 加权和目标（含 err_term/bias），但固定单一 DC
时钟周期（fixed_target_delay, ns）以保证每个样本只跑 1 次 DC。

依赖：远端独立 base 副本 sandbox_base_dcpwr（默认 POWER_MODE=dc）。
用法（务必带上环境）：
  source ~/OpenROAD-flow-scripts/env.sh && \
  /home/lee/anaconda3/envs/arith_das/bin/python scripts/train_dc.py \
      --config configs/config_groups/mul_16_and_approx_p2p1.yaml \
      --episodes 2 --samples 2 --med_budget 65536 --target_delay 2.0 \
      --out outputs/dc_train_smoke

DC 量级远小于 ABC，故 *_scale 默认按 DC 重标定（可 CLI 覆盖）。
"""
import argparse
import copy
import logging
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import trainer  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_groups/mul_16_and_approx_p2p1.yaml")
    p.add_argument("--out", required=True, help="run 目录（含 build/ logs/ best_info.json）")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--save_freq", type=int, default=None,
                   help="每多少轮存一次 best 结构 checkpoint（防中途停丢结构；默认用 config 的 100）")
    p.add_argument("--n_processing", type=int, default=None)
    p.add_argument("--med_budget", type=float, default=None)
    p.add_argument("--error_scale", type=float, default=None,
                   help="MED 软罚归一化分母；低误差段需调小（默认 1e5 在低 MED 下罚项可忽略）")
    p.add_argument("--error_weight", type=float, default=None,
                   help="error_as_metric 模式下 med 线性项权重；调小→cell 更便宜、更多 k 用 cell（更不准）")
    p.add_argument("--med_violation_weight", type=float, default=None,
                   help="MED 超预算软罚权重")
    p.add_argument("--trunc_cols", type=int, default=None, help="① 低列截断深度 k（0=无截断）")
    p.add_argument("--approx_col_window", type=int, default=None,
                   help="近似 cell 只在 [k, k+window) 列可选（收窄到截断边界，集中探索廉价低列）")
    p.add_argument("--wce_budget", type=float, default=None, help="④ WCE 上限（LSB）")
    p.add_argument("--error_metric", choices=["med", "mred"], default="med",
                   help="误差指标：med(绝对,默认) 或 mred(相对误差,重罚小积/截断)")
    p.add_argument("--mred_budget", type=float, default=None,
                   help="MRED 软罚预算(分数，如 0.005=0.5%%)；error_metric=mred 时生效")
    p.add_argument("--mred_scale", type=float, default=0.01,
                   help="MRED 软罚归一分母(默认 0.01，使超额 0.01 → 罚 1.0)")
    p.add_argument("--target_delay", type=float, default=1.5, help="DC 时钟周期 (ns)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init_pool_best_info", default=None,
                   help="温启动：用已有 run 的 best_info.json 作为初始训练池（替代默认 "
                        "wallace/dadda 种子），并种 found_best_info；须同 k/口径")
    p.add_argument("--outer_cell_search", action="store_true",
                   help="外环 cell 搜索：类型进外环状态（解析提议+闭式过滤+resample-K），内环只采布线（Appr_Comp/OUTER_CELL_SEARCH.md）")
    p.add_argument("--use_ct42", action="store_true",
                   help="把 4:2 compressor 作为可搜索架构原语接入；use_approx_types=true 时 CT42 也可选近似 cell")
    p.add_argument("--approx42_library_path", default=None,
                   help="4:2 近似压缩器库 JSON，默认使用 trainer 内置的 Appr_Comp/library42_pair32_func.json")
    p.add_argument("--approx42_rtl_path", default=None,
                   help="4:2 近似压缩器 Verilog 库，默认使用 trainer 内置的 Appr_Comp/rtl/comp42_lib.v")
    p.add_argument("--approx42_max_types", type=int, default=None,
                   help="4:2 类型菜单大小上限，含 exact；默认使用 trainer 内置值")
    # DC 重标定（按烟雾实测：DC-direct area~800µm², delay~1.44ns,
    # power~10.7mW＝0.0107W（默认 0.5 翻转率，比 XA/SAIF 高约 20×））
    p.add_argument("--delay_scale", type=float, default=1.44)
    p.add_argument("--area_scale", type=float, default=800.0)
    p.add_argument("--power_scale", type=float, default=1.07e-2)
    p.add_argument("--base_dir_dc", default="/home/lchangxian/sandbox/sandbox_base_dcpwr")
    args = p.parse_args()

    os.environ.setdefault("EDA_BASE_DIR_DC", args.base_dir_dc)

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    exp_kwargs = cfg["experiment"]["kwargs"]
    base = cfg["trainer"]["kwargs"]

    tk = copy.deepcopy(base)
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(exp_kwargs))

    run_dir = os.path.abspath(args.out)
    os.makedirs(run_dir, exist_ok=True)
    tk.update(
        {
            # ── DC-in-the-loop 核心开关 ──
            "synth": "dc",
            "power_source": "eda",
            "use_power_proxy": False,
            # unconstrained 加权和目标（含 err_term/bias）：area_budget=None；
            # 但固定单一 DC 周期，保证每样本只 1 次 DC。
            "area_budget": None,
            "fixed_target_delay": float(args.target_delay),
            # delay 约束化：固定单一 DC 周期后不再线性奖励"更快"（否则 RL 花面积
            # 把延迟压到远低于 target，导致同误差下面积偏大、与 evo 1.8ns 比不公平）。
            # delay_weight=0 ⇒ 目标只剩 (误差,面积,功耗)，delay 退化为综合时序约束。
            "delay_weight": 0.0,
            # 误差闸门用 verilator circular-wrap 真实 MED（16M 向量）取代解析 proxy（codex 审过）。
            # 每候选 worker 内并行测（~3s vs DC ~200s）；verilator 失败回退解析、不丢样本。
            "error_gate": "verilator",
            "error_gate_vectors": 16_000_000,
            # DC 量级重标定
            "delay_scale": float(args.delay_scale),
            "area_scale": float(args.area_scale),
            "power_scale": float(args.power_scale),
            # IO
            "log_dir": os.path.join(run_dir, "logs"),
            "build_dir": os.path.join(run_dir, "build"),
            "experiment_prefix": "dc_" + str(exp_kwargs.get("experiment_prefix", "mul16")),
        }
    )
    if args.use_ct42:
        tk["use_ct42"] = True
    if args.init_pool_best_info is not None:
        tk["init_pool_best_info"] = args.init_pool_best_info
    if args.outer_cell_search:
        tk["outer_cell_search"] = True
    if args.approx42_library_path is not None:
        tk["approx42_library_path"] = args.approx42_library_path
    if args.approx42_rtl_path is not None:
        tk["approx42_rtl_path"] = args.approx42_rtl_path
    if args.approx42_max_types is not None:
        tk["approx42_max_types"] = args.approx42_max_types
    if args.episodes is not None:
        tk["num_episodes"] = args.episodes
        tk.setdefault("scheduler_kwargs", {})["T_max"] = args.episodes  # T_max 必须 == episodes
    if args.samples is not None:
        tk["num_samples"] = args.samples
    if args.save_freq is not None:
        tk["save_freq"] = args.save_freq
    if args.n_processing is not None:
        tk["n_processing"] = args.n_processing
        tk["n_full_target_delay_processing"] = args.n_processing
    if args.med_budget is not None:
        tk["med_budget"] = args.med_budget
    if args.error_scale is not None:
        tk["error_scale"] = args.error_scale
    if args.error_weight is not None:
        tk["error_weight"] = args.error_weight
    if args.med_violation_weight is not None:
        tk["med_violation_weight"] = args.med_violation_weight
    if args.trunc_cols is not None:
        tk["trunc_cols"] = args.trunc_cols
    if args.approx_col_window is not None:
        tk["approx_col_window"] = args.approx_col_window
    if args.wce_budget is not None:
        tk["wce_budget"] = args.wce_budget
    if args.error_metric == "mred":
        # MRED 无法逐节点分解 → 关掉绝对-MED 可微 surrogate，纯靠 verilator-MRED reward 闸门。
        tk["use_error_loss"] = False
        tk["error_gate"] = "verilator"   # MRED 必须用实测（无解析 proxy）
    if args.device is not None:
        tk["device"] = args.device
    seed = args.seed if args.seed is not None else exp_kwargs.get("seed", 42)
    tk["seed"] = seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(run_dir, "train_dc.log")),
        ],
    )
    logging.info("EDA_BASE_DIR_DC=%s", os.environ["EDA_BASE_DIR_DC"])
    logging.info(
        "DC train: synth=dc episodes=%s samples=%s n_proc=%s med_budget=%s td=%sns use_ct42=%s "
        "scales(delay/area/power)=%s/%s/%s",
        tk.get("num_episodes"), tk.get("num_samples"), tk.get("n_processing"),
        tk.get("med_budget"), tk.get("fixed_target_delay"), tk.get("use_ct42", False),
        tk["delay_scale"], tk["area_scale"], tk["power_scale"],
    )

    set_seed(seed)
    trainer_cls = getattr(trainer, cfg["trainer"]["name"])
    exp = trainer_cls(**tk)
    # MRED 接线（get_objective 用 getattr 读，默认 med 保持向后兼容）
    exp.error_metric = args.error_metric
    exp.mred_budget = args.mred_budget
    exp.mred_scale = args.mred_scale
    if args.error_metric == "mred":
        logging.info("ERROR METRIC = MRED | mred_budget=%s mred_scale=%s use_error_loss=False",
                     args.mred_budget, args.mred_scale)
        # 截断常数在构造函数的 _start_reset 里已按 med 口径算过(当时 error_metric 还未赋值)，
        # 清缓存显式重算，让 _setup_truncation 走 mred 分支取 C*（argmin E[|C−Δ|/p]）。
        if getattr(exp, "trunc_cols", 0) > 0:
            exp._trunc_bits = {}
            exp._setup_truncation()
    exp.run_experiment()
    rtl = exp.export_best_candidate(run_dir)
    logging.info("done. best RTL -> %s", rtl)


if __name__ == "__main__":
    main()
