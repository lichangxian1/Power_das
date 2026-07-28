 #!/usr/bin/env python3
"""Launch the resumable structure -> cell NSGA-II -> routing search."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from omegaconf import OmegaConf
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from trainer.arith_das_v5 import CompressorRouting  # noqa: E402
from trainer.arith_three_stage import Candidate, ThreeStageConfig, ThreeStageRunner  # noqa: E402


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class BeijingFormatter(logging.Formatter):
    """Render log timestamps in Beijing time regardless of the host timezone."""

    def formatTime(self, record, datefmt=None):
        timestamp = datetime.fromtimestamp(record.created, BEIJING_TZ)
        if datefmt:
            return timestamp.strftime(datefmt)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")


class ConciseTrainingFilter(logging.Filter):
    """Drop high-volume construction details while preserving run summaries."""

    _DROP_PREFIXES = ("remain_pp", "[trunc]", "[trunc-mred]")

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.getMessage()).startswith(self._DROP_PREFIXES)


def set_seed(seed: int, device: str = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_engine(args):
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    exp_kwargs = copy.deepcopy(cfg["experiment"]["kwargs"])
    kwargs = copy.deepcopy(cfg["trainer"]["kwargs"])
    kwargs.pop("area_budgets", None)
    kwargs.update(exp_kwargs)
    kwargs.update(
        {
            "synth": "dc",
            "power_source": "eda",
            "use_power_proxy": False,
            "fixed_target_delay": float(args.target_delay),
            "delay_weight": 0.0,
            "delay_as_constraint": True,
            "delay_target_ns": float(args.target_delay),
            "area_budget": None,
            "error_gate": "verilator",
            "error_gate_vectors": int(args.error_vectors),
            "log_dir": os.path.join(args.out, "policy_logs"),
            "build_dir": os.path.join(args.out, "engine_build"),
            "num_episodes": 1,
            "num_samples": int(args.dc_batch),
            "num_epochs": int(getattr(args, "stage3_num_epochs", 1)),
            "save_freq": 1,
            "front_dump_freq": 0,
            "n_processing": int(args.dc_parallelism),
            "n_full_target_delay_processing": int(args.dc_parallelism),
            "device": args.device,
            "seed": int(args.seed),
            "trunc_cols": int(args.k_min),
            "use_ct42": True,
            "outer_cell_search": True,
            "outer_bandit": False,
            "outer_crossover": False,
            "outer_multi_config": 1,
            "outer_zero_ops": True,
            "inject_exact_candidate": False,
            "approx_max_col": 30,
            "approx_col_window": int(args.approx_col_window),
            "approx_lib_path": args.approx_lib_path,
            "approx42_library_path": args.approx42_library_path,
            "approx42_rtl_path": args.approx42_rtl_path,
            "normalize_advantage": bool(args.stage3_normalize_advantage),
        }
    )
    stage3_elites = 1 if getattr(args, "stage3_single_elite_index", None) is not None else 24
    kwargs.setdefault("scheduler_kwargs", {})["T_max"] = max(
        1, int(args.stage3_episodes_per_elite) * stage3_elites
    )
    engine = CompressorRouting(**kwargs)
    engine.error_metric = "mred"
    engine.mred_budget = None
    engine.mred_scale = 0.01
    engine._trunc_profiles = {}
    engine._trunc_bits = {}
    engine._setup_truncation()
    engine._trunc_profiles[int(args.k_min)] = engine._capture_trunc_profile()
    return engine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config_groups/mul_16_approx_error_obj.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_delay", type=float, default=1.5)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--offspring", type=int, default=128)
    p.add_argument("--dc_batch", type=int, default=64)
    p.add_argument("--dc_parallelism", type=int, default=32)
    p.add_argument("--error_vectors", type=int, default=16_000_000)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=24)
    p.add_argument("--mred_lo", type=float, default=1e-7)
    p.add_argument("--mred_hi", type=float, default=2e-1)
    p.add_argument(
        "--stage1_backbones_source",
        help="existing Stage-1 backbones_32.json; skip Stage 1 and run later stages in a fresh directory",
    )
    p.add_argument("--stage1_generations", type=int, default=120)
    p.add_argument("--stage2_generations", type=int, default=120)
    p.add_argument(
        "--stage2_search_mode",
        choices=("ga", "diffam", "cem", "diffam_proxy"),
        default="ga",
    )
    p.add_argument(
        "--stage2_diffam_device",
        choices=("cpu", "cuda:0", "cuda:2"),
        help="DiffAM tensor device; defaults to --device",
    )
    p.add_argument("--stage2_diffam_vectors", type=int, default=16_000_000)
    p.add_argument("--stage2_diffam_vector_seed", type=int, default=12345)
    p.add_argument("--stage2_diffam_steps", type=int, default=40)
    p.add_argument("--stage2_diffam_budget_count", type=int, default=8)
    p.add_argument("--stage2_diffam_restarts", type=int, default=1)
    p.add_argument("--stage2_diffam_samples", type=int, default=8)
    p.add_argument("--stage2_diffam_lr", type=float, default=0.03)
    p.add_argument("--stage2_diffam_lam0", type=float, default=50.0)
    p.add_argument("--stage2_diffam_lam_step", type=float, default=100.0)
    p.add_argument("--stage2_diffam_dual_every", type=int, default=10)
    p.add_argument("--stage2_diffam_tau_min", type=float, default=0.25)
    p.add_argument("--stage2_diffam_init_std", type=float, default=0.70)
    p.add_argument("--stage2_diffam_exact_bias", type=float, default=0.80)
    p.add_argument("--stage2_diffam_warm_bias", type=float, default=2.0)
    p.add_argument("--stage2_cem_elite_fraction", type=float, default=0.20)
    p.add_argument("--stage2_cem_smoothing", type=float, default=0.25)
    p.add_argument("--stage2_cem_exploration", type=float, default=0.05)
    p.add_argument("--stage2_cem_temperature", type=float, default=1.0)
    p.add_argument("--stage2_cem_init_approx_cells", type=float, default=4.0)
    p.add_argument("--stage2_cem_history_per_structure", type=int, default=128)
    p.add_argument("--stage2_proxy_ensemble", type=int, default=3)
    p.add_argument("--stage2_proxy_min_samples", type=int, default=384)
    p.add_argument("--stage2_proxy_observation_cap", type=int, default=8192)
    p.add_argument("--stage2_proxy_replay_samples", type=int, default=4096)
    p.add_argument("--stage2_proxy_batch_size", type=int, default=256)
    p.add_argument("--stage2_proxy_epochs", type=int, default=4)
    p.add_argument("--stage2_proxy_lr", type=float, default=3e-4)
    p.add_argument("--stage2_proxy_weight_decay", type=float, default=1e-4)
    p.add_argument("--stage2_proxy_rank_weight", type=float, default=0.30)
    p.add_argument("--stage2_proxy_diffam_steps", type=int, default=8)
    p.add_argument("--stage2_proxy_diffam_lr", type=float, default=5e-3)
    p.add_argument("--stage2_proxy_tau_start", type=float, default=1.5)
    p.add_argument("--stage2_proxy_logit_noise", type=float, default=0.02)
    p.add_argument("--stage2_proxy_uncertainty_weight", type=float, default=0.25)
    p.add_argument("--stage2_proxy_nominal_area_weight", type=float, default=0.05)
    p.add_argument("--stage2_proxy_delay_weight", type=float, default=10.0)
    p.add_argument("--stage2_proxy_entropy_weight", type=float, default=0.02)
    p.add_argument("--stage3_episodes_per_elite", type=int, default=5)
    p.add_argument("--stage3_routes_per_episode", type=int, default=64)
    p.add_argument(
        "--stage3_num_epochs",
        type=int,
        default=1,
        help="PPO optimizer epochs over each fixed Stage-3 DC batch",
    )
    p.add_argument(
        "--stage3_ratio_mode",
        choices=("trajectory", "action"),
        default="trajectory",
        help="PPO importance ratio granularity for Stage 3",
    )
    p.add_argument(
        "--stage3_learning_rate",
        type=float,
        default=1e-4,
        help="Adam learning rate for Stage-3 PPO updates",
    )
    p.add_argument(
        "--stage3_normalize_advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize Stage-3 PPO advantages within each real-evaluated batch",
    )
    p.add_argument(
        "--stage3_policy_mode",
        choices=("ppo", "frozen", "random", "cem", "cem_reheat"),
        default="ppo",
        help=(
            "ppo updates the sampled policy; frozen samples the identical initial "
            "policy without updates; random samples uniformly over legal actions; "
            "cem updates direct routing logits from DC/Verilator-selected matchings; "
            "cem_reheat adds stagnation-triggered partial restart and temporary reheating"
        ),
    )
    p.add_argument(
        "--stage3_rule_loss_weight",
        type=float,
        default=0.0,
        help="weight of valid-source-normalized Stage-3 routing rule loss",
    )
    p.add_argument(
        "--stage3_discrete_loss_weight",
        type=float,
        default=0.0,
        help="maximum weight of valid-action-normalized Stage-3 discrete loss",
    )
    p.add_argument(
        "--stage3_discrete_start_fraction",
        type=float,
        default=0.5,
        help="episode fraction at which discrete-loss weight starts linear ramp-up",
    )
    p.add_argument(
        "--stage3_cem_elite_fraction",
        type=float,
        default=0.20,
        help="fraction of each real-evaluated Stage-3 batch used for CEM updates",
    )
    p.add_argument(
        "--stage3_cem_smoothing",
        type=float,
        default=0.25,
        help="CEM interpolation weight from old probabilities to elite frequencies",
    )
    p.add_argument(
        "--stage3_cem_exploration",
        type=float,
        default=0.05,
        help="uniform probability mixed into every CEM update",
    )
    p.add_argument(
        "--stage3_cem_temperature",
        type=float,
        default=1.0,
        help="logit temperature for Gumbel-Hungarian CEM sampling",
    )
    p.add_argument(
        "--stage3_cem_init",
        choices=("policy", "uniform"),
        default="policy",
        help="initialize direct CEM logits from the initial policy or uniformly",
    )
    p.add_argument("--stage3_cem_reheat_patience", type=int, default=30)
    p.add_argument(
        "--stage3_cem_reheat_entropy_threshold", type=float, default=0.25
    )
    p.add_argument("--stage3_cem_reheat_temperature", type=float, default=2.0)
    p.add_argument("--stage3_cem_reheat_episodes", type=int, default=10)
    p.add_argument("--stage3_cem_restart_fraction", type=float, default=0.30)
    p.add_argument(
        "--stage3_single_elite_source",
        help="Stage-2 elites_24.json to use for an isolated Stage-3-only ablation",
    )
    p.add_argument(
        "--stage3_single_elite_index",
        type=int,
        help="zero-based elite index in --stage3_single_elite_source",
    )
    p.add_argument("--front_snapshot_every", type=int, default=5)
    p.add_argument(
        "--stage1_archive_variants_per_objective", type=int, default=4,
        help="maximum structurally diverse variants per exact Stage 1 objective tuple",
    )
    p.add_argument(
        "--stage2_archive_variants_per_objective", type=int, default=2,
        help="maximum structurally diverse variants per exact Stage 2 objective tuple",
    )
    p.add_argument(
        "--stage3_archive_variants_per_objective", type=int, default=1,
        help="maximum structurally diverse variants per exact Stage 3 objective tuple",
    )
    p.add_argument(
        "--stage1_init_only",
        action="store_true",
        help=(
            "evaluate and save the 128-candidate Stage 1 generation-0 "
            "population, then stop before producing generation-1 offspring"
        ),
    )
    p.add_argument(
        "--stop_after_stage1",
        action="store_true",
        help="finish the requested Stage 1 generations, then stop before Stage 2",
    )
    p.add_argument(
        "--stop_after_stage2",
        action="store_true",
        help="finish the requested Stage 2 generations, then stop before Stage 3",
    )
    p.add_argument("--approx_col_window", type=int, default=6)
    p.add_argument(
        "--approx_lib_path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    p.add_argument(
        "--approx42_library_path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    p.add_argument(
        "--approx42_rtl_path",
        default="Appr_Comp/rtl/comp42s_standalone.v",
    )
    p.add_argument(
        "--base_dir_dc",
        default="/home/lchangxian/sandbox/sandbox_base_dcpwr",
    )
    args = p.parse_args()

    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    os.environ["EDA_BASE_DIR_DC"] = args.base_dir_dc
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(args.out, "train_three_stage.log")),
    ]
    concise_filter = ConciseTrainingFilter()
    formatter = BeijingFormatter("%(asctime)s %(levelname)s %(message)s")
    for handler in handlers:
        handler.addFilter(concise_filter)
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
    )
    if args.population != 128 or args.offspring != 128:
        raise SystemExit("正式三阶段实现当前固定 P=128, Q=128")
    isolated_stage3 = (
        args.stage3_single_elite_source is not None
        and args.stage3_single_elite_index is not None
    )
    if args.dc_batch != 64 and not (isolated_stage3 and args.dc_batch == 32):
        raise SystemExit(
            "完整三阶段固定每批64个候选；单点Stage3对照允许每批32或64个候选"
        )
    if not 1 <= args.dc_parallelism <= 64:
        raise SystemExit("--dc_parallelism 必须在 [1, 64] 内")
    if args.device.startswith("cuda:") and args.device not in ("cuda:0", "cuda:2"):
        raise SystemExit("GPU 只允许使用 cuda:0 或 cuda:2")
    if args.stage2_diffam_device is None:
        args.stage2_diffam_device = args.device
    if (
        args.stage2_diffam_device.startswith("cuda:")
        and args.stage2_diffam_device not in ("cuda:0", "cuda:2")
    ):
        raise SystemExit("Stage2 DiffAM GPU 只允许使用 cuda:0 或 cuda:2")
    if args.stage2_search_mode == "diffam_proxy" and args.device != "cpu":
        raise SystemExit(
            "diffam_proxy流水模式要求--device cpu，避免主进程CUDA上下文与"
            "DC fork冲突；代理/DiffAM仍由--stage2_diffam_device使用GPU"
        )
    if args.stage3_num_epochs < 1:
        raise SystemExit("--stage3_num_epochs 必须 >= 1")
    if args.stage3_learning_rate <= 0:
        raise SystemExit("--stage3_learning_rate 必须 > 0")
    if args.stage3_rule_loss_weight < 0:
        raise SystemExit("--stage3_rule_loss_weight 必须 >= 0")
    if args.stage3_discrete_loss_weight < 0:
        raise SystemExit("--stage3_discrete_loss_weight 必须 >= 0")
    if not 0 <= args.stage3_discrete_start_fraction <= 1:
        raise SystemExit("--stage3_discrete_start_fraction 必须在 [0, 1] 内")
    if not 0 < args.stage3_cem_elite_fraction <= 1:
        raise SystemExit("--stage3_cem_elite_fraction 必须在 (0, 1] 内")
    if not 0 < args.stage3_cem_smoothing <= 1:
        raise SystemExit("--stage3_cem_smoothing 必须在 (0, 1] 内")
    if not 0 <= args.stage3_cem_exploration < 1:
        raise SystemExit("--stage3_cem_exploration 必须在 [0, 1) 内")
    if args.stage3_cem_temperature <= 0:
        raise SystemExit("--stage3_cem_temperature 必须 > 0")
    if args.stage3_cem_reheat_patience < 1:
        raise SystemExit("--stage3_cem_reheat_patience 必须为正数")
    if not 0 <= args.stage3_cem_reheat_entropy_threshold <= 1:
        raise SystemExit("--stage3_cem_reheat_entropy_threshold 必须在 [0, 1] 内")
    if args.stage3_cem_reheat_temperature <= 0:
        raise SystemExit("--stage3_cem_reheat_temperature 必须 > 0")
    if args.stage3_cem_reheat_episodes < 1:
        raise SystemExit("--stage3_cem_reheat_episodes 必须为正数")
    if not 0 < args.stage3_cem_restart_fraction <= 1:
        raise SystemExit("--stage3_cem_restart_fraction 必须在 (0, 1] 内")
    if args.stage3_policy_mode != "ppo" and (
        args.stage3_rule_loss_weight > 0
        or args.stage3_discrete_loss_weight > 0
    ):
        raise SystemExit("非 PPO Stage 3 模式不能启用 rule/discrete 辅助损失")
    single_elite_mode = (
        args.stage3_single_elite_source is not None
        or args.stage3_single_elite_index is not None
    )
    if single_elite_mode and (
        args.stage3_single_elite_source is None
        or args.stage3_single_elite_index is None
    ):
        raise SystemExit(
            "单点 Stage 3 必须同时提供 --stage3_single_elite_source 和 "
            "--stage3_single_elite_index"
        )
    selected_elite = None
    source_path = None
    stage1_backbones_source = (
        os.path.abspath(args.stage1_backbones_source)
        if args.stage1_backbones_source is not None
        else None
    )
    if single_elite_mode:
        source_path = os.path.abspath(args.stage3_single_elite_source)
        with open(source_path) as f:
            source_elites = json.load(f)
        if len(source_elites) != 24:
            raise SystemExit(f"单点消融要求原始24点交接文件，实际得到 {len(source_elites)} 点")
        if not 0 <= args.stage3_single_elite_index < len(source_elites):
            raise SystemExit(
                f"elite index {args.stage3_single_elite_index} 越界；"
                f"合法范围 0..{len(source_elites) - 1}"
            )
        selected_elite = Candidate.from_dict(
            source_elites[args.stage3_single_elite_index]
        )
    logging.info("three-stage run directory: %s", args.out)
    logging.info("EDA_BASE_DIR_DC=%s", os.environ["EDA_BASE_DIR_DC"])
    set_seed(args.seed, args.device)
    engine = build_engine(args)
    search_cfg = ThreeStageConfig(
        population_size=args.population,
        offspring_size=args.offspring,
        dc_batch_size=args.dc_batch,
        dc_parallelism=args.dc_parallelism,
        delay_limit=args.target_delay,
        error_vectors=args.error_vectors,
        seed=args.seed,
        k_min=args.k_min,
        k_max=args.k_max,
        mred_lo=args.mred_lo,
        mred_hi=args.mred_hi,
        engine_config_path=os.path.abspath(args.config),
        approx_col_window=args.approx_col_window,
        approx_lib_path=os.path.abspath(args.approx_lib_path),
        approx42_library_path=os.path.abspath(args.approx42_library_path),
        approx42_rtl_path=os.path.abspath(args.approx42_rtl_path),
        stage1_backbones_source=stage1_backbones_source,
        stage1_generations=args.stage1_generations,
        stage2_generations=args.stage2_generations,
        stage2_search_mode=args.stage2_search_mode,
        stage2_diffam_device=args.stage2_diffam_device,
        stage2_diffam_vectors=args.stage2_diffam_vectors,
        stage2_diffam_vector_seed=args.stage2_diffam_vector_seed,
        stage2_diffam_steps=args.stage2_diffam_steps,
        stage2_diffam_budget_count=args.stage2_diffam_budget_count,
        stage2_diffam_restarts=args.stage2_diffam_restarts,
        stage2_diffam_samples=args.stage2_diffam_samples,
        stage2_diffam_lr=args.stage2_diffam_lr,
        stage2_diffam_lam0=args.stage2_diffam_lam0,
        stage2_diffam_lam_step=args.stage2_diffam_lam_step,
        stage2_diffam_dual_every=args.stage2_diffam_dual_every,
        stage2_diffam_tau_min=args.stage2_diffam_tau_min,
        stage2_diffam_init_std=args.stage2_diffam_init_std,
        stage2_diffam_exact_bias=args.stage2_diffam_exact_bias,
        stage2_diffam_warm_bias=args.stage2_diffam_warm_bias,
        stage2_cem_elite_fraction=args.stage2_cem_elite_fraction,
        stage2_cem_smoothing=args.stage2_cem_smoothing,
        stage2_cem_exploration=args.stage2_cem_exploration,
        stage2_cem_temperature=args.stage2_cem_temperature,
        stage2_cem_init_approx_cells=args.stage2_cem_init_approx_cells,
        stage2_cem_history_per_structure=args.stage2_cem_history_per_structure,
        stage2_proxy_ensemble=args.stage2_proxy_ensemble,
        stage2_proxy_min_samples=args.stage2_proxy_min_samples,
        stage2_proxy_observation_cap=args.stage2_proxy_observation_cap,
        stage2_proxy_replay_samples=args.stage2_proxy_replay_samples,
        stage2_proxy_batch_size=args.stage2_proxy_batch_size,
        stage2_proxy_epochs=args.stage2_proxy_epochs,
        stage2_proxy_lr=args.stage2_proxy_lr,
        stage2_proxy_weight_decay=args.stage2_proxy_weight_decay,
        stage2_proxy_rank_weight=args.stage2_proxy_rank_weight,
        stage2_proxy_diffam_steps=args.stage2_proxy_diffam_steps,
        stage2_proxy_diffam_lr=args.stage2_proxy_diffam_lr,
        stage2_proxy_tau_start=args.stage2_proxy_tau_start,
        stage2_proxy_logit_noise=args.stage2_proxy_logit_noise,
        stage2_proxy_uncertainty_weight=args.stage2_proxy_uncertainty_weight,
        stage2_proxy_nominal_area_weight=args.stage2_proxy_nominal_area_weight,
        stage2_proxy_delay_weight=args.stage2_proxy_delay_weight,
        stage2_proxy_entropy_weight=args.stage2_proxy_entropy_weight,
        stage3_elites=1 if single_elite_mode else 24,
        stage3_episodes_per_elite=args.stage3_episodes_per_elite,
        stage3_routes_per_episode=args.stage3_routes_per_episode,
        stage3_num_epochs=args.stage3_num_epochs,
        stage3_ratio_mode=args.stage3_ratio_mode,
        stage3_learning_rate=args.stage3_learning_rate,
        stage3_normalize_advantage=args.stage3_normalize_advantage,
        stage3_policy_mode=args.stage3_policy_mode,
        stage3_rule_loss_weight=args.stage3_rule_loss_weight,
        stage3_discrete_loss_weight=args.stage3_discrete_loss_weight,
        stage3_discrete_start_fraction=args.stage3_discrete_start_fraction,
        stage3_cem_elite_fraction=args.stage3_cem_elite_fraction,
        stage3_cem_smoothing=args.stage3_cem_smoothing,
        stage3_cem_exploration=args.stage3_cem_exploration,
        stage3_cem_temperature=args.stage3_cem_temperature,
        stage3_cem_init=args.stage3_cem_init,
        stage3_cem_reheat_patience=args.stage3_cem_reheat_patience,
        stage3_cem_reheat_entropy_threshold=args.stage3_cem_reheat_entropy_threshold,
        stage3_cem_reheat_temperature=args.stage3_cem_reheat_temperature,
        stage3_cem_reheat_episodes=args.stage3_cem_reheat_episodes,
        stage3_cem_restart_fraction=args.stage3_cem_restart_fraction,
        front_snapshot_every=args.front_snapshot_every,
        stage1_archive_variants_per_objective=args.stage1_archive_variants_per_objective,
        stage2_archive_variants_per_objective=args.stage2_archive_variants_per_objective,
        stage3_archive_variants_per_objective=args.stage3_archive_variants_per_objective,
        stage3_single_elite_source=source_path,
        stage3_single_elite_index=(
            args.stage3_single_elite_index if single_elite_mode else None
        ),
        stage3_single_elite_id=(
            selected_elite.candidate_id if selected_elite is not None else None
        ),
        stage1_init_only=args.stage1_init_only,
        stop_after_stage1=args.stop_after_stage1,
        stop_after_stage2=args.stop_after_stage2,
    )
    runner = ThreeStageRunner(engine, args.out, search_cfg)
    if selected_elite is not None:
        runner.run_stage3_only(selected_elite)
    else:
        runner.run()


if __name__ == "__main__":
    main()
