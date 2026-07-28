#!/usr/bin/env python3
"""Train Stage-2 cell-only action-PPO on 32 fixed Stage-1 backbones."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_three_stage import BeijingFormatter, build_engine
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.pareto import environmental_select
from trainer.arith_three_stage.runner import ThreeStageConfig, ThreeStageRunner
from trainer.arith_three_stage.selection import select_banded
from trainer.arith_three_stage.stage2_action_ppo import (
    CellActionPPO,
    CellPPOConfig,
)


BEIJING = ZoneInfo("Asia/Shanghai")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_logging(out: Path) -> None:
    formatter = BeijingFormatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out / "train_stage2_action_ppo.log"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--backbones",
        type=Path,
        default=Path(
            "outputs/2026-07-21_1147_arith_stage2_extend240_compare_baseline/"
            "stage1/backbones_32.json"
        ),
    )
    parser.add_argument("--generations", type=int, default=640)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--offspring", type=int, default=128)
    parser.add_argument("--dc-batch", type=int, default=64)
    parser.add_argument("--dc-parallelism", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-delay", type=float, default=1.5)
    parser.add_argument("--error-vectors", type=int, default=16_000_000)
    parser.add_argument("--mred-lo", type=float, default=1e-7)
    parser.add_argument("--mred-hi", type=float, default=2e-1)
    parser.add_argument("--mred-budget-count", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-3)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-grad-clip", type=float, default=0.5)
    parser.add_argument("--ppo-exploration", type=float, default=0.05)
    parser.add_argument("--ppo-temperature", type=float, default=1.0)
    parser.add_argument("--ppo-init-approx-cells", type=float, default=4.0)
    parser.add_argument("--ppo-delay-weight", type=float, default=5.0)
    parser.add_argument("--ppo-mred-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--front-snapshot-every", type=int, default=5)
    parser.add_argument("--config", default="configs/config_groups/mul_16_approx_error_obj.yaml")
    parser.add_argument(
        "--base-dir-dc",
        default="/home/lchangxian/sandbox/sandbox_base_dcpwr",
    )
    parser.add_argument("--approx-col-window", type=int, default=6)
    parser.add_argument(
        "--approx-lib-path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    parser.add_argument(
        "--approx42-library-path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    parser.add_argument(
        "--approx42-rtl-path",
        default="Appr_Comp/rtl/comp42s_standalone.v",
    )
    args = parser.parse_args()
    if args.population != 128 or args.offspring != 128:
        parser.error("fair Stage2 comparison requires population=offspring=128")
    if args.dc_batch != 64 or args.dc_parallelism != 64:
        parser.error("requested comparison requires dc-batch=dc-parallelism=64")
    if args.device != "cpu":
        parser.error("cell logits run on CPU; --device must be cpu for safe DC fork")
    if args.generations < 1 or args.mred_budget_count < 1:
        parser.error("generations and mred-budget-count must be positive")
    return args


def engine_namespace(args):
    return argparse.Namespace(
        config=args.config,
        out=str(args.out),
        target_delay=args.target_delay,
        error_vectors=args.error_vectors,
        dc_batch=args.dc_batch,
        dc_parallelism=args.dc_parallelism,
        device=args.device,
        seed=args.seed,
        k_min=2,
        stage3_num_epochs=1,
        stage3_normalize_advantage=True,
        stage3_episodes_per_elite=1,
        approx_col_window=args.approx_col_window,
        approx_lib_path=args.approx_lib_path,
        approx42_library_path=args.approx42_library_path,
        approx42_rtl_path=args.approx42_rtl_path,
    )


def load_backbones(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    backbones = [Candidate.from_dict(item) for item in payload]
    if len(backbones) != 32:
        raise ValueError(f"expected 32 fixed backbones, got {len(backbones)}")
    if len({candidate.structure_hash for candidate in backbones}) != 32:
        raise ValueError("fixed backbone source contains duplicate structures")
    if any(candidate.cells for candidate in backbones):
        raise ValueError("Stage1 backbone source must not contain cell choices")
    if any(not candidate.evaluated for candidate in backbones):
        raise ValueError("all fixed backbones must have real evaluation anchors")
    return backbones


def seed_population(runner, backbones):
    population = []
    for backbone in backbones:
        population.extend(
            runner.cell_operator.make_seed_variants(backbone, runner.rng)
        )
    if len(population) != runner.cfg.population_size:
        raise RuntimeError(
            f"expected {runner.cfg.population_size} common seeds, "
            f"got {len(population)}"
        )
    return population


def append_metric(path: Path, payload: dict) -> None:
    record = dict(payload)
    record["timestamp_beijing"] = datetime.now(BEIJING).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_stage2(runner, backbones, ppo_config: CellPPOConfig):
    done = Path(runner.run_dir) / "stage2" / "elites_24.json"
    if done.exists():
        logging.info("Stage2 action-PPO already complete: %s", done)
        return
    searcher = CellActionPPO(
        runner.engine, runner.cfg, ppo_config, backbones
    )
    restored = runner._restore_generation(2)
    if restored is None:
        population = seed_population(runner, backbones)
        runner._evaluate_with_retries(population)
        if sum(candidate.evaluated for candidate in population) != len(population):
            raise RuntimeError("common Stage2 seeds have unresolved failures")
        seen_hashes = {candidate.cell_hash for candidate in population}
        population = environmental_select(
            population, runner.cfg.population_size, runner.cfg.delay_limit
        )
        archive = runner._new_archive(2)
        archive.update(population)
        start_generation = 0
        runner._save_generation(
            2,
            0,
            population,
            archive,
            runner.cell_bandit,
            extra_state={
                "search_mode": "action_ppo",
                "seen_cell_hashes": sorted(seen_hashes),
                "ppo_state": searcher.state_dict(),
            },
        )
    else:
        start_generation, population, archive = restored
        extra = runner._restored_extra_state.get(2, {})
        if extra.get("search_mode") != "action_ppo":
            raise RuntimeError("checkpoint does not belong to action-PPO")
        seen_hashes = set(extra.get("seen_cell_hashes") or ())
        seen_hashes.update(candidate.cell_hash for candidate in population)
        seen_hashes.update(candidate.cell_hash for candidate in archive.items)
        searcher.load_state_dict(extra["ppo_state"])
        logging.info(
            "resuming Stage2 action-PPO after generation %d, seen=%d",
            start_generation,
            len(seen_hashes),
        )

    metrics_path = Path(runner.run_dir) / "stage2" / "ppo_metrics.jsonl"
    for generation in range(
        int(start_generation) + 1,
        int(runner.cfg.stage2_generations) + 1,
    ):
        offspring = searcher.propose(
            backbones,
            size=runner.cfg.offspring_size,
            round_index=generation,
            excluded_hashes=seen_hashes,
        )
        runner._evaluate_with_retries(offspring)
        if sum(candidate.evaluated for candidate in offspring) != len(offspring):
            raise RuntimeError(
                f"generation {generation} has unresolved evaluation failures"
            )
        stats = searcher.update(offspring)
        seen_hashes.update(candidate.cell_hash for candidate in offspring)
        population = environmental_select(
            list(population) + offspring,
            runner.cfg.population_size,
            runner.cfg.delay_limit,
        )
        archive.update(offspring)
        append_metric(
            metrics_path,
            {
                "generation": generation,
                "archive_size": len(archive.items),
                "seen": len(seen_hashes),
                **stats,
            },
        )
        logging.info(
            "Stage2 action-PPO generation %d/%d complete: "
            "archive=%d seen=%d reward(mean=%.6g best=%.6g) feasible=%d/%d",
            generation,
            runner.cfg.stage2_generations,
            len(archive.items),
            len(seen_hashes),
            stats["reward_mean"],
            stats["reward_best"],
            stats["budget_feasible"],
            stats["samples"],
        )
        if generation % runner.cfg.checkpoint_every == 0:
            runner._save_generation(
                2,
                generation,
                population,
                archive,
                runner.cell_bandit,
                extra_state={
                    "search_mode": "action_ppo",
                    "seen_cell_hashes": sorted(seen_hashes),
                    "ppo_state": searcher.state_dict(),
                },
            )

    elites = select_banded(
        list(archive.items) + list(population),
        n_bins=runner.cfg.handoff_bins,
        roles=("area", "power", "knee"),
        mred_lo=runner.cfg.mred_lo,
        mred_hi=runner.cfg.mred_hi,
        delay_limit=runner.cfg.delay_limit,
    )
    atomic_json(done, [candidate.to_dict() for candidate in elites])
    logging.info("Stage2 action-PPO complete: selected %d elites", len(elites))


def main():
    args = parse_args()
    args.out = args.out.resolve()
    args.backbones = args.backbones.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_logging(args.out)
    os.environ["EDA_BASE_DIR_DC"] = args.base_dir_dc
    ppo_config = CellPPOConfig(
        epochs=args.ppo_epochs,
        learning_rate=args.ppo_learning_rate,
        clip_range=args.ppo_clip_range,
        grad_clip=args.ppo_grad_clip,
        exploration=args.ppo_exploration,
        temperature=args.ppo_temperature,
        init_approx_cells=args.ppo_init_approx_cells,
        delay_weight=args.ppo_delay_weight,
        mred_weight=args.ppo_mred_weight,
    )
    ppo_config.validate()
    config = ThreeStageConfig(
        population_size=args.population,
        offspring_size=args.offspring,
        dc_batch_size=args.dc_batch,
        dc_parallelism=args.dc_parallelism,
        delay_limit=args.target_delay,
        error_vectors=args.error_vectors,
        seed=args.seed,
        mred_lo=args.mred_lo,
        mred_hi=args.mred_hi,
        stage2_generations=args.generations,
        stage2_search_mode="cem",
        stage2_diffam_budget_count=args.mred_budget_count,
        checkpoint_every=args.checkpoint_every,
        front_snapshot_every=args.front_snapshot_every,
        stop_after_stage2=True,
    )
    engine = build_engine(engine_namespace(args))
    runner = ThreeStageRunner(engine, str(args.out), config)
    runner.cfg.stage2_search_mode = "action_ppo"
    backbones = load_backbones(args.backbones)
    atomic_json(
        args.out / "stage1" / "backbones_32.json",
        [candidate.to_dict() for candidate in backbones],
    )
    config_payload = asdict(runner.cfg)
    config_payload["stage1_backbones_source"] = str(args.backbones)
    config_payload["stage2_action_ppo"] = asdict(ppo_config)
    config_payload["timestamp_beijing"] = datetime.now(BEIJING).isoformat()
    atomic_json(args.out / "three_stage_config.json", config_payload)
    logging.info(
        "Stage2 action-PPO: backbones=%s generations=%d candidates/gen=%d "
        "dc_parallel=%d epochs=%d lr=%.4g clip=%.3f",
        args.backbones,
        args.generations,
        args.offspring,
        args.dc_parallelism,
        ppo_config.epochs,
        ppo_config.learning_rate,
        ppo_config.clip_range,
    )
    try:
        run_stage2(runner, backbones, ppo_config)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
