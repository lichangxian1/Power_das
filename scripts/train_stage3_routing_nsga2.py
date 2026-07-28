#!/usr/bin/env python3
"""Run fixed-e12 routing NSGA-II with real DC and Verilator objectives."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_three_stage import BeijingFormatter, build_engine, set_seed
from trainer.arith_three_stage import Candidate, ThreeStageConfig, ThreeStageRunner
from trainer.arith_three_stage.pareto import environmental_select
from trainer.arith_three_stage.routing_nsga2 import RoutingNSGA2


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def configure_logging(out: Path) -> None:
    formatter = BeijingFormatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out / "train_nsga2.log"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/2026-07-19_arith_three_stage_quiet_main_baseline/stage2/elites_24.json"),
    )
    parser.add_argument("--elite-index", type=int, default=12)
    parser.add_argument("--generations", type=int, default=360)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--offspring", type=int, default=64)
    parser.add_argument("--dc-parallelism", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-delay", type=float, default=1.5)
    parser.add_argument("--error-vectors", type=int, default=16_000_000)
    parser.add_argument("--crossover-probability", type=float, default=0.70)
    parser.add_argument("--crossover-blocks", type=int, default=8)
    parser.add_argument("--mutation-blocks", type=int, default=4)
    parser.add_argument("--immigrants", type=int, default=6)
    parser.add_argument("--donor-temperature", type=float, default=1.25)
    parser.add_argument("--front-snapshot-every", type=int, default=5)
    parser.add_argument("--config", default="configs/config_groups/mul_16_approx_error_obj.yaml")
    parser.add_argument("--base-dir-dc", default="/home/lchangxian/sandbox/sandbox_base_dcpwr")
    parser.add_argument("--approx-col-window", type=int, default=6)
    parser.add_argument("--approx-lib-path", default="Appr_Comp/selected_compressors_all_substd.json")
    parser.add_argument("--approx42-library-path", default="Appr_Comp/selected_compressors_all_substd.json")
    parser.add_argument("--approx42-rtl-path", default="Appr_Comp/rtl/comp42s_standalone.v")
    return parser.parse_args()


def engine_namespace(args):
    return argparse.Namespace(
        config=args.config,
        out=str(args.out),
        target_delay=args.target_delay,
        error_vectors=args.error_vectors,
        dc_batch=args.offspring,
        dc_parallelism=args.dc_parallelism,
        device=args.device,
        seed=args.seed,
        k_min=2,
        stage3_episodes_per_elite=args.generations,
        stage3_num_epochs=1,
        stage3_normalize_advantage=False,
        stage3_single_elite_index=args.elite_index,
        approx_col_window=args.approx_col_window,
        approx_lib_path=args.approx_lib_path,
        approx42_library_path=args.approx42_library_path,
        approx42_rtl_path=args.approx42_rtl_path,
    )


def load_baseline(path: Path, index: int) -> Candidate:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= index < len(items):
        raise SystemExit(f"elite index {index} is outside 0..{len(items)-1}")
    return Candidate.from_dict(items[index])


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.source = args.source.resolve()
    if args.population < 2 or args.offspring < 2:
        raise SystemExit("population and offspring must both be at least two")
    if args.offspring != 64:
        raise SystemExit("the fair e12 comparison requires exactly 64 real offspring per generation")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["EDA_BASE_DIR_DC"] = args.base_dir_dc
    configure_logging(args.out)
    set_seed(args.seed, args.device)
    baseline = load_baseline(args.source, args.elite_index)
    atomic_json(
        args.out / "nsga2_config.json",
        {
            "created_at_beijing": datetime.now(BEIJING_TZ).isoformat(),
            "source": str(args.source),
            "elite_index": args.elite_index,
            "candidate_id": baseline.candidate_id,
            "selection_role": baseline.metadata.get("selection_role", "area"),
            "generations": args.generations,
            "population": args.population,
            "offspring": args.offspring,
            "dc_parallelism": args.dc_parallelism,
            "device": args.device,
            "seed": args.seed,
            "target_delay": args.target_delay,
            "error_vectors": args.error_vectors,
            "crossover_probability": args.crossover_probability,
            "crossover_blocks_max": args.crossover_blocks,
            "mutation_blocks_max": args.mutation_blocks,
            "random_immigrants": args.immigrants,
            "donor_temperature": args.donor_temperature,
            "selection": "constrained NSGA-II over area,power,mred; delay<=limit",
            "scalar_visual_objective": "same negative Stage3 routing reward as PPO/CEM",
        },
    )

    engine = build_engine(engine_namespace(args))
    config = ThreeStageConfig(
        dc_batch_size=args.offspring,
        dc_parallelism=args.dc_parallelism,
        delay_limit=args.target_delay,
        error_vectors=args.error_vectors,
        seed=args.seed,
        stage3_elites=1,
        stage3_episodes_per_elite=args.generations,
        stage3_routes_per_episode=args.offspring,
        stage3_num_epochs=1,
        stage3_ratio_mode="trajectory",
        stage3_learning_rate=1e-4,
        stage3_normalize_advantage=False,
        stage3_policy_mode="nsga2",
        front_snapshot_every=args.front_snapshot_every,
        stage3_single_elite_source=str(args.source),
        stage3_single_elite_index=args.elite_index,
        stage3_single_elite_id=baseline.candidate_id,
    )
    runner = ThreeStageRunner(engine, str(args.out), config)
    atomic_json(args.out / "stage2" / "source_elite.json", [baseline.to_dict()])

    checkpoint_path = args.out / "stage3" / "nsga2_checkpoint.pt"
    metrics_path = args.out / "routing_metrics.jsonl"
    population_path = args.out / "stage3" / "population.json"
    archive_path = args.out / "stage3" / "archive.json"
    archive = runner._new_archive(3)
    archive.update([baseline])
    population = []
    seen = set()
    records = []
    start = 0
    cumulative_best = math.inf

    try:
        runner.evaluator._prepare(baseline)
        with torch.no_grad():
            template = engine.get_Z_mat()
        algorithm = RoutingNSGA2(
            engine,
            runner.rng,
            crossover_probability=args.crossover_probability,
            crossover_blocks=args.crossover_blocks,
            mutation_blocks=args.mutation_blocks,
            immigrants=args.immigrants,
            donor_temperature=args.donor_temperature,
        )
        logits = algorithm.initialize_logits(template)

        if checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location=engine.device, weights_only=False)
            if state["baseline_id"] != baseline.candidate_id:
                raise ValueError("NSGA-II checkpoint baseline does not match requested e12")
            start = int(state["generation"])
            population = [Candidate.from_dict(item) for item in state["population"]]
            archive.items = [Candidate.from_dict(item) for item in state["archive"]]
            archive.update(())
            seen = set(state["seen"])
            records = list(state.get("records", []))
            cumulative_best = float(state.get("cumulative_best", math.inf))
            runner.rng.setstate(state["rng_state"])
            torch.set_rng_state(state["torch_rng_state"].detach().cpu())
            logging.info("resuming routing NSGA-II at generation %d/%d", start, args.generations)

        runner._save_front_snapshot(3, start, archive.items)
        for generation in range(start, args.generations):
            runner.evaluator._prepare(baseline)
            with torch.no_grad():
                if population:
                    specifications = algorithm.make_offspring(
                        population,
                        logits,
                        args.offspring,
                        seen,
                        args.target_delay,
                    )
                else:
                    specifications = algorithm.initial_population(
                        logits, args.offspring, seen
                    )

            candidates = []
            for specification in specifications:
                candidate = baseline.clone(stage=3)
                candidate.routing = specification["connection"]
                candidate.parent_ids = list(specification["parent_ids"])
                candidate.operator = specification["operator"]
                candidate.operator_context = (
                    f"x{specification['crossover_blocks']}|"
                    f"m{specification['mutation_blocks']}"
                )
                candidate.metadata.update(
                    {
                        "baseline_id": baseline.candidate_id,
                        "selection_role": baseline.metadata.get("selection_role", "area"),
                        "mred_band_edges": baseline.metadata.get("mred_band_edges"),
                        "stage3_policy_mode": "nsga2",
                        "stage3_episode": generation + 1,
                        "stage3_schedule_index": generation + 1,
                        "route_digest": specification["digest"],
                        "crossover_blocks": specification["crossover_blocks"],
                        "mutation_blocks": specification["mutation_blocks"],
                    }
                )
                candidate.refresh_id()
                candidates.append(candidate)

            runner._evaluate_with_retries(candidates)
            usable = [candidate for candidate in candidates if candidate.evaluated]
            if len(usable) < 2:
                raise RuntimeError(
                    f"NSGA-II generation {generation + 1} has only {len(usable)} usable routes"
                )
            population = environmental_select(
                list(population) + usable, args.population, args.target_delay
            )
            archive.update(usable)

            objectives = [runner._routing_objective(item, baseline) for item in usable]
            batch_best = float(min(objectives))
            cumulative_best = min(cumulative_best, batch_best)
            record = {
                "timestamp_beijing": datetime.now(BEIJING_TZ).isoformat(),
                "episode": generation + 1,
                "generation": generation + 1,
                "evaluated": len(usable),
                "mean_objective": float(np.mean(objectives)),
                "batch_best_objective": batch_best,
                "cumulative_best_objective": float(cumulative_best),
                "population_size": len(population),
                "pareto_archive_size": len(archive.items),
                "feasible_population": sum(
                    float(item.delay) <= args.target_delay for item in population
                ),
                "genetic_offspring": sum(
                    item.operator == "nsga2_genetic" for item in usable
                ),
                "immigrants": sum(item.operator == "nsga2_immigrant" for item in usable),
            }
            records.append(record)
            logging.info(
                "Stage 3 NSGA-II %d/%d usable=%d population=%d archive=%d "
                "objective(mean=%.6g batch_best=%.6g cumulative_best=%.6g)",
                generation + 1,
                args.generations,
                len(usable),
                len(population),
                len(archive.items),
                record["mean_objective"],
                batch_best,
                cumulative_best,
            )

            runner._save_front_snapshot(3, generation + 1, archive.items)
            atomic_json(population_path, [item.to_dict() for item in population])
            atomic_json(archive_path, archive.to_list())
            atomic_torch(
                checkpoint_path,
                {
                    "generation": generation + 1,
                    "baseline_id": baseline.candidate_id,
                    "population": [item.to_dict() for item in population],
                    "archive": archive.to_list(),
                    "seen": sorted(seen),
                    "records": records,
                    "cumulative_best": cumulative_best,
                    "rng_state": runner.rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "config": vars(args),
                },
            )
            atomic_jsonl(metrics_path, records)

        runner._export_final(archive.items)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
