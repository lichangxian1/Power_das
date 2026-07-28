#!/usr/bin/env python3
"""EDA-free legality and selection smoke for Stage3 routing NSGA-II."""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage import Candidate
from trainer.arith_three_stage.evaluator import V5CandidateEvaluator
from trainer.arith_three_stage.pareto import environmental_select
from trainer.arith_three_stage.routing_nsga2 import RoutingNSGA2, route_digest


def engine_args(out):
    return argparse.Namespace(
        config="configs/config_groups/mul_16_approx_error_obj.yaml",
        out=out,
        target_delay=1.5,
        error_vectors=1000,
        dc_batch=64,
        dc_parallelism=1,
        device="cpu",
        seed=42,
        k_min=2,
        stage3_episodes_per_elite=2,
        stage3_num_epochs=1,
        stage3_normalize_advantage=False,
        stage3_single_elite_index=12,
        approx_col_window=6,
        approx_lib_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_library_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42s_standalone.v",
    )


def main():
    torch.manual_seed(42)
    source = Path(
        "outputs/2026-07-19_arith_three_stage_quiet_main_baseline/stage2/elites_24.json"
    )
    baseline = Candidate.from_dict(json.loads(source.read_text())[12])
    with tempfile.TemporaryDirectory(prefix="stage3-nsga2-") as tmp:
        engine = build_engine(engine_args(tmp))
        evaluator = V5CandidateEvaluator(
            engine, tmp, batch_size=64, n_processing=1,
            target_delay=1.5, error_vectors=1000,
        )
        evaluator._prepare(baseline)
        algorithm = RoutingNSGA2(engine, random.Random(42))
        logits = algorithm.initialize_logits(engine.get_Z_mat())
        seen = set()
        initial = algorithm.initial_population(logits, 12, seen)
        population = []
        for index, specification in enumerate(initial):
            candidate = baseline.clone(stage=3)
            candidate.routing = specification["connection"]
            candidate.refresh_id()
            candidate.area = 620.0 + index
            candidate.power = 0.0087 - index * 1e-6
            candidate.delay = 1.45 + (index % 3) * 0.01
            candidate.mred = 0.0008 + index * 1e-6
            evaluator._prepare(candidate)
            population.append(candidate)
        population = environmental_select(population, 8, 1.5)
        offspring = algorithm.make_offspring(
            population, logits, 16, seen, 1.5
        )
        assert len(offspring) == 16
        assert len({route_digest(item["connection"]) for item in offspring}) == 16
        assert len(seen) == 28
        expected_edges = len(initial[0]["connection"])
        for specification in offspring:
            assert len(specification["connection"]) == expected_edges
            probe = baseline.clone(stage=3)
            probe.routing = specification["connection"]
            evaluator._prepare(probe)
        evaluator.close()
        if getattr(engine, "tb_logger", None) is not None:
            engine.tb_logger.close()
    print(f"stage3 routing NSGA-II smoke: PASS edges={expected_edges} unique=28")


if __name__ == "__main__":
    main()
