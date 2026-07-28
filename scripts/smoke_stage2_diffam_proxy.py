#!/usr/bin/env python3
"""Small no-DC smoke for spawned proxy-DiffAM production."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.cell_ops import CellOperator
from trainer.arith_three_stage.diffam_pipeline import DiffAMProxyProducer
from trainer.arith_three_stage.runner import ThreeStageConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:2"), default="cuda:0")
    parser.add_argument("--backbones", required=True)
    args = parser.parse_args()
    run_dir = tempfile.mkdtemp(prefix="diffam_proxy_pipeline_", dir="/tmp")
    engine_args = argparse.Namespace(
        config="configs/config_groups/mul_16_approx_error_obj.yaml",
        target_delay=1.5,
        error_vectors=65_536,
        out=run_dir,
        dc_batch=64,
        stage3_num_epochs=1,
        dc_parallelism=1,
        device="cpu",
        seed=42,
        k_min=2,
        approx_col_window=6,
        approx_lib_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_library_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42s_standalone.v",
        stage3_normalize_advantage=True,
        stage3_episodes_per_elite=5,
    )
    engine = build_engine(engine_args)
    with open(os.path.abspath(args.backbones)) as stream:
        backbone = Candidate.from_dict(json.load(stream)[0])
    config = ThreeStageConfig(
        stage2_diffam_device=args.device,
        stage2_diffam_vectors=65_536,
        stage2_diffam_samples=1,
        stage2_diffam_dual_every=1,
        stage2_proxy_ensemble=2,
        stage2_proxy_min_samples=4,
        stage2_proxy_observation_cap=64,
        stage2_proxy_replay_samples=64,
        stage2_proxy_batch_size=4,
        stage2_proxy_epochs=1,
        stage2_proxy_diffam_steps=1,
    )
    observations = CellOperator(engine).make_seed_variants(
        backbone, random.Random(7)
    )
    for index, candidate in enumerate(observations):
        candidate.area = 900.0 - index
        candidate.power = 0.05 - index * 1e-4
        candidate.delay = 1.0 + index * 0.01
        candidate.mred = 1e-5 * (index + 1)
        candidate.valid = True
        candidate.metadata["generation"] = 0
    producer = DiffAMProxyProducer(config, run_dir, [backbone])
    started = time.monotonic()
    try:
        producer.request(
            size=2,
            round_index=1,
            excluded_hashes={candidate.cell_hash for candidate in observations},
            warm_starts=observations,
            observations=observations,
        )
        candidates, state = producer.receive()
        assert len(candidates) == 2
        assert state["proxy"]["trained_updates"] == 1
    finally:
        producer.close()
    print(
        "proxy DiffAM spawned pipeline smoke: PASS "
        f"seconds={time.monotonic() - started:.2f}"
    )


if __name__ == "__main__":
    main()
