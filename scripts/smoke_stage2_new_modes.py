#!/usr/bin/env python3
"""Two-backbone, no-DC end-to-end smoke for Stage-2 CEM/proxy DiffAM."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.runner import ThreeStageConfig, ThreeStageRunner


def _fake_evaluate(self, candidates):
    # Mirror the production evaluator's fork pool while the spawned GPU
    # producer may be active.
    with multiprocessing.get_context("fork").Pool(2) as pool:
        pool.map(abs, (-1, -2, -3, -4))
    for candidate in candidates:
        fraction = int(
            hashlib.sha256(candidate.cell_hash.encode()).hexdigest()[:8], 16
        ) / float(0xFFFFFFFF)
        candidate.area = 1000.0 - 100.0 * fraction
        candidate.power = 0.04 + 0.01 * fraction
        candidate.delay = 1.0
        candidate.mred = 10.0 ** (-7.0 + 6.0 * fraction)
        candidate.valid = True
        candidate.failure_reason = None
    return list(candidates)


def _engine(run_dir: str):
    return build_engine(
        argparse.Namespace(
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
    )


def _run(mode: str, backbones, device: str):
    run_dir = tempfile.mkdtemp(prefix=f"stage2_{mode}_e2e_", dir="/tmp")
    config = ThreeStageConfig(
        population_size=8,
        offspring_size=8,
        dc_parallelism=1,
        error_vectors=65_536,
        stage2_generations=2,
        stage2_search_mode=mode,
        stage2_diffam_device=device,
        stage2_diffam_vectors=65_536,
        stage2_diffam_samples=1,
        stage2_diffam_dual_every=1,
        stage2_proxy_ensemble=2,
        stage2_proxy_min_samples=8,
        stage2_proxy_observation_cap=64,
        stage2_proxy_replay_samples=64,
        stage2_proxy_batch_size=8,
        stage2_proxy_epochs=1,
        stage2_proxy_diffam_steps=1,
        stop_after_stage2=True,
    )
    runner = ThreeStageRunner(_engine(run_dir), run_dir, config)
    runner._evaluate_with_retries = types.MethodType(_fake_evaluate, runner)
    try:
        runner._claim_stage2_search_mode()
        elites = runner._load_or_run_stage2(backbones)
        assert len(elites) == 24
        with open(os.path.join(run_dir, "stage2", "checkpoint.pt"), "rb"):
            pass
    finally:
        runner.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", required=True)
    parser.add_argument(
        "--device", choices=("cpu", "cuda:0", "cuda:2"), default="cuda:0"
    )
    args = parser.parse_args()
    with open(os.path.abspath(args.backbones)) as stream:
        backbones = [
            Candidate.from_dict(payload) for payload in json.load(stream)[:2]
        ]
    cem_dir = _run("cem", backbones, "cpu")
    proxy_dir = _run("diffam_proxy", backbones, args.device)
    print(
        "Stage-2 new-mode end-to-end smoke: PASS "
        f"cem={cem_dir} diffam_proxy={proxy_dir}"
    )


if __name__ == "__main__":
    main()
