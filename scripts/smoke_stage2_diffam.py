#!/usr/bin/env python3
"""No-DC smoke checks for fixed-structure Stage-2 DiffAM proposal."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.diffam_search import (
    DiffAMStage2Search,
    FrozenDiffAMProblem,
    _balanced_quota,
)
from trainer.arith_three_stage.runner import ThreeStageConfig
from utils import CompressorTree


class BaselineOrderingSolver:
    """Two-point proxy where an approximate cell beats the exact baseline."""

    @staticmethod
    def gate_mred(config):
        return 0.008 if config else 0.010

    gate_screen = gate_mred

    @staticmethod
    def area_saving(config):
        return 1.0 if config else 0.0


def _engine_args(out: str, device: str):
    return argparse.Namespace(
        config="configs/config_groups/mul_16_approx_error_obj.yaml",
        out=out,
        target_delay=1.5,
        error_vectors=1000,
        dc_batch=4,
        dc_parallelism=1,
        device=device,
        seed=17,
        k_min=2,
        stage3_num_epochs=1,
        stage3_episodes_per_elite=1,
        stage3_single_elite_index=None,
        stage3_normalize_advantage=True,
        approx_col_window=6,
        approx_lib_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_library_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42s_standalone.v",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("cpu", "cuda:0", "cuda:2"), default="cuda:0"
    )
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    remaining = 128
    quotas = []
    for backbone_index in range(32):
        quota = _balanced_quota(remaining, 32, backbone_index)
        quotas.append(quota)
        remaining -= quota
    assert quotas == [4] * 32 and remaining == 0
    if args.vectors < 65536:
        raise SystemExit(
            "--vectors must be at least 65536 for the stratified train batch"
        )

    with tempfile.TemporaryDirectory(prefix="stage2-diffam-smoke-") as tmp:
        engine = build_engine(_engine_args(tmp, args.device))
        pp = np.asarray(engine.initial_pp, dtype=int)
        dadda = CompressorTree.dadda(pp)
        backbone = Candidate(
            2,
            dadda.ct22.tolist(),
            dadda.ct32.tolist(),
            np.zeros_like(dadda.ct32).tolist(),
            stage=1,
            operator="diffam_smoke_backbone",
        )
        config = ThreeStageConfig(
            population_size=4,
            offspring_size=4,
            dc_batch_size=4,
            dc_parallelism=1,
            seed=17,
            mred_lo=1e-5,
            mred_hi=1e-2,
            stage2_search_mode="diffam",
            stage2_diffam_device=args.device,
            stage2_diffam_vectors=args.vectors,
            stage2_diffam_steps=args.steps,
            stage2_diffam_budget_count=3,
            stage2_diffam_restarts=1,
            stage2_diffam_samples=2,
            stage2_diffam_dual_every=1,
        )
        search = DiffAMStage2Search(engine, config, tmp)
        baseline_context = FrozenDiffAMProblem(
            backbone=backbone,
            graph=SimpleNamespace(vertex_list=[(0, 2, 0, 0)]),
            tree=None,
            pp_specs={},
            solver=BaselineOrderingSolver(),
        )
        below_baseline = search._select(
            baseline_context,
            raw=[({0: (0, 1)}, {"source": "cancellation_probe"})],
            budget=0.005,
            size=1,
            excluded_hashes=set(),
        )
        assert below_baseline[0].cells == [[0, 2, 0, 0, 1]]

        round_zero = search.propose([backbone], size=4, round_index=0)
        seen = {candidate.cell_hash for candidate in round_zero}
        round_one = search.propose(
            [backbone],
            size=4,
            round_index=1,
            excluded_hashes=seen,
            warm_starts=round_zero,
        )
        assert len(round_zero) == len(seen) == 4
        assert len({candidate.cell_hash for candidate in round_one}) == 4
        assert not seen.intersection(candidate.cell_hash for candidate in round_one)
        for candidate in [*round_zero, *round_one]:
            assert candidate.structure_hash == backbone.structure_hash
            assert candidate.routing is None
            assert candidate.operator == "diffam_ste"

        if getattr(engine, "tb_logger", None) is not None:
            engine.tb_logger.close()
    print("stage2 DiffAM smoke: PASS " "(two unseen fixed-structure proposal rounds)")


if __name__ == "__main__":
    main()
