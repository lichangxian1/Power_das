#!/usr/bin/env python3
"""No-DC integration smoke test for the three-stage implementation."""
from __future__ import annotations

import argparse
import os
import random
import tempfile

import numpy as np

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage.bandit import ContextualThompsonBandit
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.cell_ops import CellOperator
from trainer.arith_three_stage.evaluator import V5CandidateEvaluator
from trainer.arith_three_stage.pareto import ExternalArchive, environmental_select
from trainer.arith_three_stage.structure_actions import StructureMutator
from utils import CompressorTree


def engine_args(out):
    return argparse.Namespace(
        config="configs/config_groups/mul_16_approx_error_obj.yaml",
        out=out,
        target_delay=1.5,
        error_vectors=1000,
        dc_batch=64,
        dc_parallelism=1,
        device="cpu",
        seed=7,
        k_min=2,
        stage3_episodes_per_elite=1,
        approx_col_window=6,
        approx_lib_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_library_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42s_standalone.v",
    )


def main():
    rng = random.Random(7)
    with tempfile.TemporaryDirectory(prefix="arith-three-stage-") as tmp:
        engine = build_engine(engine_args(tmp))
        pp = np.asarray(engine.initial_pp, dtype=int)
        dadda = CompressorTree.dadda(pp)
        base = Candidate(
            2,
            dadda.ct22.tolist(),
            dadda.ct32.tolist(),
            np.zeros_like(dadda.ct32).tolist(),
        )

        mutator = StructureMutator(engine, k_min=2, k_max=24)
        assert mutator.validate(base)
        arms = mutator.legal_arms(base)
        assert "classic" in arms and "boundary_k" in arms
        children = []
        for arm in arms:
            for _ in range(10):
                child = mutator.mutate(base, arm, rng)
                if child is not None:
                    assert mutator.validate(child)
                    children.append(child)
                    break
        assert children

        evaluator = V5CandidateEvaluator(
            engine, tmp, batch_size=64, n_processing=1,
            target_delay=1.5, error_vectors=1000,
        )
        _ct, connection, rtl = evaluator._prepare(base)
        assert connection and os.path.isfile(rtl)
        z_mat = engine.get_Z_mat()
        rule_raw = engine.get_rule_loss(z_mat)
        rule_norm = engine.get_rule_loss(z_mat, normalize=True)
        discrete_raw = engine.get_discrete_loss(z_mat)
        discrete_norm = engine.get_discrete_loss(z_mat, normalize=True)
        for loss in (rule_raw, rule_norm, discrete_raw, discrete_norm):
            assert np.isfinite(float(loss.item())) and float(loss.item()) >= 0.0
        assert float(rule_norm.item()) <= float(rule_raw.item())
        assert float(discrete_norm.item()) <= float(discrete_raw.item())

        cell_op = CellOperator(engine)
        seeds = cell_op.make_seed_variants(base, rng)
        assert len(seeds) == 4 and len(seeds[0].cells) == 0
        cross = cell_op.crossover_a(seeds[2], seeds[3], rng)
        mutated = []
        for arm in cell_op.legal_arms(cross):
            for _ in range(20):
                child = cell_op.mutate(cross, arm, rng)
                if child is not None:
                    mutated.append(child)
                    break
        assert mutated
        for child in mutated:
            evaluator._prepare(child)

        synthetic = []
        for i, c in enumerate((base, *children[:3])):
            x = c.clone()
            x.area = 100.0 + i
            x.power = 0.01 - i * 0.0005
            x.delay = 1.4
            x.mred = 1e-5 * (i + 1)
            synthetic.append(x)
        selected = environmental_select(synthetic, 3, 1.5)
        assert len(selected) == 3

        equivalent = []
        for k, delay in ((2, 1.0), (3, 1.1), (12, 1.4)):
            c = Candidate(k, [k], [0], [0], stage=1)
            c.area = 100.0
            c.power = 0.01
            c.delay = delay
            c.mred = 1e-5
            equivalent.append(c)

        diverse_archive = ExternalArchive(1.5, variants_per_objective=2)
        diverse_archive.items = equivalent
        diverse_archive.update(())
        assert len(diverse_archive.items) == 2
        assert min(c.delay for c in diverse_archive.items) == 1.0
        assert {c.k for c in diverse_archive.items} == {2, 12}

        singleton_archive = ExternalArchive(1.5, variants_per_objective=1)
        singleton_archive.update(equivalent)
        assert [c.delay for c in singleton_archive.items] == [1.0]

        bandit = ContextualThompsonBandit(["a", "b"], window=8, explore=0.0)
        arm = bandit.choose("ctx", ["a", "b"], rng)
        bandit.update("ctx", arm, True)
        assert bandit.stats("ctx", arm) == (1, 1)
        evaluator.close()
        if getattr(engine, "tb_logger", None) is not None:
            engine.tb_logger.close()
    print("three-stage smoke: PASS")


if __name__ == "__main__":
    main()
