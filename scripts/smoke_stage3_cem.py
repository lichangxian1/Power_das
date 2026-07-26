#!/usr/bin/env python3
"""EDA-free legality, update, diversity, and serialization smoke for Stage-3 CEM."""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage.candidate import Candidate
from trainer.arith_three_stage.cem import RoutingCEM
from trainer.arith_three_stage.evaluator import V5CandidateEvaluator
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
        seed=17,
        k_min=2,
        stage3_episodes_per_elite=1,
        stage3_num_epochs=1,
        approx_col_window=6,
        approx_lib_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_library_path="Appr_Comp/selected_compressors_all_substd.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42s_standalone.v",
    )


def signature(connection):
    return tuple(
        (int(src), int(dst), int(port), str(meta["src_output"]))
        for src, dst, port, meta in connection
    )


def main():
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory(prefix="stage3-cem-") as tmp:
        engine = build_engine(engine_args(tmp))
        pp = np.asarray(engine.initial_pp, dtype=int)
        dadda = CompressorTree.dadda(pp)
        candidate = Candidate(
            2,
            dadda.ct22.tolist(),
            dadda.ct32.tolist(),
            np.zeros_like(dadda.ct32).tolist(),
        )
        evaluator = V5CandidateEvaluator(
            engine,
            tmp,
            batch_size=64,
            n_processing=1,
            target_delay=1.5,
            error_vectors=1000,
        )
        evaluator._prepare(candidate)
        template = engine.get_Z_mat()
        cem = RoutingCEM(
            engine,
            smoothing=0.5,
            exploration=0.1,
            temperature=1.0,
            init_mode="policy",
        )
        logits = cem.initialize(template)
        before = cem.stats(logits)
        samples = cem.sample_many(logits, 12)
        assert len({signature(connection) for connection, _score in samples}) == 12

        expected_edges = sum(
            len(engine.comp_graph.slice_indice_map[(stage - 1, column)])
            + (
                len(engine.comp_graph.get_slice_carry_sources(stage, column))
                if column > 0
                else 0
            )
            for stage, column in logits
        )
        for connection, _score in samples:
            assert len(connection) == expected_edges
            per_slice_rows = {}
            per_slice_slots = {}
            for _src, _dst, _port, meta in connection:
                key = tuple(meta["slice"])
                per_slice_rows.setdefault(key, set()).add(int(meta["flat_row"]))
                per_slice_slots.setdefault(key, set()).add(int(meta["sample"]))
            assert all(
                len(per_slice_rows[key]) == len(per_slice_slots[key])
                for key in per_slice_rows
            )
            probe = candidate.clone(stage=3)
            probe.routing = connection
            evaluator._prepare(probe)

        updated, after = cem.update(
            logits, [connection for connection, _score in samples[:4]]
        )
        assert after["sources"] == before["sources"] == expected_edges
        assert any(
            not torch.equal(logits[key][port], updated[key][port])
            for key in logits
            for port in logits[key]
        )
        assert 0.0 <= after["normalized_entropy"] <= 1.0 + 1.0e-6
        assert 0.0 < after["mean_max_probability"] <= 1.0

        buffer = io.BytesIO()
        torch.save({"cem_logits": updated}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
        assert restored["cem_logits"].keys() == updated.keys()

        evaluator.close()
        if getattr(engine, "tb_logger", None) is not None:
            engine.tb_logger.close()
    print(
        "stage3 CEM smoke: PASS "
        f"edges={expected_edges} "
        f"entropy={before['normalized_entropy']:.4f}"
        f"->{after['normalized_entropy']:.4f}"
    )


if __name__ == "__main__":
    main()
