#!/usr/bin/env python3
"""Bit-exact cross-check of Stage-2 hard tensor simulation against emitted RTL."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Appr_Comp.cellsolver import sim as diff_sim  # noqa: E402
from scripts.compare_stage2_cellonly_diffam import build_frozen_context  # noqa: E402
from scripts.train_three_stage import build_engine  # noqa: E402
from trainer.arith_three_stage.candidate import Candidate  # noqa: E402
from trainer.arith_three_stage.cell_ops import CellOperator  # noqa: E402
from trainer.arith_three_stage.evaluator import V5CandidateEvaluator  # noqa: E402

CellConfig = Dict[int, Tuple[int, int]]


def _engine_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        out=args.out,
        device=args.device,
        seed=args.seed,
        target_delay=1.5,
        error_vectors=args.vectors,
        k_min=2,
        dc_batch=1,
        dc_parallelism=1,
        stage3_num_epochs=1,
        stage3_episodes_per_elite=1,
        stage3_single_elite_index=None,
        stage3_normalize_advantage=True,
        approx_col_window=args.approx_col_window,
        approx_lib_path=args.approx_lib_path,
        approx42_library_path=args.approx42_library_path,
        approx42_rtl_path=args.approx42_rtl_path,
    )


def _context_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        device=args.device,
        mred_budgets=[args.mred_budget],
        diffam_vectors=max(args.vectors, 32768),
        vector_seed=args.vector_seed,
        out=args.out,
    )


def _cell_config(candidate: Candidate, graph) -> CellConfig:
    config: CellConfig = {}
    for entry in candidate.cells:
        slot = tuple(int(value) for value in entry[:4])
        node = graph.indice_map.get(slot)
        if node is None:
            raise RuntimeError(f"cell slot missing from frozen graph: {slot}")
        config[int(node)] = (int(entry[2]), int(entry[4]))
    return config


def _make_candidates(
    backbone: Candidate,
    operator: CellOperator,
    *,
    count: int,
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    exact = backbone.clone(stage=2)
    exact.cells = []
    exact.operator = "semantic_exact"
    exact.refresh_id()
    candidates = [exact]
    seen = {exact.cell_hash}
    targets = (1, 2, 4, 8, 12, 16, 24)
    attempts = 0
    while len(candidates) < count and attempts < 10000:
        attempts += 1
        candidate = backbone.clone(stage=2)
        candidate.cells = []
        target = targets[(len(candidates) - 1) % len(targets)]
        target = max(1, target + rng.choice((-1, 0, 0, 1)))
        for _ in range(target):
            if rng.random() < 0.20 and operator._zero_toggle(candidate, rng):
                continue
            if not operator._add(candidate, rng, prefer_low=rng.random() < 0.5):
                break
        candidate.cells = sorted(candidate.cells, key=lambda item: tuple(item))
        candidate.operator = "semantic_random"
        candidate.refresh_id()
        if candidate.cell_hash in seen:
            continue
        seen.add(candidate.cell_hash)
        candidates.append(candidate)
    if len(candidates) != count:
        raise RuntimeError(
            f"generated only {len(candidates)}/{count} unique semantic candidates"
        )
    return candidates


def _rtl_trace(rtl_path: str, build_dir: str, vectors: int) -> np.ndarray:
    harness = os.path.join(_ROOT, "verilate", "mul_trace_wrap.cpp")
    obj_dir = os.path.join(build_dir, "obj_dir")
    executable = os.path.join(obj_dir, "mul_trace")
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)
    command = [
        "verilator",
        "--cc",
        "--exe",
        "--build",
        "-j",
        "1",
        "-O3",
        "-Wno-fatal",
        "--top-module",
        "MUL",
        "--Mdir",
        obj_dir,
        os.path.abspath(rtl_path),
        harness,
        "-o",
        "mul_trace",
    ]
    built = subprocess.run(
        command,
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if built.returncode != 0 or not os.path.isfile(executable):
        raise RuntimeError(
            f"verilator build failed rc={built.returncode}\n"
            f"stdout:\n{built.stdout[-4000:]}\nstderr:\n{built.stderr[-4000:]}"
        )
    traced = subprocess.run(
        [executable, str(int(vectors))],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if traced.returncode != 0:
        raise RuntimeError(
            f"verilator trace failed rc={traced.returncode}\n"
            f"stdout:\n{traced.stdout[-4000:]}\nstderr:\n{traced.stderr[-4000:]}"
        )
    rows = []
    for line in traced.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 4:
            raise RuntimeError(f"malformed trace row: {line!r}")
        rows.append(tuple(int(value) for value in fields))
    result = np.asarray(rows, dtype=np.int64)
    if result.shape != (vectors, 4):
        raise RuntimeError(
            f"trace shape mismatch: {result.shape}, expected {(vectors, 4)}"
        )
    if not np.array_equal(result[:, 0], np.arange(vectors, dtype=np.int64)):
        raise RuntimeError("verilator trace indices are not contiguous")
    return result


def _atomic_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", required=True)
    parser.add_argument("--backbone_index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--config",
        default="configs/config_groups/mul_16_approx_error_obj.yaml",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda:0", "cuda:2"), default="cuda:0"
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--vector_seed", type=int, default=12345)
    parser.add_argument("--vectors", type=int, default=4096)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--mred_budget", type=float, default=1e-2)
    parser.add_argument("--approx_col_window", type=int, default=6)
    parser.add_argument(
        "--approx_lib_path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    parser.add_argument(
        "--approx42_library_path",
        default="Appr_Comp/selected_compressors_all_substd.json",
    )
    parser.add_argument(
        "--approx42_rtl_path",
        default="Appr_Comp/rtl/comp42s_standalone.v",
    )
    args = parser.parse_args()
    if args.vector_seed != 12345:
        raise SystemExit("the RTL trace harness currently requires --vector_seed=12345")
    if args.vectors < 1 or args.candidates < 2:
        raise SystemExit(
            "--vectors must be positive and --candidates must be at least 2"
        )
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    with open(args.backbones) as stream:
        backbone_rows = json.load(stream)
    if not 0 <= args.backbone_index < len(backbone_rows):
        raise IndexError(
            f"backbone index {args.backbone_index} outside 0..{len(backbone_rows) - 1}"
        )
    backbone = Candidate.from_dict(backbone_rows[args.backbone_index])
    engine = build_engine(_engine_args(args))
    context = build_frozen_context(engine, backbone, _context_args(args))
    evaluator = V5CandidateEvaluator(
        engine,
        os.path.join(args.out, "rtl_emit"),
        batch_size=1,
        n_processing=1,
        target_delay=1.5,
        error_vectors=args.vectors,
    )
    operator = CellOperator(engine)
    candidates = _make_candidates(
        backbone,
        operator,
        count=args.candidates,
        seed=args.seed,
    )
    a_values, b_values = diff_sim.xorshift_ab(
        args.vectors,
        seed=args.vector_seed,
        cache_dir=os.path.join(args.out, "vector_cache"),
    )
    pp_bits = diff_sim.compute_pp_bits(
        context.pp_specs,
        a_values,
        b_values,
        engine.bit_width,
        args.device,
    )

    results = []
    try:
        for index, candidate in enumerate(candidates):
            _ct, _connection, rtl_path = evaluator._prepare(candidate)
            config = _cell_config(candidate, context.graph)
            tensor_out = (
                (
                    context.tree.eval_exact(
                        pp_bits,
                        context.base_solver.space.cell_luts_of(config),
                    )
                    & diff_sim.MASK31
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            trace = _rtl_trace(
                rtl_path,
                os.path.join(args.out, "verilator", f"candidate_{index:02d}"),
                args.vectors,
            )
            if not np.array_equal(trace[:, 1], a_values.astype(np.int64)):
                raise AssertionError(f"candidate {index}: input-a stream mismatch")
            if not np.array_equal(trace[:, 2], b_values.astype(np.int64)):
                raise AssertionError(f"candidate {index}: input-b stream mismatch")
            mismatch = np.flatnonzero(trace[:, 3] != tensor_out)
            row = {
                "candidate_index": index,
                "candidate_id": candidate.candidate_id,
                "cell_hash": candidate.cell_hash,
                "cells": candidate.cells,
                "n_cells": len(candidate.cells),
                "vectors": args.vectors,
                "mismatch_count": int(mismatch.size),
            }
            results.append(row)
            print(
                f"[{index + 1}/{len(candidates)}] cells={len(candidate.cells):2d} "
                f"mismatches={mismatch.size}"
            )
            if mismatch.size:
                first = int(mismatch[0])
                row["first_mismatch"] = {
                    "index": first,
                    "a": int(a_values[first]),
                    "b": int(b_values[first]),
                    "tensor": int(tensor_out[first]),
                    "rtl": int(trace[first, 3]),
                }
                _atomic_json(
                    os.path.join(args.out, "semantic_validation.json"),
                    {"passed": False, "results": results, "arguments": vars(args)},
                )
                raise AssertionError(
                    f"candidate {index} differs at vector {first}: "
                    f"tensor={tensor_out[first]} rtl={trace[first, 3]}"
                )
    finally:
        evaluator.close()
        if getattr(engine, "tb_logger", None) is not None:
            engine.tb_logger.close()

    payload = {"passed": True, "results": results, "arguments": vars(args)}
    _atomic_json(os.path.join(args.out, "semantic_validation.json"), payload)
    print(
        f"stage2 DiffAM hard-tensor/RTL semantics: PASS "
        f"({len(candidates)} candidates × {args.vectors} vectors)"
    )


if __name__ == "__main__":
    main()
