"""Adapter from immutable candidates to the existing v5 RTL/DC/Verilator stack."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import multiprocessing
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils import CompressorTree, Mul
from trainer.arith_das_v5.compressor_graph import CompressorGraph
from trainer.arith_das_v5.core import CompressorRouting

from .candidate import Candidate
from .canonical_router import CanonicalRouter, CanonicalRoutingError


def _dc_worker(params):
    # Worker-level EDA chatter is too verbose for the training log. The parent
    # reports one compact summary after each batch instead.
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            return CompressorRouting.parallel_simulate_worker(*params)


class EvaluationCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS evaluations "
            "(cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT result_json FROM evaluations WHERE cache_key=?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, result: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO evaluations(cache_key,result_json) VALUES (?,?)",
            (key, json.dumps(result, sort_keys=True)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class V5CandidateEvaluator:
    def __init__(
        self,
        engine,
        run_dir: str,
        batch_size: int = 64,
        n_processing: int = 64,
        target_delay: float = 1.5,
        error_vectors: int = 16_000_000,
    ):
        self.engine = engine
        self.run_dir = os.path.abspath(run_dir)
        self.build_dir = os.path.join(self.run_dir, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.batch_size = int(batch_size)
        self.n_processing = int(n_processing)
        self.target_delay = float(target_delay)
        self.error_vectors = int(error_vectors)
        self.router = CanonicalRouter()
        self.cache = EvaluationCache(os.path.join(self.run_dir, "evaluation_cache.sqlite"))
        self.fingerprint = json.dumps(
            {
                "bit_width": int(engine.bit_width),
                "encode_type": str(engine.encode_type),
                "delay": self.target_delay,
                "vectors": self.error_vectors,
                "approx_lib": str(getattr(engine, "approx_lib_path", "")),
                "approx42_lib": str(getattr(engine, "approx42_library_path", "")),
            },
            sort_keys=True,
        )

    def _cache_key(self, c: Candidate) -> str:
        return f"{self.fingerprint}|{c.routing_hash}"

    def _prepare(self, c: Candidate) -> Tuple[tuple, list, str]:
        self.engine._activate_trunc_profile(c.k)
        self.engine.state = {
            "k": int(c.k),
            "ct22": np.asarray(c.ct22, dtype=int),
            "ct32": np.asarray(c.ct32, dtype=int),
            "ct42": np.asarray(c.ct42, dtype=int),
            "cells": copy.deepcopy(c.cells),
        }
        pp = np.asarray(self.engine.initial_pp, dtype=int)
        ct = CompressorTree(
            pp,
            np.asarray(c.ct32, dtype=int),
            np.asarray(c.ct22, dtype=int),
            np.asarray(c.ct42, dtype=int),
        )
        assignment = ct.compressor_assignment_fused()
        self.engine.assignment = assignment
        self.engine.comp_graph = CompressorGraph(
            pp, assignment, num_node_types=self.engine.num_node_types
        )
        connection = c.routing if c.routing is not None else self.router.route(self.engine.comp_graph)

        cell_types = {}
        for entry in c.cells:
            slot = tuple(int(x) for x in entry[:4])
            node_idx = self.engine.comp_graph.indice_map.get(slot)
            if node_idx is None:
                raise ValueError(f"cell slot disappeared: {slot}")
            t, type_idx = int(entry[2]), int(entry[4])
            _head, table = self.engine._type_head_and_table(t)
            if type_idx <= 0 or type_idx >= len(table):
                raise ValueError(f"invalid cell type {type_idx} for slot {slot}")
            cell_types[int(node_idx)] = (t, type_idx)
        cell_map = self.engine._cell_map_from_types(cell_types)

        if c.k > 0:
            ct.trunc_cols = int(c.k)
            ct.trunc_bits = dict(self.engine._trunc_bits)
        mul = Mul(self.engine.bit_width, self.engine.encode_type, ct)
        route_assignment = self.engine.emit_assignment(connection, cell_map=cell_map)
        key = self._cache_key(c)
        candidate_dir = os.path.join(self.build_dir, key.split("|")[-1][:20])
        os.makedirs(candidate_dir, exist_ok=True)
        rtl_path = os.path.join(candidate_dir, "MUL.v")
        mul.emit_verilog(
            rtl_path,
            assignment=route_assignment,
            extra_modules_src=self.engine._approx_modules_src(cell_map),
        )
        return ct, connection, rtl_path

    def evaluate(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        candidates = list(candidates)
        pending = []
        for c in candidates:
            cached = self.cache.get(self._cache_key(c))
            if cached is not None:
                c.set_result(cached)
                c.metadata["cache_hit"] = True
                continue
            try:
                ct, connection, rtl_path = self._prepare(c)
            except (AssertionError, ValueError, IndexError, CanonicalRoutingError) as exc:
                c.valid = False
                c.failure_reason = f"design_invalid:{type(exc).__name__}:{exc}"
                logging.warning("candidate %s invalid: %s", c.candidate_id, exc)
                continue
            pending.append((c, ct, connection, rtl_path))

        for batch_start in range(0, len(pending), self.batch_size):
            batch = pending[batch_start : batch_start + self.batch_size]
            params = []
            for local_id, (_c, ct, _connection, rtl_path) in enumerate(batch):
                worker_dir = os.path.join(
                    os.path.dirname(rtl_path), f"worker_{batch_start + local_id}"
                )
                params.append(
                    (
                        int(self.engine.bit_width),
                        str(self.engine.encode_type),
                        copy.deepcopy(ct),
                        rtl_path,
                        worker_dir,
                        self.target_delay,
                        local_id,
                        0,
                        "dc",
                        "verilator",
                        self.error_vectors,
                    )
                )
            logging.info(
                "[three-stage] evaluating batch %d..%d (%d candidates, dc_workers=%d)",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
                min(self.n_processing, len(batch)),
            )
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(max(1, min(self.n_processing, len(batch)))) as pool:
                results = pool.map(_dc_worker, params)
            ok_count = 0
            dc_failed_count = 0
            verilator_failed_count = 0
            for (c, _ct, connection, rtl_path), raw in zip(batch, results):
                if raw.get("failed") or not raw.get("result"):
                    c.valid = False
                    c.failure_reason = "dc_failed"
                    dc_failed_count += 1
                    continue
                measured = raw.get("measured_error")
                if measured is None or measured.get("mred") is None:
                    c.valid = False
                    c.failure_reason = "verilator_failed"
                    verilator_failed_count += 1
                    continue
                ppa = raw["result"][0]
                result = {
                    "area": float(ppa["area"]),
                    "power": float(ppa["power"]),
                    "delay": float(ppa["delay"]),
                    "mred": float(measured["mred"]),
                    "valid": True,
                    "metadata": {
                        "med": measured.get("med"),
                        "bias": measured.get("bias"),
                        "wce_mc": measured.get("wce_mc"),
                        "error_source": measured.get("source", "verilator"),
                        "rtl_path": rtl_path,
                    },
                }
                c.set_result(result)
                ok_count += 1
                if c.routing is not None:
                    c.metadata["connection"] = connection
                self.cache.put(self._cache_key(c), result)
            logging.info(
                "[three-stage] batch complete: ok=%d dc_failed=%d verilator_failed=%d",
                ok_count,
                dc_failed_count,
                verilator_failed_count,
            )
        return candidates

    def close(self):
        self.cache.close()
