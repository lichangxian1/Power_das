#!/usr/bin/env python3
"""Compare routing-only PPO, active-block PPO, and CEM on frozen Arith-DAS.

The official checkout is imported from ``../Arith-DAS`` without modifying its
tracked sources.  Both optimizers reuse one immutable Dadda/Wallace tree and one
fixed DC target. PPO keeps the original GCN, categorical sampler, and loss;
active-block PPO rotates a small set of mutable (stage, column) blocks; CEM
uses Power-DAS's direct-logit Gumbel-Hungarian implementation.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import logging
import multiprocessing
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf


POWER_DAS_ROOT = Path(__file__).resolve().parents[1]
ARITH_DAS_ROOT = Path(
    os.environ.get("ARITH_DAS_ROOT", POWER_DAS_ROOT.parent / "Arith-DAS")
).resolve()
if not (ARITH_DAS_ROOT / "trainer" / "arith_das.py").is_file():
    raise RuntimeError(f"official Arith-DAS checkout not found: {ARITH_DAS_ROOT}")

# It is important that the official checkout wins over Power_das here: the
# experiment is specifically testing the vanilla PPO implementation.
sys.path.insert(0, str(ARITH_DAS_ROOT))
from trainer.arith_das import CompressorGraph  # noqa: E402
from trainer.arith_das import CompressorRouting as VanillaCompressorRouting  # noqa: E402
from utils import CompressorTree, Mul, get_initial_partial_product  # noqa: E402


def _load_routing_cem():
    """Load Power-DAS CEM without colliding with vanilla's trainer package."""
    cem_path = POWER_DAS_ROOT / "trainer" / "arith_three_stage" / "cem.py"
    spec = importlib.util.spec_from_file_location("power_das_routing_cem", cem_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Power-DAS CEM: {cem_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoutingCEM, cem_path


RoutingCEM, CEM_FILE = _load_routing_cem()


VANILLA_TRAINER_FILE = Path(inspect.getsourcefile(VanillaCompressorRouting)).resolve()
if ARITH_DAS_ROOT not in VANILLA_TRAINER_FILE.parents:
    raise RuntimeError(
        "import isolation failed: expected vanilla trainer under "
        f"{ARITH_DAS_ROOT}, got {VANILLA_TRAINER_FILE}"
    )

_DC_ADAPTER_MODULE = None


class CEMCompatibleCompressorGraph(CompressorGraph):
    """Vanilla graph plus the read-only descriptors required by CEM."""

    port_num = 3

    def get_slice_carry_sources(self, stage: int, column: int):
        if column <= 0:
            return []
        sources = []
        for src_idx in self.slice_indice_map[(stage - 1, column - 1)]:
            if self.vertex_list[src_idx][2] in (0, 1):
                sources.append((src_idx, "carry"))
        return sources


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_hash(state: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("ct32", "ct22"):
        arr = np.ascontiguousarray(state[key], dtype=np.int64)
        h.update(key.encode("ascii"))
        h.update(arr.shape.__repr__().encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _load_dc_adapter(path: str):
    global _DC_ADAPTER_MODULE
    if _DC_ADAPTER_MODULE is None:
        adapter_path = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(
            "vanilla_arith_dc_adapter", adapter_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import DC adapter: {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DC_ADAPTER_MODULE = module
    return _DC_ADAPTER_MODULE


def _evaluate_dc(job: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one emitted routing with the project's remote DC harness."""
    adapter = _load_dc_adapter(job["dc_adapter_path"])
    adapter.EDA_BASE_DIR = job["dc_base_dir"]

    rtl_path = Path(job["rtl_path"]).resolve()
    build_path = Path(job["build_path"]).resolve()
    build_path.mkdir(parents=True, exist_ok=True)
    rtl_src = rtl_path.read_text()

    previous_cwd = Path.cwd()
    try:
        os.chdir(build_path)
        result = adapter.evaluate_single_routing(
            int(job["id"]),
            rtl_src,
            bit_width=int(job["bit_width"]),
            target_delay=float(job["target_delay"]),
        )
    finally:
        os.chdir(previous_cwd)

    if (
        not result
        or not result.get("success")
        or result.get("area") is None
        or result.get("power_mw") is None
    ):
        return {
            "id": int(job["id"]),
            "failed": True,
            "log": None if not result else result.get("log"),
        }

    area = float(result["area"])
    power_mw = float(result["power_mw"])
    if area <= 10.0 or power_mw <= 0.0:
        return {
            "id": int(job["id"]),
            "failed": True,
            "log": f"rejected nonphysical PPA: area={area}, power_mw={power_mw}",
        }

    delay = result.get("delay")
    if delay is None:
        delay = float(job["target_delay"])
    return {
        "id": int(job["id"]),
        "failed": False,
        "result": {
            "delay": abs(float(delay)),
            "area": area,
            "power": power_mw / 1000.0,
            "target_delay": float(job["target_delay"]),
            "worker_id": int(job["id"]),
        },
    }


class RoutingOnlyDC(VanillaCompressorRouting):
    """Vanilla routing policy with a structurally immutable compressor tree."""

    optimizer_name = "ppo"

    def __init__(
        self,
        fixed_target_delay: float,
        dc_adapter_path: str,
        dc_base_dir: str,
        metrics_path: str,
        **kwargs,
    ):
        self.fixed_target_delay = float(fixed_target_delay)
        self.dc_adapter_path = str(Path(dc_adapter_path).resolve())
        self.dc_base_dir = str(dc_base_dir)
        self.metrics_path = str(Path(metrics_path).resolve())
        self._fixed_state: Dict[str, np.ndarray] | None = None
        self._fixed_state_hash: str | None = None
        self._last_sample_info: List[Dict[str, Any]] = []
        super().__init__(**kwargs)

    def _start_reset(self):
        self.initial_pp = get_initial_partial_product(
            self.bit_width, self.encode_type
        ).astype(int)
        if self.ct_arch == "wallace":
            ct = CompressorTree.wallace(self.initial_pp)
        elif self.ct_arch == "dadda":
            ct = CompressorTree.dadda(self.initial_pp)
        else:
            raise ValueError(f"invalid frozen compressor tree: {self.ct_arch}")

        self._fixed_state = {
            "ct32": np.asarray(ct.ct32, dtype=int).copy(),
            "ct22": np.asarray(ct.ct22, dtype=int).copy(),
        }
        self._fixed_state_hash = _state_hash(self._fixed_state)
        logging.info(
            "Frozen %s structure initialized: hash=%s",
            self.ct_arch,
            self._fixed_state_hash,
        )

    def _assert_frozen(self):
        if self._fixed_state is None or self._fixed_state_hash is None:
            raise RuntimeError("frozen structure was not initialized")
        if _state_hash(self.state) != self._fixed_state_hash:
            raise RuntimeError("compressor-tree structure changed in routing-only run")
        if not np.array_equal(self.state["ct32"], self._fixed_state["ct32"]):
            raise RuntimeError("ct32 changed in routing-only run")
        if not np.array_equal(self.state["ct22"], self._fixed_state["ct22"]):
            raise RuntimeError("ct22 changed in routing-only run")

    def reset(self):
        self.state = copy.deepcopy(self._fixed_state)
        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct = CompressorTree(pp, self.state["ct32"], self.state["ct22"])
        self.assignment = ct.compressor_assignment_fused()
        self.comp_graph = CEMCompatibleCompressorGraph(pp, self.assignment)
        self._assert_frozen()

    def transition(self, action: int):
        raise RuntimeError(
            f"structural transition(action={action}) is disabled in routing-only ablation"
        )

    def update_pool(self, objective: float, state: Dict[str, np.ndarray]):
        self._assert_frozen()
        # Deliberately do not admit any architecture into the vanilla structure pool.

    @staticmethod
    def _sample_metric(sample: Dict[str, Any]) -> Dict[str, Any]:
        result = sample["result"]
        metric = {
            "objective": float(sample["objective"]),
            "delay": float(np.mean([x["delay"] for x in result])),
            "area": float(np.mean([x["area"] for x in result])),
            "power": float(np.mean([x["power"] for x in result])),
            "old_log_prob": float(sample["overall_log_prob"]),
        }
        if "proposal_source" in sample:
            metric["proposal_source"] = str(sample["proposal_source"])
        return metric

    def _sample_routes(self):
        with torch.no_grad():
            sample_info = []
            z_mat_dict = self.get_Z_mat()
            for _ in range(self.num_samples):
                connection, overall_log_prob = self.sample_from_logits(z_mat_dict)
                sample_info.append(
                    {
                        "connection": connection,
                        "overall_log_prob": overall_log_prob,
                    }
                )
        return sample_info

    def _sample_one_route(self):
        with torch.no_grad():
            return self.sample_from_logits(self.get_Z_mat())

    def get_samples(self):
        self._assert_frozen()
        Path(self.build_dir).mkdir(parents=True, exist_ok=True)

        sample_info = self._sample_routes()
        with torch.no_grad():
            for sample_idx, sample in enumerate(sample_info):
                assignment = self.emit_assignment(sample["connection"])
                ct = CompressorTree(
                    self.initial_pp, self.state["ct32"], self.state["ct22"]
                )
                mul = Mul(self.bit_width, self.encode_type, ct)
                rtl_path = Path(self.build_dir).resolve() / f"MUL-{sample_idx}.v"
                mul.emit_verilog(str(rtl_path), assignment=assignment)
                sample["rtl_path"] = str(rtl_path)

        jobs = [
            {
                "id": i,
                "bit_width": self.bit_width,
                "rtl_path": sample["rtl_path"],
                "build_path": str(
                    Path(self.build_dir).resolve()
                    / f"worker_{i}_td{self.fixed_target_delay:g}"
                ),
                "target_delay": self.fixed_target_delay,
                "dc_adapter_path": self.dc_adapter_path,
                "dc_base_dir": self.dc_base_dir,
            }
            for i, sample in enumerate(sample_info)
        ]
        logging.info(
            "DC batch: samples=%d parallelism=%d target_delay=%g ns",
            len(jobs),
            self.n_processing,
            self.fixed_target_delay,
        )
        if self.n_processing == 1:
            results = [_evaluate_dc(job) for job in jobs]
        else:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=self.n_processing) as pool:
                results = pool.map(_evaluate_dc, jobs)

        result_by_id = {
            item["id"]: item
            for item in results
            if not item.get("failed") and item.get("result") is not None
        }
        failures = [item for item in results if item.get("failed")]
        if failures:
            logging.warning(
                "DC failures: %d/%d; failed routings are excluded from update",
                len(failures),
                len(results),
            )

        kept = []
        for i, sample in enumerate(sample_info):
            if i not in result_by_id:
                continue
            sample["result"] = [result_by_id[i]["result"]]
            sample["objective"] = self.get_objective(sample["result"])
            kept.append(sample)
        if not kept:
            raise RuntimeError("all DC routing evaluations failed")

        self._last_sample_info = kept
        self._assert_frozen()
        return kept

    def get_full_target_delay_result(self):
        self._assert_frozen()
        build_dir = Path(f"{self.build_dir}_full_ppa").resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        rtl_path = build_dir / "MUL.v"

        ct = CompressorTree(
            self.initial_pp, self.state["ct32"], self.state["ct22"]
        )
        mul = Mul(self.bit_width, self.encode_type, ct)
        assignment = self.emit_assignment(self.found_best_info["connection"])
        mul.emit_verilog(str(rtl_path), assignment=assignment)
        evaluated = _evaluate_dc(
            {
                "id": 1_000_000,
                "bit_width": self.bit_width,
                "rtl_path": str(rtl_path),
                "build_path": str(build_dir / "worker"),
                "target_delay": self.fixed_target_delay,
                "dc_adapter_path": self.dc_adapter_path,
                "dc_base_dir": self.dc_base_dir,
            }
        )
        if evaluated.get("failed") or evaluated.get("result") is None:
            raise RuntimeError(
                f"checkpoint full-PPA DC evaluation failed: {evaluated.get('log')}"
            )
        return [evaluated["result"]]

    def log_episode(self, episode_idx, info):
        super().log_episode(episode_idx, info)
        samples = [self._sample_metric(x) for x in self._last_sample_info]
        objectives = [x["objective"] for x in samples]
        record = {
            "episode": int(episode_idx),
            "optimizer": self.optimizer_name,
            "structure_hash": self._fixed_state_hash,
            "num_successful_dc": len(samples),
            "batch_objective_min": float(np.min(objectives)),
            "batch_objective_mean": float(np.mean(objectives)),
            "batch_objective_std": float(np.std(objectives)),
            "found_best_objective": float(self.found_best_info["objective"]),
            "epoch_loss": info["epoch_loss"],
            "optimizer_stats": getattr(self, "_optimizer_stats", None),
            "samples": samples,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class RoutingOnlyActiveBlockPPODC(RoutingOnlyDC):
    """Vanilla PPO with rotating mutable blocks and an incumbent route."""

    optimizer_name = "active_block_ppo"

    def log_episode(self, episode_idx, info):
        self._optimizer_stats = self.get_active_block_stats()
        super().log_episode(episode_idx, info)


class RoutingOnlyCEMDC(RoutingOnlyDC):
    """Power-DAS CEM over the exact frozen graph and DC objective used by PPO."""

    optimizer_name = "cem"

    def __init__(
        self,
        cem_elite_fraction: float,
        cem_smoothing: float,
        cem_exploration: float,
        cem_temperature: float,
        cem_init: str,
        **kwargs,
    ):
        self.cem_elite_fraction = float(cem_elite_fraction)
        if not 0.0 < self.cem_elite_fraction <= 1.0:
            raise ValueError("CEM elite fraction must be in (0, 1]")
        self._cem_logits = None
        self._optimizer_stats = None
        super().__init__(**kwargs)
        self.cem = RoutingCEM(
            self,
            smoothing=cem_smoothing,
            exploration=cem_exploration,
            temperature=cem_temperature,
            init_mode=cem_init,
        )

    def _eligible_carry_rows(self, stage: int, column: int):
        return [
            i
            for i, src_idx in enumerate(
                self.comp_graph.slice_indice_map[(stage - 1, column - 1)]
            )
            if self.comp_graph.vertex_list[src_idx][2] in (0, 1)
        ]

    def _filter_carry_logits(self, logits):
        filtered = {
            key: {name: value.detach().clone() for name, value in values.items()}
            for key, values in logits.items()
        }
        for (stage, column), values in filtered.items():
            if column <= 0:
                continue
            rows = self._eligible_carry_rows(stage, column)
            for name in ("ca", "cb", "cc"):
                values[name] = values[name][rows, :]
        return filtered

    def get_cache(self, z_mat_dict):
        """Build CEM matrices without vanilla's all-false carry-source rows."""
        mask_cache = {}
        z_cache = {}
        for (stage, column), values in z_mat_dict.items():
            sum_mask = self.comp_graph.get_slice_sum_mask(stage, column).to(
                self.device
            )
            sum_blocks = [sum_mask[port] for port in range(3)]
            if column == 0:
                mask_blocks = sum_blocks
                z_blocks = [values[name] for name in ("sa", "sb", "sc")]
            else:
                carry_mask = self.comp_graph.get_slice_carry_mask(
                    stage, column
                ).to(self.device)
                rows = self._eligible_carry_rows(stage, column)
                carry_blocks = [carry_mask[port, rows, :] for port in range(3)]
                mask_blocks = [
                    torch.cat((sum_blocks[port], carry_blocks[port]), dim=0)
                    for port in range(3)
                ]
                z_blocks = [
                    torch.cat((values[sname], values[cname]), dim=0)
                    for sname, cname in (
                        ("sa", "ca"),
                        ("sb", "cb"),
                        ("sc", "cc"),
                    )
                ]
            mask_cache[(stage, column)] = torch.cat(mask_blocks, dim=1)
            z_cache[(stage, column)] = torch.cat(z_blocks, dim=1)
        return mask_cache, z_cache

    def _ensure_cem_logits(self):
        if self._cem_logits is None:
            with torch.no_grad():
                template = self._filter_carry_logits(self.get_Z_mat())
            self._cem_logits = self.cem.initialize(template)
            self._optimizer_stats = self.cem.stats(self._cem_logits)

    def _sample_routes(self):
        self._ensure_cem_logits()
        with torch.no_grad():
            sampled = self.cem.sample_many(self._cem_logits, self.num_samples)
        return [
            {"connection": connection, "overall_log_prob": score}
            for connection, score in sampled
        ]

    def _sample_one_route(self):
        self._ensure_cem_logits()
        return self.cem.sample(self._cem_logits)

    def run_episode(self, episode_idx):
        logging.info("Episode %d start (CEM)", episode_idx)
        self.reset()
        sample_info_list = self.get_samples()
        self.update_found_best_info(sample_info_list)

        ranked = sorted(sample_info_list, key=lambda item: item["objective"])
        elite_count = max(
            1, int(np.ceil(self.cem_elite_fraction * len(sample_info_list)))
        )
        self._cem_logits, self._optimizer_stats = self.cem.update(
            self._cem_logits,
            [item["connection"] for item in ranked[:elite_count]],
        )
        self._optimizer_stats = dict(self._optimizer_stats)
        self._optimizer_stats["elite_count"] = elite_count
        self._optimizer_stats["elite_fraction"] = self.cem_elite_fraction

        best = ranked[0]
        info = {
            "objective": best["objective"],
            "simulated_result": best["result"],
            "epoch_loss": [
                {
                    "cem_normalized_entropy": self._optimizer_stats[
                        "normalized_entropy"
                    ],
                    "cem_mean_max_probability": self._optimizer_stats[
                        "mean_max_probability"
                    ],
                }
            ],
        }
        if episode_idx % self.log_freq == 0:
            self.log_episode(episode_idx, info)

    def save_experiment(self, episode_idx):
        super().save_experiment(episode_idx)
        save_dir = Path(self.log_dir) / f"save_iter{episode_idx}"
        torch.save(self._cem_logits, save_dir / "cem_logits.pth")


class RoutingOnlyCEMEscapeDC(RoutingOnlyCEMDC):
    """CEM with random immigrants, incumbent mutation, and partial restarts."""

    optimizer_name = "cem_escape"

    def __init__(
        self,
        cem_random_immigrants: int,
        cem_incumbent_mutations: int,
        cem_mutation_blocks: int,
        cem_stagnation_patience: int,
        cem_restart_block_fraction: float,
        cem_restart_entropy_threshold: float,
        cem_restart_temperature: float,
        cem_restart_temperature_episodes: int,
        **kwargs,
    ):
        self.cem_random_immigrants = int(cem_random_immigrants)
        self.cem_incumbent_mutations = int(cem_incumbent_mutations)
        self.cem_mutation_blocks = int(cem_mutation_blocks)
        self.cem_stagnation_patience = int(cem_stagnation_patience)
        self.cem_restart_block_fraction = float(cem_restart_block_fraction)
        self.cem_restart_entropy_threshold = float(
            cem_restart_entropy_threshold
        )
        self.cem_restart_temperature = float(cem_restart_temperature)
        self.cem_restart_temperature_episodes = int(
            cem_restart_temperature_episodes
        )
        self._cem_initial_logits = None
        self._cem_episode_idx = 0
        self._cem_stagnation_steps = 0
        self._cem_restart_count = 0
        self._cem_last_restart_blocks = 0
        self._cem_restart_temperature_until = -1
        super().__init__(**kwargs)
        self._cem_base_temperature = float(self.cem.temperature)
        if self.cem_random_immigrants < 0:
            raise ValueError("CEM random immigrants must be non-negative")
        if self.cem_incumbent_mutations < 0:
            raise ValueError("CEM incumbent mutations must be non-negative")
        if (
            self.cem_random_immigrants + self.cem_incumbent_mutations
            >= self.num_samples
        ):
            raise ValueError(
                "CEM escape proposals must leave at least one main-distribution sample"
            )
        if self.cem_mutation_blocks < 1:
            raise ValueError("CEM mutation blocks must be positive")
        if self.cem_stagnation_patience < 1:
            raise ValueError("CEM stagnation patience must be positive")
        if not 0.0 < self.cem_restart_block_fraction <= 1.0:
            raise ValueError("CEM restart block fraction must be in (0, 1]")
        if not 0.0 <= self.cem_restart_entropy_threshold <= 1.0:
            raise ValueError("CEM restart entropy threshold must be in [0, 1]")
        if self.cem_restart_temperature <= 0.0:
            raise ValueError("CEM restart temperature must be positive")
        if self.cem_restart_temperature_episodes < 1:
            raise ValueError("CEM restart temperature episodes must be positive")

    @staticmethod
    def _clone_routing_logits(logits):
        return {
            key: {
                name: value.detach().clone()
                for name, value in values.items()
            }
            for key, values in logits.items()
        }

    def _ensure_cem_logits(self):
        super()._ensure_cem_logits()
        if self._cem_initial_logits is None:
            self._cem_initial_logits = self._clone_routing_logits(
                self._cem_logits
            )

    def _append_distinct_samples(
        self,
        sampled,
        seen,
        logits,
        count,
        source,
    ):
        added = 0
        attempts = 0
        max_attempts = max(100, count * 200)
        while added < count and attempts < max_attempts:
            attempts += 1
            connection, score = self.cem.sample(logits)
            signature = self.cem._route_signature(connection)
            if signature in seen:
                continue
            seen.add(signature)
            sampled.append(
                {
                    "connection": connection,
                    "overall_log_prob": score,
                    "proposal_source": source,
                }
            )
            added += 1
        if added != count:
            raise RuntimeError(
                f"CEM escape produced only {added}/{count} distinct {source} routes"
            )

    def _mutate_incumbent(self):
        incumbent = self.found_best_info.get("connection")
        if not incumbent:
            return self.cem.sample(self._cem_initial_logits)
        donor, donor_score = self.cem.sample(self._cem_logits)
        slice_keys = sorted(
            {
                tuple((edge[3] or {}).get("slice", ()))
                for edge in incumbent
                if (edge[3] or {}).get("slice") is not None
            }
        )
        block_count = min(self.cem_mutation_blocks, len(slice_keys))
        selected_indices = np.random.choice(
            len(slice_keys), size=block_count, replace=False
        )
        selected = {slice_keys[int(index)] for index in selected_indices}
        mutated = []
        if len(incumbent) != len(donor):
            raise RuntimeError(
                "CEM incumbent and donor connection sizes do not match"
            )
        for incumbent_edge, donor_edge in zip(incumbent, donor):
            incumbent_meta = incumbent_edge[3] or {}
            donor_meta = donor_edge[3] or {}
            if (
                tuple(incumbent_meta.get("slice", ()))
                != tuple(donor_meta.get("slice", ()))
                or int(incumbent_meta.get("flat_row", -1))
                != int(donor_meta.get("flat_row", -1))
            ):
                raise RuntimeError(
                    "CEM incumbent and donor edge order does not match"
                )
            if tuple(incumbent_meta.get("slice", ())) in selected:
                mutated.append(copy.deepcopy(donor_edge))
            else:
                mutated.append(copy.deepcopy(incumbent_edge))
        return mutated, donor_score

    def _sample_routes(self):
        self._ensure_cem_logits()
        self.cem.temperature = (
            self.cem_restart_temperature
            if self._cem_episode_idx < self._cem_restart_temperature_until
            else self._cem_base_temperature
        )
        sampled = []
        seen = set()
        main_count = (
            self.num_samples
            - self.cem_random_immigrants
            - self.cem_incumbent_mutations
        )
        with torch.no_grad():
            self._append_distinct_samples(
                sampled,
                seen,
                self._cem_logits,
                main_count,
                "cem",
            )
            self._append_distinct_samples(
                sampled,
                seen,
                self._cem_initial_logits,
                self.cem_random_immigrants,
                "random_immigrant",
            )
            for _ in range(self.cem_incumbent_mutations):
                attempts = 0
                while True:
                    attempts += 1
                    connection, score = self._mutate_incumbent()
                    signature = self.cem._route_signature(connection)
                    if signature not in seen:
                        seen.add(signature)
                        sampled.append(
                            {
                                "connection": connection,
                                "overall_log_prob": score,
                                "proposal_source": "incumbent_mutation",
                            }
                        )
                        break
                    if attempts >= 200:
                        raise RuntimeError(
                            "CEM escape could not produce a distinct incumbent mutation"
                        )
        if len(sampled) != self.num_samples:
            raise RuntimeError(
                f"CEM escape sampled {len(sampled)}/{self.num_samples} routes"
            )
        return sampled

    def _partial_restart(self):
        keys = sorted(self._cem_logits)
        count = max(
            1, int(np.ceil(self.cem_restart_block_fraction * len(keys)))
        )
        selected_indices = np.random.choice(
            len(keys), size=count, replace=False
        )
        for index in selected_indices:
            key = keys[int(index)]
            self._cem_logits[key] = {
                name: value.detach().clone()
                for name, value in self._cem_initial_logits[key].items()
            }
        self._cem_restart_count += 1
        self._cem_last_restart_blocks = count
        self._cem_restart_temperature_until = (
            self._cem_episode_idx + self.cem_restart_temperature_episodes + 1
        )
        self._cem_stagnation_steps = 0
        logging.info(
            "CEM escape partial restart: episode=%d blocks=%d/%d temperature=%g "
            "until_episode=%d",
            self._cem_episode_idx,
            count,
            len(keys),
            self.cem_restart_temperature,
            self._cem_restart_temperature_until - 1,
        )

    def run_episode(self, episode_idx):
        self._cem_episode_idx = int(episode_idx)
        logging.info("Episode %d start (CEM_ESCAPE)", episode_idx)
        self.reset()
        previous_best = float(self.found_best_info["objective"])
        sample_info_list = self.get_samples()
        self.update_found_best_info(sample_info_list)
        current_best = float(self.found_best_info["objective"])
        if current_best < previous_best - 1.0e-12:
            self._cem_stagnation_steps = 0
        else:
            self._cem_stagnation_steps += 1

        ranked = sorted(sample_info_list, key=lambda item: item["objective"])
        elite_count = max(
            1, int(np.ceil(self.cem_elite_fraction * len(sample_info_list)))
        )
        elite_connections = []
        elite_signatures = set()
        incumbent = self.found_best_info.get("connection")
        if incumbent:
            elite_connections.append(incumbent)
            elite_signatures.add(self.cem._route_signature(incumbent))
        for item in ranked:
            signature = self.cem._route_signature(item["connection"])
            if signature in elite_signatures:
                continue
            elite_connections.append(item["connection"])
            elite_signatures.add(signature)
            if len(elite_connections) >= elite_count:
                break
        self._cem_logits, self._optimizer_stats = self.cem.update(
            self._cem_logits,
            elite_connections,
        )
        restarted = False
        if (
            self._cem_stagnation_steps >= self.cem_stagnation_patience
            and self._optimizer_stats["normalized_entropy"]
            < self.cem_restart_entropy_threshold
        ):
            self._partial_restart()
            self._optimizer_stats = self.cem.stats(self._cem_logits)
            restarted = True

        self._optimizer_stats = dict(self._optimizer_stats)
        self._optimizer_stats.update(
            {
                "elite_count": len(elite_connections),
                "elite_fraction": self.cem_elite_fraction,
                "global_incumbent_in_elite": True,
                "main_samples": (
                    self.num_samples
                    - self.cem_random_immigrants
                    - self.cem_incumbent_mutations
                ),
                "random_immigrants": self.cem_random_immigrants,
                "incumbent_mutations": self.cem_incumbent_mutations,
                "mutation_blocks": self.cem_mutation_blocks,
                "stagnation_steps": self._cem_stagnation_steps,
                "restart_count": self._cem_restart_count,
                "restart_triggered": restarted,
                "restart_blocks": (
                    self._cem_last_restart_blocks if restarted else 0
                ),
                "sampling_temperature": self.cem.temperature,
            }
        )

        best = ranked[0]
        info = {
            "objective": best["objective"],
            "simulated_result": best["result"],
            "epoch_loss": [
                {
                    "cem_normalized_entropy": self._optimizer_stats[
                        "normalized_entropy"
                    ],
                    "cem_mean_max_probability": self._optimizer_stats[
                        "mean_max_probability"
                    ],
                }
            ],
        }
        if episode_idx % self.log_freq == 0:
            self.log_episode(episode_idx, info)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_manifest(
    args,
    config_path: Path,
    experiment: RoutingOnlyDC,
    kwargs: Dict[str, Any],
):
    now_bjt = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    manifest = {
        "purpose": "routing-only PPO, active-block PPO, and CEM comparison",
        "created_at_beijing": now_bjt.isoformat(),
        "output_dir": str(Path(args.out).resolve()),
        "vanilla_arith_root": str(ARITH_DAS_ROOT),
        "vanilla_git_commit": _git_head(ARITH_DAS_ROOT),
        "vanilla_trainer_file": str(VANILLA_TRAINER_FILE),
        "vanilla_trainer_sha256": _sha256_file(VANILLA_TRAINER_FILE),
        "vanilla_config_file": str(config_path),
        "vanilla_config_sha256": _sha256_file(config_path),
        "dc_adapter_file": str(Path(args.dc_adapter).resolve()),
        "dc_adapter_sha256": _sha256_file(Path(args.dc_adapter).resolve()),
        "optimizer": args.optimizer,
        "cem_file": (
            str(CEM_FILE) if args.optimizer in {"cem", "cem_escape"} else None
        ),
        "cem_sha256": (
            _sha256_file(CEM_FILE)
            if args.optimizer in {"cem", "cem_escape"}
            else None
        ),
        "frozen_structure": {
            "arch": experiment.ct_arch,
            "hash": experiment._fixed_state_hash,
            "ct32": experiment._fixed_state["ct32"].tolist(),
            "ct22": experiment._fixed_state["ct22"].tolist(),
        },
        "ablation_guards": {
            "reset_reuses_fixed_structure": True,
            "transition_raises": True,
            "architecture_pool_updates_disabled": True,
            "use_ppo_loss": kwargs["use_ppo_loss"],
            "use_delay_loss": kwargs["use_delay_loss"],
            "use_rule_loss": kwargs["use_rule_loss"],
            "use_disc_loss": kwargs["use_disc_loss"],
        },
        "training": {
            "bit_width": kwargs["bit_width"],
            "encode_type": kwargs["encode_type"],
            "episodes": kwargs["num_episodes"],
            "routes_per_episode": kwargs["num_samples"],
            "ppo_epochs_per_episode": kwargs["num_epochs"],
            "dc_parallelism": kwargs["n_processing"],
            "target_delay_ns": args.target_delay,
            "device": kwargs["device"],
            "seed": args.seed,
            "learning_rate": kwargs["optim_kwargs"]["lr"],
            "clip_range": kwargs["clip_range"],
            "objective_weights": {
                "delay": kwargs["delay_weight"],
                "area": kwargs["area_weight"],
                "power": kwargs["power_weight"],
            },
            "objective_scales": {
                "delay": kwargs["delay_scale"],
                "area": kwargs["area_scale"],
                "power": kwargs["power_scale"],
            },
            "cem": (
                {
                    "elite_fraction": args.cem_elite_fraction,
                    "smoothing": args.cem_smoothing,
                    "exploration": args.cem_exploration,
                    "temperature": args.cem_temperature,
                    "init": args.cem_init,
                }
                if args.optimizer in {"cem", "cem_escape"}
                else None
            ),
            "cem_escape": (
                {
                    "main_samples": (
                        args.samples
                        - args.cem_random_immigrants
                        - args.cem_incumbent_mutations
                    ),
                    "random_immigrants": args.cem_random_immigrants,
                    "incumbent_mutations": args.cem_incumbent_mutations,
                    "mutation_blocks": args.cem_mutation_blocks,
                    "global_incumbent_in_elite": True,
                    "stagnation_patience": args.cem_stagnation_patience,
                    "restart_block_fraction": args.cem_restart_block_fraction,
                    "restart_entropy_threshold": (
                        args.cem_restart_entropy_threshold
                    ),
                    "restart_temperature": args.cem_restart_temperature,
                    "restart_temperature_episodes": (
                        args.cem_restart_temperature_episodes
                    ),
                }
                if args.optimizer == "cem_escape"
                else None
            ),
            "active_block": (
                {
                    "blocks_per_episode": args.active_blocks_per_episode,
                    "seed": (
                        args.seed
                        if args.active_block_seed is None
                        else args.active_block_seed
                    ),
                    "include_incumbent": not args.no_active_incumbent,
                    "rotation": "fixed_seed_cyclic_permutation",
                    "unit": "(stage,column)",
                }
                if args.optimizer == "active_ppo"
                else None
            ),
        },
    }
    manifest_path = Path(args.out).resolve() / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _validate_without_dc(experiment: RoutingOnlyDC, out_dir: Path):
    hashes = []
    for _ in range(3):
        experiment.reset()
        hashes.append(_state_hash(experiment.state))
    if len(set(hashes)) != 1:
        raise RuntimeError(f"structure hashes changed across reset: {hashes}")

    active_validation = None
    escape_validation = None
    with torch.no_grad():
        if getattr(experiment, "active_blocks_per_episode", 0) > 0:
            experiment._begin_active_block_episode(0)
            incumbent, incumbent_log_prob = experiment._sample_one_route()
            connection, log_prob = experiment._sample_one_route()
            active_edges = [
                edge for edge in connection if edge[3].get("ppo_active", False)
            ]
            frozen_edge_count = 0
            for proposal_edge, incumbent_edge in zip(connection, incumbent):
                if proposal_edge[3].get("ppo_active", False):
                    continue
                if proposal_edge[:3] != incumbent_edge[:3]:
                    raise RuntimeError("non-active edge changed during validation")
                frozen_edge_count += 1
            active_validation = {
                "incumbent_log_probability": float(incumbent_log_prob),
                "active_edge_count": len(active_edges),
                "frozen_edge_count": frozen_edge_count,
                "stats": experiment.get_active_block_stats(),
            }
        elif isinstance(experiment, RoutingOnlyCEMEscapeDC):
            sampled = experiment._sample_routes()
            signatures = {
                experiment.cem._route_signature(item["connection"])
                for item in sampled
            }
            source_counts = {}
            for item in sampled:
                source = item["proposal_source"]
                source_counts[source] = source_counts.get(source, 0) + 1
            if len(sampled) != experiment.num_samples:
                raise RuntimeError("CEM escape validation sample count mismatch")
            if len(signatures) != len(sampled):
                raise RuntimeError("CEM escape validation produced duplicate routes")
            expected_sources = {
                "cem": (
                    experiment.num_samples
                    - experiment.cem_random_immigrants
                    - experiment.cem_incumbent_mutations
                ),
                "random_immigrant": experiment.cem_random_immigrants,
                "incumbent_mutation": experiment.cem_incumbent_mutations,
            }
            if source_counts != expected_sources:
                raise RuntimeError(
                    f"CEM escape source mix mismatch: {source_counts} != "
                    f"{expected_sources}"
                )
            experiment.found_best_info["connection"] = copy.deepcopy(
                sampled[0]["connection"]
            )
            connection, log_prob = experiment._mutate_incumbent()
            changed_slices = {
                tuple((mutated[3] or {}).get("slice", ()))
                for original, mutated in zip(
                    experiment.found_best_info["connection"], connection
                )
                if original[:3] != mutated[:3]
            }
            experiment.found_best_info["connection"] = []
            if len(connection) != 884:
                raise RuntimeError("CEM escape incumbent mutation is incomplete")
            if len(changed_slices) > experiment.cem_mutation_blocks:
                raise RuntimeError(
                    "CEM escape incumbent mutation changed too many blocks"
                )
            escape_validation = {
                "distinct_routes": len(signatures),
                "source_counts": source_counts,
                "mutation_blocks": experiment.cem_mutation_blocks,
                "mutated_slices_with_changed_edges": len(changed_slices),
            }
        else:
            connection, log_prob = experiment._sample_one_route()
    assignment = experiment.emit_assignment(connection)
    ct = CompressorTree(
        experiment.initial_pp,
        experiment.state["ct32"],
        experiment.state["ct22"],
    )
    mul = Mul(experiment.bit_width, experiment.encode_type, ct)
    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    rtl_path = validation_dir / "MUL.v"
    mul.emit_verilog(str(rtl_path), assignment=assignment)
    experiment._assert_frozen()

    validation = {
        "status": "ok",
        "structure_hashes_across_3_resets": hashes,
        "sampled_connection_count": len(connection),
        "sampled_log_probability": float(log_prob),
        "active_block_validation": active_validation,
        "cem_escape_validation": escape_validation,
        "rtl_path": str(rtl_path),
        "rtl_size_bytes": rtl_path.stat().st_size,
    }
    (out_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--config",
        default=str(ARITH_DAS_ROOT / "configs/config_groups/mul_16_and.yaml"),
    )
    parser.add_argument(
        "--optimizer",
        choices=("ppo", "active_ppo", "cem", "cem_escape"),
        default="ppo",
    )
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n-processing", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--target-delay", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--active-blocks-per-episode", type=int, default=32)
    parser.add_argument("--active-block-seed", type=int, default=None)
    parser.add_argument("--no-active-incumbent", action="store_true")
    parser.add_argument("--cem-elite-fraction", type=float, default=0.20)
    parser.add_argument("--cem-smoothing", type=float, default=0.25)
    parser.add_argument("--cem-exploration", type=float, default=0.05)
    parser.add_argument("--cem-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cem-init", choices=("policy", "uniform"), default="policy"
    )
    parser.add_argument("--cem-random-immigrants", type=int, default=7)
    parser.add_argument("--cem-incumbent-mutations", type=int, default=1)
    parser.add_argument("--cem-mutation-blocks", type=int, default=32)
    parser.add_argument("--cem-stagnation-patience", type=int, default=40)
    parser.add_argument("--cem-restart-block-fraction", type=float, default=0.30)
    parser.add_argument(
        "--cem-restart-entropy-threshold", type=float, default=0.25
    )
    parser.add_argument("--cem-restart-temperature", type=float, default=2.0)
    parser.add_argument(
        "--cem-restart-temperature-episodes", type=int, default=10
    )
    parser.add_argument("--delay-scale", type=float, default=1.44)
    parser.add_argument("--area-scale", type=float, default=800.0)
    parser.add_argument("--power-scale", type=float, default=1.07e-2)
    parser.add_argument(
        "--dc-adapter", default=str(POWER_DAS_ROOT / "run_power_sweep.py")
    )
    parser.add_argument(
        "--dc-base-dir",
        default="/home/lchangxian/sandbox/sandbox_base_dcpwr",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.device not in {"cuda:0", "cuda:2"}:
        raise SystemExit("device must be physical cuda:0 or cuda:2")
    if args.episodes < 1 or args.samples < 1 or args.n_processing < 1:
        raise SystemExit("episodes, samples, and n-processing must be positive")
    if args.active_blocks_per_episode < 1:
        raise SystemExit("active-blocks-per-episode must be positive")
    if args.n_processing > args.samples:
        raise SystemExit("n-processing cannot exceed samples in the fixed-delay run")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    experiment_kwargs = copy.deepcopy(cfg["experiment"]["kwargs"])
    kwargs = copy.deepcopy(cfg["trainer"]["kwargs"])
    kwargs.update(experiment_kwargs)
    kwargs.update(
        {
            "num_episodes": args.episodes,
            "num_samples": args.samples,
            "num_epochs": args.ppo_epochs,
            "save_freq": args.save_freq,
            "n_processing": args.n_processing,
            "n_full_target_delay_processing": args.n_processing,
            "device": args.device,
            "log_dir": str(out_dir / "logs"),
            "build_dir": str(out_dir / "build"),
            "use_ppo_loss": args.optimizer in {"ppo", "active_ppo"},
            "use_delay_loss": False,
            "use_rule_loss": False,
            "use_disc_loss": False,
            "optim_kwargs": {"lr": args.learning_rate},
            "scheduler_kwargs": {
                "T_max": args.episodes,
                "eta_min": cfg["trainer"]["kwargs"]["scheduler_kwargs"]["eta_min"],
            },
            "delay_scale": args.delay_scale,
            "area_scale": args.area_scale,
            "power_scale": args.power_scale,
            "synth": "dc",
            "active_blocks_per_episode": (
                args.active_blocks_per_episode
                if args.optimizer == "active_ppo"
                else 0
            ),
            "active_block_seed": (
                args.seed
                if args.active_block_seed is None
                else args.active_block_seed
            ),
            "active_block_include_incumbent": not args.no_active_incumbent,
        }
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "train.log"),
        ],
    )
    _set_seed(args.seed)
    logging.info("vanilla trainer=%s", VANILLA_TRAINER_FILE)
    logging.info("vanilla commit=%s", _git_head(ARITH_DAS_ROOT))
    logging.info(
        "routing-only %s: episodes=%d samples=%d DC_parallel=%d device=%s",
        args.optimizer.upper(),
        args.episodes,
        args.samples,
        args.n_processing,
        args.device,
    )

    experiment_classes = {
        "ppo": RoutingOnlyDC,
        "active_ppo": RoutingOnlyActiveBlockPPODC,
        "cem": RoutingOnlyCEMDC,
        "cem_escape": RoutingOnlyCEMEscapeDC,
    }
    experiment_class = experiment_classes[args.optimizer]
    optimizer_kwargs = {}
    if args.optimizer in {"cem", "cem_escape"}:
        optimizer_kwargs = {
            "cem_elite_fraction": args.cem_elite_fraction,
            "cem_smoothing": args.cem_smoothing,
            "cem_exploration": args.cem_exploration,
            "cem_temperature": args.cem_temperature,
            "cem_init": args.cem_init,
        }
    if args.optimizer == "cem_escape":
        optimizer_kwargs.update(
            {
                "cem_random_immigrants": args.cem_random_immigrants,
                "cem_incumbent_mutations": args.cem_incumbent_mutations,
                "cem_mutation_blocks": args.cem_mutation_blocks,
                "cem_stagnation_patience": args.cem_stagnation_patience,
                "cem_restart_block_fraction": args.cem_restart_block_fraction,
                "cem_restart_entropy_threshold": (
                    args.cem_restart_entropy_threshold
                ),
                "cem_restart_temperature": args.cem_restart_temperature,
                "cem_restart_temperature_episodes": (
                    args.cem_restart_temperature_episodes
                ),
            }
        )
    experiment = experiment_class(
        fixed_target_delay=args.target_delay,
        dc_adapter_path=args.dc_adapter,
        dc_base_dir=args.dc_base_dir,
        metrics_path=str(out_dir / "routing_metrics.jsonl"),
        **optimizer_kwargs,
        **kwargs,
    )
    _write_manifest(args, config_path, experiment, kwargs)
    validation = _validate_without_dc(experiment, out_dir)
    logging.info("no-DC validation: %s", validation)
    if args.validate_only:
        logging.info("validate-only complete")
        return

    experiment.run_experiment()
    logging.info(
        "training complete: best_objective=%s",
        experiment.found_best_info["objective"],
    )


if __name__ == "__main__":
    main()
