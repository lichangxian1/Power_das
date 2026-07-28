"""Pure DiffAM Stage-2 cell proposer for fixed Stage-1 backbones."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from Appr_Comp.cellsolver import sim as diff_sim
from Appr_Comp.cellsolver.solver import GradientCellSolver
from trainer.arith_das_v5.compressor_graph import CompressorGraph
from utils import CompressorTree, Mul

from .candidate import Candidate
from .canonical_router import CanonicalRouter

CellConfig = Dict[int, Tuple[int, int]]


def _config_key(config: CellConfig) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(sorted((int(node), int(t), int(k)) for node, (t, k) in config.items()))


def _config_distance(a: CellConfig, b: CellConfig) -> int:
    return sum(a.get(node) != b.get(node) for node in set(a) | set(b))


def _balanced_quota(remaining: int, total_backbones: int, backbone_index: int) -> int:
    remaining_backbones = int(total_backbones) - int(backbone_index)
    if int(remaining) < 1 or remaining_backbones < 1:
        raise ValueError("invalid balanced-quota state")
    return max(1, math.ceil(int(remaining) / remaining_backbones))


@dataclass
class FrozenDiffAMProblem:
    backbone: Candidate
    graph: CompressorGraph
    tree: diff_sim.TreeSim
    pp_specs: dict
    solver: GradientCellSolver


class DiffAMStage2Search:
    """Generate Stage-2 candidates only through hard-forward STE optimization."""

    def __init__(self, engine, config, run_dir: str):
        self.engine = engine
        self.cfg = config
        self.run_dir = os.path.abspath(run_dir)
        self.device = str(config.stage2_diffam_device)
        if self.device not in ("cpu", "cuda:0", "cuda:2"):
            raise ValueError(
                "Stage-2 DiffAM device must be cpu, cuda:0, or cuda:2; "
                f"got {self.device!r}"
            )
        if int(config.stage2_diffam_steps) < 1:
            raise ValueError("stage2_diffam_steps must be positive")
        if int(config.stage2_diffam_budget_count) < 1:
            raise ValueError("stage2_diffam_budget_count must be positive")
        if int(config.stage2_diffam_samples) < 1:
            raise ValueError("stage2_diffam_samples must be positive")
        if int(config.stage2_diffam_vectors) < 65536:
            raise ValueError("stage2_diffam_vectors must be at least 65536")
        if float(config.stage2_diffam_lr) <= 0:
            raise ValueError("stage2_diffam_lr must be positive")
        if not 0 < float(config.stage2_diffam_tau_min) <= 1:
            raise ValueError("stage2_diffam_tau_min must be in (0, 1]")
        self.budgets = np.geomspace(
            float(config.mred_lo),
            float(config.mred_hi),
            int(config.stage2_diffam_budget_count),
        ).tolist()
        self.contexts: Dict[str, FrozenDiffAMProblem] = {}
        self.shared_estimator = None

    def _build_context(self, backbone: Candidate) -> FrozenDiffAMProblem:
        cached = self.contexts.get(backbone.structure_hash)
        if cached is not None:
            return cached
        self.engine._activate_trunc_profile(backbone.k)
        pp = np.asarray(self.engine.initial_pp, dtype=int)
        compressor_tree = CompressorTree(
            pp,
            np.asarray(backbone.ct32, dtype=int),
            np.asarray(backbone.ct22, dtype=int),
            np.asarray(backbone.ct42, dtype=int),
        )
        assignment = compressor_tree.compressor_assignment_fused()
        graph = CompressorGraph(
            pp,
            assignment,
            num_node_types=self.engine.num_node_types,
        )
        routing = CanonicalRouter().route(graph)
        compressor_tree.trunc_cols = int(backbone.k)
        compressor_tree.trunc_bits = dict(self.engine._trunc_bits)
        pp_specs = diff_sim.parse_pp_specs(
            Mul(
                self.engine.bit_width,
                self.engine.encode_type,
                compressor_tree,
            ).emit_pp_encoder()
        )
        expected_pp = int(pp.sum())
        if len(pp_specs) != expected_pp:
            raise RuntimeError(
                f"partial-product parser mismatch: {len(pp_specs)}/{expected_pp}"
            )
        tree = diff_sim.TreeSim(graph, routing, pp_specs, self.device)
        solver_kwargs = {
            "device": self.device,
            "pool_vectors": int(self.cfg.stage2_diffam_vectors),
            "seed": int(self.cfg.stage2_diffam_vector_seed),
            "cache_dir": os.path.join(self.run_dir, "stage2", "diffam_cache"),
        }
        if self.shared_estimator is not None:
            solver_kwargs["est"] = self.shared_estimator
        solver = GradientCellSolver(
            self.engine,
            tree,
            pp_specs,
            max(self.budgets),
            **solver_kwargs,
        )
        if self.shared_estimator is None:
            self.shared_estimator = solver.est
        context = FrozenDiffAMProblem(backbone, graph, tree, pp_specs, solver)
        self.contexts[backbone.structure_hash] = context
        return context

    @staticmethod
    def _config_from_candidate(
        candidate: Candidate, graph: CompressorGraph
    ) -> CellConfig:
        config: CellConfig = {}
        for entry in candidate.cells:
            slot = tuple(int(value) for value in entry[:4])
            node = graph.indice_map.get(slot)
            if node is None:
                continue
            config[int(node)] = (int(entry[2]), int(entry[4]))
        return config

    @staticmethod
    def _candidate_from_config(
        context: FrozenDiffAMProblem,
        config: CellConfig,
        metadata: dict,
    ) -> Candidate:
        cells = []
        for node, (expected_type, cell_type) in sorted(config.items()):
            stage, column, graph_type, local_index = context.graph.vertex_list[
                int(node)
            ]
            if int(graph_type) != int(expected_type):
                raise ValueError(
                    f"node {node} type mismatch: config={expected_type}, graph={graph_type}"
                )
            if int(cell_type):
                cells.append(
                    [
                        int(stage),
                        int(column),
                        int(graph_type),
                        int(local_index),
                        int(cell_type),
                    ]
                )
        candidate = context.backbone.clone(stage=2)
        candidate.cells = sorted(cells, key=lambda item: tuple(item))
        candidate.routing = None
        candidate.operator = "diffam_ste"
        candidate.metadata.update(copy.deepcopy(metadata))
        candidate.refresh_id()
        if candidate.structure_hash != context.backbone.structure_hash:
            raise AssertionError("DiffAM changed the fixed Stage-1 structure")
        return candidate

    def _train(
        self,
        context: FrozenDiffAMProblem,
        budget: float,
        seed: int,
        warm_start: Candidate | None,
    ) -> List[Tuple[CellConfig, dict]]:
        self.engine._activate_trunc_profile(context.backbone.k)
        solver = GradientCellSolver(
            self.engine,
            context.tree,
            context.pp_specs,
            budget,
            device=self.device,
            est=context.solver.est,
        )
        solver.est.rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        with torch.no_grad():
            solver.logits.normal_(
                mean=0.0,
                std=float(self.cfg.stage2_diffam_init_std),
                generator=generator,
            )
            solver.logits[:, 0] += float(self.cfg.stage2_diffam_exact_bias)
            if warm_start is not None:
                warm_config = self._config_from_candidate(warm_start, context.graph)
                slot_rows = {
                    int(node): row
                    for row, (node, _t, _column) in enumerate(solver.space.slots)
                }
                for node, (_t, cell_type) in warm_config.items():
                    row = slot_rows.get(int(node))
                    if row is not None and cell_type < solver.logits.shape[1]:
                        solver.logits[row, int(cell_type)] += float(
                            self.cfg.stage2_diffam_warm_bias
                        )

        optimizer = torch.optim.Adam(
            [solver.logits],
            lr=float(self.cfg.stage2_diffam_lr),
        )
        lam = float(self.cfg.stage2_diffam_lam0)
        proposals: List[Tuple[CellConfig, dict]] = []
        last_key = None
        steps = int(self.cfg.stage2_diffam_steps)
        dual_every = max(1, int(self.cfg.stage2_diffam_dual_every))
        for step in range(steps):
            tau = max(
                float(self.cfg.stage2_diffam_tau_min),
                1.0
                - (1.0 - float(self.cfg.stage2_diffam_tau_min))
                * step
                / max(steps - 1, 1),
            )
            selection = solver.weights(tau)
            selected_tables = solver.sel_dict(selection)
            ratio_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            for a, b, golden, weight in solver.est.train_batch():
                pp_bits = diff_sim.compute_pp_bits(
                    solver.pp_specs,
                    a,
                    b,
                    self.engine.bit_width,
                    self.device,
                )
                output = solver.tree.eval_diff(pp_bits, selected_tables)
                ratio_sum = ratio_sum + weight * solver.est._ratio_sum_diff(
                    output, golden
                )
            mred = ratio_sum / solver.est.n_rel
            area = solver.area_term(selection)
            loss = area + lam * torch.relu(mred / budget - 1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            config = solver.hard_config()
            key = _config_key(config)
            if key != last_key:
                proposals.append(
                    (
                        dict(config),
                        {
                            "source": "trajectory",
                            "step": step + 1,
                            "tau": float(tau),
                            "train_mred": float(mred.detach()),
                            "train_area_fraction": float(area.detach()),
                        },
                    )
                )
                last_key = key
            if (step + 1) % dual_every == 0:
                hard_mred = solver.gate_mred(config)
                lam = max(
                    5.0,
                    lam
                    + float(self.cfg.stage2_diffam_lam_step)
                    * (hard_mred / budget - 1.0),
                )

        proposals.append(
            (dict(solver.hard_config()), {"source": "hard_final", "step": steps})
        )
        with torch.no_grad():
            masked = solver.logits + solver.mask
            for temperature in (0.35, 0.60, 1.0, 1.6):
                probabilities = torch.softmax(masked / temperature, dim=-1)
                for sample_id in range(int(self.cfg.stage2_diffam_samples)):
                    config: CellConfig = {}
                    for row, (node, cell_kind, _column) in enumerate(
                        solver.space.slots
                    ):
                        cell_type = int(
                            torch.multinomial(
                                probabilities[row],
                                1,
                                generator=generator,
                            ).item()
                        )
                        if cell_type:
                            config[int(node)] = (int(cell_kind), cell_type)
                    proposals.append(
                        (
                            config,
                            {
                                "source": "logit_sample",
                                "temperature": float(temperature),
                                "sample_id": sample_id,
                            },
                        )
                    )
        for _config, metadata in proposals:
            metadata.update({"target_mred": float(budget), "seed": int(seed)})
        return proposals

    def _select(
        self,
        context: FrozenDiffAMProblem,
        raw: Iterable[Tuple[CellConfig, dict]],
        budget: float,
        size: int,
        excluded_hashes: set[str],
    ) -> List[Candidate]:
        unique = {}
        for config, metadata in [({}, {"source": "exact"}), *list(raw)]:
            key = _config_key(config)
            unique.setdefault(key, (dict(config), copy.deepcopy(metadata)))

        exact_full = context.solver.gate_mred({})
        exact_screen = context.solver.gate_screen({})
        screen_offset = exact_full - exact_screen
        screened = []
        for config, metadata in unique.values():
            candidate = self._candidate_from_config(context, config, metadata)
            if candidate.cell_hash in excluded_hashes:
                continue
            proxy_screen = context.solver.gate_screen(config) + screen_offset
            screened.append(
                (
                    config,
                    metadata,
                    candidate,
                    float(proxy_screen),
                    float(context.solver.area_saving(config)),
                )
            )

        def screen_rank(item):
            proxy_mred, area_saving = item[3], item[4]
            feasible = proxy_mred <= budget * 1.10
            if feasible:
                return (0, -area_saving, abs(proxy_mred - budget), _config_key(item[0]))
            # The all-exact configuration is only a baseline.  When even it is
            # over budget, an approximate configuration may still lower MRED
            # through error cancellation, so rank infeasible points by MRED.
            return (1, proxy_mred, -area_saving, _config_key(item[0]))

        screened.sort(key=screen_rank)
        shortlist = screened[: max(size * 2, size)]
        gated = []
        for config, metadata, candidate, proxy_screen, area_saving in shortlist:
            proxy_mred = context.solver.gate_mred(config)
            candidate.metadata.update(
                {
                    "method": "diffam",
                    "proxy_screen_mred": proxy_screen,
                    "proxy_mred": float(proxy_mred),
                    "proxy_area_saving": area_saving,
                }
            )
            gated.append((config, candidate))

        def gate_rank(item):
            candidate = item[1]
            proxy_mred = float(candidate.metadata["proxy_mred"])
            area_saving = float(candidate.metadata["proxy_area_saving"])
            if proxy_mred <= budget:
                return (0, -area_saving, abs(proxy_mred - budget), candidate.cell_hash)
            return (1, proxy_mred, -area_saving, candidate.cell_hash)

        gated.sort(key=gate_rank)

        selected: List[Tuple[CellConfig, Candidate]] = []
        remaining = list(gated)
        while remaining and len(selected) < size:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            position = max(
                range(len(remaining)),
                key=lambda index: (
                    min(
                        _config_distance(remaining[index][0], old[0])
                        for old in selected
                    ),
                    remaining[index][1].metadata["proxy_area_saving"],
                ),
            )
            selected.append(remaining.pop(position))
        output = [candidate for _config, candidate in selected]
        context.solver.clear_gate_pp()
        return output

    def propose(
        self,
        backbones: Sequence[Candidate],
        *,
        size: int,
        round_index: int,
        excluded_hashes: Iterable[str] = (),
        warm_starts: Sequence[Candidate] = (),
    ) -> List[Candidate]:
        unique_backbones = {}
        for backbone in backbones:
            unique_backbones.setdefault(backbone.structure_hash, backbone)
        backbone_list = list(unique_backbones.values())
        if not backbone_list:
            raise ValueError("DiffAM Stage 2 requires at least one backbone")
        excluded = set(excluded_hashes)
        warm_by_structure: Dict[str, List[Candidate]] = {}
        for candidate in warm_starts:
            warm_by_structure.setdefault(candidate.structure_hash, []).append(candidate)

        selected: List[Candidate] = []
        selected_hashes = set(excluded)
        attempts = 0
        maximum_attempts = max(4, int(self.cfg.stage2_diffam_restarts) + 3)
        while len(selected) < size and attempts < maximum_attempts:
            progress = False
            for backbone_index, backbone in enumerate(backbone_list):
                if len(selected) >= size:
                    break
                remaining = size - len(selected)
                quota = _balanced_quota(remaining, len(backbone_list), backbone_index)
                context = self._build_context(backbone)
                band = int(backbone.metadata.get("mred_band", backbone_index))
                budget_index = (band + round_index + attempts) % len(self.budgets)
                budget = float(self.budgets[budget_index])
                seed = (
                    int(self.cfg.seed)
                    + int(round_index) * 1_000_003
                    + attempts * 10_007
                    + int(backbone.structure_hash[:8], 16)
                )
                warm_pool = warm_by_structure.get(backbone.structure_hash, [])
                warm = warm_pool[attempts % len(warm_pool)] if warm_pool else None
                logging.info(
                    "[stage2/diffam] round=%d backbone=%s budget=%.3e "
                    "attempt=%d quota=%d",
                    round_index,
                    backbone.candidate_id,
                    budget,
                    attempts + 1,
                    quota,
                )
                raw = self._train(context, budget, seed, warm)
                candidates = self._select(
                    context,
                    raw,
                    budget,
                    min(quota, remaining),
                    selected_hashes,
                )
                for candidate in candidates:
                    if candidate.cell_hash in selected_hashes:
                        continue
                    candidate.metadata["diffam_round"] = int(round_index)
                    candidate.metadata["diffam_attempt"] = int(attempts)
                    candidate.metadata["generation"] = int(round_index)
                    selected.append(candidate)
                    selected_hashes.add(candidate.cell_hash)
                    progress = True
                    if len(selected) >= size:
                        break
            if not progress:
                break
            attempts += 1
        if len(selected) != size:
            raise RuntimeError(
                f"DiffAM generated only {len(selected)}/{size} unseen Stage-2 candidates"
            )
        return selected
