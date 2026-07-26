#!/usr/bin/env python3
"""Fair fixed-backbone cell-only GA vs DiffAM pilot.

The structure and canonical routing are frozen for both methods.  GA uses the
existing Stage-2 operators and exact DC/Verilator feedback.  DiffAM uses
per-slot categorical logits, hard one-hot forward, STE backward, exact truth
tables, and a stratified MRED estimator to propose the same number of
candidates.  Both methods are finally judged by the same V5CandidateEvaluator.

Primary metric:
  best measured area after N exact evaluations, subject to measured
  MRED <= budget and delay <= target.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Appr_Comp.cellsolver import sim as diff_sim  # noqa: E402
from Appr_Comp.cellsolver.solver import GradientCellSolver  # noqa: E402
from scripts.train_three_stage import build_engine  # noqa: E402
from trainer.arith_das_v5.compressor_graph import CompressorGraph  # noqa: E402
from trainer.arith_three_stage.bandit import ContextualThompsonBandit  # noqa: E402
from trainer.arith_three_stage.candidate import Candidate  # noqa: E402
from trainer.arith_three_stage.canonical_router import CanonicalRouter  # noqa: E402
from trainer.arith_three_stage.cell_ops import CELL_ARMS, CellOperator  # noqa: E402
from trainer.arith_three_stage.evaluator import V5CandidateEvaluator  # noqa: E402
from trainer.arith_three_stage.pareto import (  # noqa: E402
    assign_crowding,
    environmental_select,
    fast_non_dominated_sort,
    tournament,
)
from utils import CompressorTree, Mul  # noqa: E402


CellConfig = Dict[int, Tuple[int, int]]


def atomic_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def config_key(config: CellConfig) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(sorted((int(node), int(t), int(k)) for node, (t, k) in config.items()))


def config_distance(a: CellConfig, b: CellConfig) -> int:
    nodes = set(a) | set(b)
    return sum(a.get(n) != b.get(n) for n in nodes)


def parse_budgets(raw: str) -> List[float]:
    budgets = sorted({float(x) for x in raw.split(",") if x.strip()})
    if not budgets or budgets[0] <= 0:
        raise ValueError("--mred_budgets must contain positive values")
    return budgets


def make_engine_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        out=args.out,
        device=args.device,
        seed=args.seed,
        target_delay=args.target_delay,
        error_vectors=args.error_vectors,
        k_min=2,
        dc_batch=args.dc_batch,
        dc_parallelism=args.dc_parallelism,
        stage3_num_epochs=1,
        stage3_episodes_per_elite=1,
        stage3_single_elite_index=None,
        approx_col_window=args.approx_col_window,
        approx_lib_path=args.approx_lib_path,
        approx42_library_path=args.approx42_library_path,
        approx42_rtl_path=args.approx42_rtl_path,
    )


@dataclass
class FrozenContext:
    backbone: Candidate
    graph: CompressorGraph
    routing: list
    tree: diff_sim.TreeSim
    pp_specs: list
    base_solver: GradientCellSolver


def build_frozen_context(engine, backbone: Candidate, args) -> FrozenContext:
    engine._activate_trunc_profile(backbone.k)
    pp = np.asarray(engine.initial_pp, dtype=int)
    ct = CompressorTree(
        pp,
        np.asarray(backbone.ct32, dtype=int),
        np.asarray(backbone.ct22, dtype=int),
        np.asarray(backbone.ct42, dtype=int),
    )
    assignment = ct.compressor_assignment_fused()
    graph = CompressorGraph(pp, assignment, num_node_types=engine.num_node_types)
    routing = CanonicalRouter().route(graph)
    ct.trunc_cols = int(backbone.k)
    ct.trunc_bits = dict(engine._trunc_bits)
    pp_specs = diff_sim.parse_pp_specs(
        Mul(engine.bit_width, engine.encode_type, ct).emit_pp_encoder()
    )
    expected_pp = int(np.asarray(pp).sum())
    if len(pp_specs) != expected_pp:
        raise RuntimeError(f"partial-product parser mismatch: {len(pp_specs)}/{expected_pp}")
    tree = diff_sim.TreeSim(graph, routing, pp_specs, args.device)
    base_solver = GradientCellSolver(
        engine,
        tree,
        pp_specs,
        max(args.mred_budgets),
        device=args.device,
        pool_vectors=args.diffam_vectors,
        seed=args.vector_seed,
        cache_dir=os.path.join(args.out, "diffam_cache"),
    )
    return FrozenContext(backbone, graph, routing, tree, pp_specs, base_solver)


def candidate_from_config(
    backbone: Candidate,
    graph: CompressorGraph,
    config: CellConfig,
    *,
    operator: str,
    metadata: dict,
) -> Candidate:
    cells = []
    for node_idx, (expected_t, type_idx) in sorted(config.items()):
        stage, column, graph_t, local_idx = graph.vertex_list[int(node_idx)]
        if int(graph_t) != int(expected_t):
            raise ValueError(
                f"node {node_idx} type mismatch: config={expected_t}, graph={graph_t}"
            )
        if int(type_idx) != 0:
            cells.append(
                [
                    int(stage),
                    int(column),
                    int(graph_t),
                    int(local_idx),
                    int(type_idx),
                ]
            )
    candidate = backbone.clone(stage=2)
    candidate.cells = sorted(cells, key=lambda x: tuple(x))
    candidate.operator = operator
    candidate.metadata.update(copy.deepcopy(metadata))
    candidate.refresh_id()
    return candidate


def evaluate_exact(
    evaluator: V5CandidateEvaluator,
    candidates: Sequence[Candidate],
    *,
    tag: str,
) -> None:
    for batch_id in range(0, len(candidates), evaluator.batch_size):
        batch = list(candidates[batch_id : batch_id + evaluator.batch_size])
        logging.info(
            "[%s] exact evaluation %d..%d/%d",
            tag,
            batch_id,
            batch_id + len(batch) - 1,
            len(candidates),
        )
        evaluator.evaluate(batch)
        for retry_idx in range(2):
            retry = [
                c
                for c in batch
                if not c.evaluated
                and c.failure_reason in ("dc_failed", "verilator_failed")
            ]
            if not retry:
                break
            logging.warning("[%s] retry %d: %d candidates", tag, retry_idx + 1, len(retry))
            for candidate in retry:
                candidate.valid = True
                candidate.failure_reason = None
            evaluator.evaluate(retry)


def prepare_population(population: Sequence[Candidate], delay_limit: float) -> None:
    for front in fast_non_dominated_sort(population, delay_limit):
        assign_crowding(front)


def random_ga_seed(
    backbone: Candidate,
    operator: CellOperator,
    rng: random.Random,
    target: int,
) -> Candidate:
    candidate = backbone.clone(stage=2)
    candidate.cells = []
    for _ in range(int(target)):
        if rng.random() < 0.15 and operator._zero_toggle(candidate, rng):
            continue
        if not operator._add(candidate, rng, prefer_low=rng.random() < 0.60):
            break
    candidate.cells = sorted(candidate.cells, key=lambda x: tuple(x))
    candidate.operator = "ga_seed"
    candidate.metadata["target_cells"] = int(target)
    candidate.refresh_id()
    return candidate


def initial_ga_population(
    backbone: Candidate,
    operator: CellOperator,
    rng: random.Random,
    size: int,
) -> List[Candidate]:
    exact = backbone.clone(stage=2)
    exact.cells = []
    exact.operator = "ga_seed_exact"
    exact.refresh_id()
    population = [exact]
    seen = {exact.cell_hash}
    targets = [1, 1, 2, 2, 4, 4, 6, 8, 8, 12, 12, 16, 20, 24]
    attempts = 0
    while len(population) < size and attempts < 100_000:
        attempts += 1
        target = targets[(len(population) - 1) % len(targets)]
        target = max(1, target + rng.choice((-1, 0, 0, 0, 1)))
        candidate = random_ga_seed(backbone, operator, rng, target)
        if candidate.cell_hash in seen:
            continue
        seen.add(candidate.cell_hash)
        population.append(candidate)
    if len(population) != size:
        raise RuntimeError(f"could only generate {len(population)}/{size} GA seeds")
    return population


def run_ga(
    backbone: Candidate,
    operator: CellOperator,
    evaluator: V5CandidateEvaluator,
    args,
) -> List[Candidate]:
    rng = random.Random(args.seed)
    bandit = ContextualThompsonBandit(CELL_ARMS, window=128, explore=0.03)
    population = initial_ga_population(backbone, operator, rng, args.population)
    for candidate in population:
        candidate.metadata["method"] = "ga"
        candidate.metadata["generation"] = 0
    evaluate_exact(evaluator, population, tag="ga/gen0")
    population = environmental_select(population, args.population, args.target_delay)
    evaluated = list(population)
    global_seen = {candidate.cell_hash for candidate in evaluated}

    for generation in range(1, args.ga_generations + 1):
        prepare_population(population, args.target_delay)
        offspring: List[Candidate] = []
        attempts = 0
        while len(offspring) < args.population and attempts < 200_000:
            attempts += 1
            parent_a = tournament(population, rng)
            parent_b = tournament(population, rng)
            if rng.random() < 0.9:
                base = operator.crossover_a(parent_a, parent_b, rng)
            else:
                base = parent_a.clone(stage=2)
            legal = operator.legal_arms(base)
            if not legal:
                continue
            mred = max(float(parent_a.mred or args.mred_budgets[0]), 1e-15)
            context = f"m{math.floor(math.log10(mred))}"
            arm = bandit.choose(context, legal, rng)
            child = operator.mutate(base, arm, rng)
            if child is None or child.cell_hash in global_seen:
                continue
            child.operator_context = context
            child.metadata["method"] = "ga"
            child.metadata["generation"] = generation
            global_seen.add(child.cell_hash)
            offspring.append(child)
        if len(offspring) != args.population:
            raise RuntimeError(
                f"GA generation {generation}: {len(offspring)}/{args.population} offspring"
            )
        evaluate_exact(evaluator, offspring, tag=f"ga/gen{generation}")
        next_population = environmental_select(
            list(population) + offspring, args.population, args.target_delay
        )
        survivor_hashes = {candidate.cell_hash for candidate in next_population}
        for child in offspring:
            if child.evaluated and child.operator and child.operator_context:
                bandit.update(
                    child.operator_context,
                    child.operator,
                    child.cell_hash in survivor_hashes,
                )
        population = next_population
        evaluated.extend(offspring)
    return evaluated


def sample_learned_configs(
    solver: GradientCellSolver,
    *,
    samples: int,
    seed: int,
) -> List[Tuple[CellConfig, dict]]:
    generator = torch.Generator(device=solver.device)
    generator.manual_seed(int(seed))
    out = []
    with torch.no_grad():
        masked = solver.logits + solver.mask
        for temperature in (0.35, 0.60, 1.0, 1.6):
            probabilities = torch.softmax(masked / temperature, dim=-1)
            for sample_id in range(samples):
                config: CellConfig = {}
                for slot_id, (node, t, _column) in enumerate(solver.space.slots):
                    k = int(
                        torch.multinomial(
                            probabilities[slot_id],
                            1,
                            generator=generator,
                        ).item()
                    )
                    if k:
                        config[int(node)] = (int(t), k)
                out.append(
                    (
                        config,
                        {
                            "source": "learned_sample",
                            "sample_temperature": temperature,
                            "sample_id": sample_id,
                        },
                    )
                )
    return out


def train_diffam_run(
    base: GradientCellSolver,
    *,
    budget: float,
    restart: int,
    args,
) -> List[Tuple[CellConfig, dict]]:
    solver = GradientCellSolver(
        base.exp,
        base.tree,
        base.pp_specs,
        budget,
        device=base.device,
        est=base.est,
    )
    seed = int(args.seed + round(-math.log10(budget) * 10_000) + restart * 1009)
    solver.est.rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device=solver.device)
    torch_generator.manual_seed(seed)
    with torch.no_grad():
        solver.logits.normal_(mean=0.0, std=args.diffam_init_std, generator=torch_generator)
        solver.logits[:, 0] += args.diffam_exact_bias

    optimizer = torch.optim.Adam([solver.logits], lr=args.diffam_lr)
    lam = float(args.diffam_lam0)
    proposals: List[Tuple[CellConfig, dict]] = []
    last_key = None
    for step in range(args.diffam_steps):
        tau = max(
            args.diffam_tau_min,
            1.0
            - (1.0 - args.diffam_tau_min)
            * step
            / max(args.diffam_steps - 1, 1),
        )
        selection = solver.weights(tau)
        selected_tables = solver.sel_dict(selection)
        ratio_sum = torch.zeros((), dtype=torch.float64, device=solver.device)
        for a, b, golden, weight in solver.est.train_batch():
            pp = diff_sim.compute_pp_bits(
                solver.pp_specs,
                a,
                b,
                solver.exp.bit_width,
                solver.device,
            )
            approximate = solver.tree.eval_diff(pp, selected_tables)
            ratio_sum = ratio_sum + weight * solver.est._ratio_sum_diff(
                approximate, golden
            )
        mred = ratio_sum / solver.est.n_rel
        area = solver.area_term(selection)
        loss = area + lam * torch.relu(mred / budget - 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        config = solver.hard_config()
        key = config_key(config)
        if key != last_key:
            proposals.append(
                (
                    dict(config),
                    {
                        "source": "trajectory",
                        "step": step + 1,
                        "tau": tau,
                        "train_mred": float(mred.detach()),
                        "train_area_fraction": float(area.detach()),
                    },
                )
            )
            last_key = key
        if (step + 1) % args.diffam_dual_every == 0:
            gate_mred = solver.gate_mred(config)
            lam = max(5.0, lam + args.diffam_lam_step * (gate_mred / budget - 1.0))

    repaired = solver.repair(solver.hard_config(), log=lambda *_a, **_k: None)
    proposals.append(
        (
            dict(repaired),
            {
                "source": "repaired_final",
                "step": args.diffam_steps,
            },
        )
    )
    proposals.extend(
        sample_learned_configs(
            solver,
            samples=args.diffam_samples_per_temperature,
            seed=seed + 17,
        )
    )
    for _config, metadata in proposals:
        metadata["target_mred"] = float(budget)
        metadata["restart"] = int(restart)
        metadata["seed"] = int(seed)
    return proposals


def proxy_select_diffam(
    base: GradientCellSolver,
    raw: Iterable[Tuple[CellConfig, dict]],
    *,
    budgets: Sequence[float],
    size: int,
    proxy_gate_multiplier: int,
) -> List[Tuple[CellConfig, dict]]:
    unique: Dict[Tuple[Tuple[int, int, int], ...], Tuple[CellConfig, dict]] = {}
    exact_key = config_key({})
    unique[exact_key] = ({}, {"source": "exact", "target_mred": budgets[0]})
    for config, metadata in raw:
        key = config_key(config)
        old = unique.get(key)
        if old is None or metadata.get("source") == "repaired_final":
            unique[key] = (dict(config), copy.deepcopy(metadata))

    exact_full = base.gate_mred({})
    exact_screen = base.gate_screen({})
    screen_offset = exact_full - exact_screen
    screened = []
    for config, metadata in unique.values():
        screen_mred = base.gate_screen(config) + screen_offset
        saving = base.area_saving(config)
        item = copy.deepcopy(metadata)
        item["proxy_screen_mred"] = float(screen_mred)
        item["proxy_area_saving"] = float(saving)
        screened.append((config, item))

    max_full = max(size * int(proxy_gate_multiplier), size)
    shortlist_keys = {exact_key}
    for budget in budgets:
        feasible = sorted(
            (
                (config, metadata)
                for config, metadata in screened
                if metadata["proxy_screen_mred"] <= budget * 1.10
            ),
            key=lambda item: (
                -item[1]["proxy_area_saving"],
                abs(item[1]["proxy_screen_mred"] - budget),
                config_key(item[0]),
            ),
        )
        for config, _metadata in feasible[: max(8, max_full // len(budgets))]:
            shortlist_keys.add(config_key(config))
    if len(shortlist_keys) < max_full:
        for config, _metadata in sorted(
            screened,
            key=lambda item: (
                min(abs(math.log(max(item[1]["proxy_screen_mred"], 1e-15) / b)) for b in budgets),
                -item[1]["proxy_area_saving"],
            ),
        ):
            shortlist_keys.add(config_key(config))
            if len(shortlist_keys) >= max_full:
                break

    fully_gated = []
    for config, metadata in screened:
        if config_key(config) not in shortlist_keys:
            continue
        item = copy.deepcopy(metadata)
        item["proxy_mred"] = float(base.gate_mred(config))
        fully_gated.append((config, item))

    buckets = {}
    for budget in budgets:
        buckets[budget] = sorted(
            (
                (config, metadata)
                for config, metadata in fully_gated
                if metadata["proxy_mred"] <= budget
            ),
            key=lambda item: (
                -item[1]["proxy_area_saving"],
                abs(item[1]["proxy_mred"] - budget),
                config_key(item[0]),
            ),
        )

    selected: List[Tuple[CellConfig, dict]] = []
    selected_keys = set()
    cursors = {budget: 0 for budget in budgets}
    while len(selected) < size:
        progress = False
        for budget in budgets:
            bucket = buckets[budget]
            while cursors[budget] < len(bucket):
                config, metadata = bucket[cursors[budget]]
                cursors[budget] += 1
                key = config_key(config)
                if key in selected_keys:
                    continue
                item = copy.deepcopy(metadata)
                item["selection_budget"] = float(budget)
                selected.append((config, item))
                selected_keys.add(key)
                progress = True
                break
            if len(selected) >= size:
                break
        if not progress:
            break

    if len(selected) < size:
        remaining = sorted(
            (
                (config, metadata)
                for config, metadata in fully_gated
                if config_key(config) not in selected_keys
            ),
            key=lambda item: (
                item[1]["proxy_mred"] > budgets[-1],
                -item[1]["proxy_area_saving"],
                item[1]["proxy_mred"],
            ),
        )
        for config, metadata in remaining:
            selected.append((config, copy.deepcopy(metadata)))
            selected_keys.add(config_key(config))
            if len(selected) >= size:
                break

    if len(selected) < size:
        raise RuntimeError(
            f"DiffAM produced only {len(selected)}/{size} proxy-selected unique configs "
            f"from {len(unique)} raw unique configs"
        )

    # Improve early-batch diversity without using any measured DC result.
    ordered: List[Tuple[CellConfig, dict]] = []
    remaining = list(selected[:size])
    if remaining:
        exact_pos = next(
            (i for i, (config, _metadata) in enumerate(remaining) if not config),
            None,
        )
        if exact_pos is not None:
            ordered.append(remaining.pop(exact_pos))
    while remaining:
        if not ordered:
            ordered.append(remaining.pop(0))
            continue
        best_pos = max(
            range(len(remaining)),
            key=lambda i: (
                min(config_distance(remaining[i][0], x[0]) for x in ordered),
                remaining[i][1].get("proxy_area_saving", 0.0),
            ),
        )
        ordered.append(remaining.pop(best_pos))
    return ordered


def propose_diffam(context: FrozenContext, args) -> List[Candidate]:
    raw = []
    for budget in args.mred_budgets:
        for restart in range(args.diffam_restarts):
            logging.info(
                "[diffam] budget=%.3e restart=%d/%d",
                budget,
                restart + 1,
                args.diffam_restarts,
            )
            raw.extend(
                train_diffam_run(
                    context.base_solver,
                    budget=budget,
                    restart=restart,
                    args=args,
                )
            )
    selected = proxy_select_diffam(
        context.base_solver,
        raw,
        budgets=args.mred_budgets,
        size=args.eval_budget,
        proxy_gate_multiplier=args.proxy_gate_multiplier,
    )
    candidates = []
    for proposal_id, (config, metadata) in enumerate(selected):
        metadata = copy.deepcopy(metadata)
        metadata.update(
            {
                "method": "diffam",
                "proposal_id": proposal_id,
                "n_cells": len(config),
            }
        )
        candidates.append(
            candidate_from_config(
                context.backbone,
                context.graph,
                config,
                operator="diffam_ste",
                metadata=metadata,
            )
        )
    return candidates


def save_candidates(path: str, candidates: Sequence[Candidate]) -> None:
    atomic_json(path, [candidate.to_dict() for candidate in candidates])


def measured_front(candidates: Iterable[Candidate], delay_limit: float) -> List[Candidate]:
    feasible = [
        candidate
        for candidate in candidates
        if candidate.evaluated
        and candidate.valid
        and float(candidate.delay) <= delay_limit
    ]
    fronts = fast_non_dominated_sort(feasible, delay_limit)
    return fronts[0] if fronts else []


def best_at_budget(
    candidates: Sequence[Candidate],
    n: int,
    mred_budget: float,
    delay_limit: float,
) -> Tuple[float | None, float | None, str | None]:
    feasible = [
        candidate
        for candidate in candidates[:n]
        if candidate.evaluated
        and candidate.valid
        and float(candidate.delay) <= delay_limit
        and float(candidate.mred) <= mred_budget
    ]
    if not feasible:
        return None, None, None
    best_area = min(feasible, key=lambda candidate: float(candidate.area))
    best_power = min(feasible, key=lambda candidate: float(candidate.power))
    return (
        float(best_area.area),
        float(best_power.power),
        best_area.candidate_id,
    )


def analyze(args, ga: Sequence[Candidate], diffam: Sequence[Candidate]) -> dict:
    checkpoints = list(range(args.population, args.eval_budget + 1, args.population))
    if checkpoints[-1] != args.eval_budget:
        checkpoints.append(args.eval_budget)
    rows = []
    for method, candidates in (("ga", ga), ("diffam", diffam)):
        for n in checkpoints:
            for budget in args.mred_budgets:
                area, power, candidate_id = best_at_budget(
                    candidates, n, budget, args.target_delay
                )
                rows.append(
                    {
                        "method": method,
                        "n_evaluated": n,
                        "mred_budget": budget,
                        "best_area": area,
                        "best_power": power,
                        "best_area_candidate_id": candidate_id,
                    }
                )
    summary = {
        "settings": {
            "backbone_file": args.backbones,
            "backbone_index": args.backbone_index,
            "population": args.population,
            "ga_generations": args.ga_generations,
            "eval_budget_per_method": args.eval_budget,
            "mred_budgets": args.mred_budgets,
            "dc_batch": args.dc_batch,
            "dc_parallelism": args.dc_parallelism,
            "error_vectors": args.error_vectors,
            "diffam_vectors": args.diffam_vectors,
            "device": args.device,
        },
        "curves": rows,
        "front_sizes": {
            "ga": len(measured_front(ga, args.target_delay)),
            "diffam": len(measured_front(diffam, args.target_delay)),
        },
    }
    atomic_json(os.path.join(args.out, "summary.json"), summary)
    with open(os.path.join(args.out, "curves.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def print_summary(summary: dict) -> None:
    rows = summary["curves"]
    checkpoints = sorted({row["n_evaluated"] for row in rows})
    budgets = sorted({row["mred_budget"] for row in rows})
    print("\nMeasured best area (delay-feasible and MRED-constrained)")
    print(f"{'evals':>7} {'budget':>11} {'GA':>12} {'DiffAM':>12} {'winner':>9}")
    for n in checkpoints:
        for budget in budgets:
            by_method = {
                row["method"]: row
                for row in rows
                if row["n_evaluated"] == n and row["mred_budget"] == budget
            }
            ga = by_method["ga"]["best_area"]
            diffam = by_method["diffam"]["best_area"]
            if ga is None and diffam is None:
                winner = "-"
            elif ga is None:
                winner = "DiffAM"
            elif diffam is None:
                winner = "GA"
            elif diffam < ga:
                winner = "DiffAM"
            elif ga < diffam:
                winner = "GA"
            else:
                winner = "tie"
            ga_text = "-" if ga is None else f"{ga:.3f}"
            diffam_text = "-" if diffam is None else f"{diffam:.3f}"
            print(f"{n:7d} {budget:11.3e} {ga_text:>12} {diffam_text:>12} {winner:>9}")
    print(
        "Measured Pareto front sizes:",
        summary["front_sizes"],
    )


def validate_args(args) -> None:
    if args.dc_parallelism > 32 or args.dc_batch > 32:
        raise ValueError("this experiment is capped at 32-way DC parallelism/batch")
    if args.device not in ("cuda:0", "cuda:2"):
        raise ValueError("GPU workloads may use only cuda:0 or cuda:2")
    expected_budget = args.population * (args.ga_generations + 1)
    if args.eval_budget != expected_budget:
        raise ValueError(
            "--eval_budget must equal population*(ga_generations+1): "
            f"{args.eval_budget} != {expected_budget}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--backbones",
        default=(
            "outputs/2026-07-20_1808_arith_three_stage_init_probe/"
            "stage1/backbones_32.json"
        ),
    )
    parser.add_argument("--backbone_index", type=int, default=16)
    parser.add_argument(
        "--config",
        default="configs/config_groups/mul_16_approx_error_obj.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vector_seed", type=int, default=12345)
    parser.add_argument("--target_delay", type=float, default=1.5)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--ga_generations", type=int, default=2)
    parser.add_argument("--eval_budget", type=int, default=96)
    parser.add_argument("--dc_batch", type=int, default=32)
    parser.add_argument("--dc_parallelism", type=int, default=32)
    parser.add_argument("--error_vectors", type=int, default=16_000_000)
    parser.add_argument("--diffam_vectors", type=int, default=16_000_000)
    parser.add_argument(
        "--mred_budgets",
        default="1.0e-3,1.5e-3,2.2e-3,3.3e-3,5.0e-3",
    )
    parser.add_argument("--diffam_steps", type=int, default=80)
    parser.add_argument("--diffam_restarts", type=int, default=2)
    parser.add_argument("--diffam_lr", type=float, default=0.03)
    parser.add_argument("--diffam_lam0", type=float, default=50.0)
    parser.add_argument("--diffam_lam_step", type=float, default=100.0)
    parser.add_argument("--diffam_dual_every", type=int, default=10)
    parser.add_argument("--diffam_tau_min", type=float, default=0.25)
    parser.add_argument("--diffam_init_std", type=float, default=0.70)
    parser.add_argument("--diffam_exact_bias", type=float, default=0.80)
    parser.add_argument("--diffam_samples_per_temperature", type=int, default=12)
    parser.add_argument("--proxy_gate_multiplier", type=int, default=3)
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
    parser.add_argument(
        "--base_dir_dc",
        default="/home/lchangxian/sandbox/sandbox_base_dcpwr",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="generate both methods' candidates but skip DC/Verilator",
    )
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    args.mred_budgets = parse_budgets(args.mred_budgets)
    validate_args(args)

    os.makedirs(args.out, exist_ok=True)
    os.environ["EDA_BASE_DIR_DC"] = args.base_dir_dc
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.out, "experiment.log")),
        ],
    )
    atomic_json(os.path.join(args.out, "arguments.json"), vars(args))

    all_backbones = json.load(open(args.backbones))
    if not 0 <= args.backbone_index < len(all_backbones):
        raise IndexError(f"backbone index {args.backbone_index}/{len(all_backbones)}")
    backbone = Candidate.from_dict(all_backbones[args.backbone_index])
    if backbone.cells:
        raise ValueError("pilot requires an all-exact Stage-1 backbone")
    logging.info(
        "fixed backbone index=%d id=%s k=%d area=%.3f power=%.6g delay=%.3f mred=%.3e",
        args.backbone_index,
        backbone.candidate_id,
        backbone.k,
        float(backbone.area),
        float(backbone.power),
        float(backbone.delay),
        float(backbone.mred),
    )

    engine = build_engine(make_engine_args(args))
    context = build_frozen_context(engine, backbone, args)
    logging.info(
        "frozen graph nodes=%d legal_slots=%d tables=(T32=%d,T22=%d,T42=%d) "
        "diffam_exact_proxy_mred=%.3e",
        len(context.graph.vertex_list),
        len(context.base_solver.space.slots),
        len(engine.type_table_32),
        len(engine.type_table_22),
        len(engine.type_table_42),
        context.base_solver.gate_mred({}),
    )

    t0 = time.time()
    diffam_candidates = propose_diffam(context, args)
    logging.info(
        "DiffAM proposed %d candidates in %.1fs",
        len(diffam_candidates),
        time.time() - t0,
    )
    save_candidates(
        os.path.join(args.out, "diffam_candidates_proposed.json"),
        diffam_candidates,
    )
    if args.prepare_only:
        operator = CellOperator(engine)
        ga_candidates = initial_ga_population(backbone, operator, random.Random(args.seed), args.population)
        save_candidates(os.path.join(args.out, "ga_initial_candidates.json"), ga_candidates)
        logging.info("prepare-only complete")
        return

    evaluator = V5CandidateEvaluator(
        engine,
        os.path.join(args.out, "exact_eval"),
        batch_size=args.dc_batch,
        n_processing=args.dc_parallelism,
        target_delay=args.target_delay,
        error_vectors=args.error_vectors,
    )
    try:
        operator = CellOperator(engine)
        ga_candidates = run_ga(backbone, operator, evaluator, args)
        if len(ga_candidates) != args.eval_budget:
            raise RuntimeError(f"GA exact budget mismatch: {len(ga_candidates)}")
        save_candidates(os.path.join(args.out, "ga_candidates_measured.json"), ga_candidates)

        evaluate_exact(evaluator, diffam_candidates, tag="diffam")
        if len(diffam_candidates) != args.eval_budget:
            raise RuntimeError(
                f"DiffAM exact budget mismatch: {len(diffam_candidates)}"
            )
        save_candidates(
            os.path.join(args.out, "diffam_candidates_measured.json"),
            diffam_candidates,
        )
    finally:
        evaluator.close()

    summary = analyze(args, ga_candidates, diffam_candidates)
    print_summary(summary)
    logging.info("experiment complete: %s", args.out)


if __name__ == "__main__":
    main()
