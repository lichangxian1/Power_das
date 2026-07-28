"""Additional Stage-2 search loops kept separate from the main runner."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Sequence

from .candidate import Candidate
from .diffam_pipeline import DiffAMProxyProducer
from .pareto import environmental_select
from .selection import select_banded
from .stage2_cem import CellCEMStage2Search


def _atomic_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _load_candidates(path: str) -> List[Candidate]:
    with open(path) as stream:
        return [Candidate.from_dict(payload) for payload in json.load(stream)]


def _seed_population(runner, backbones: Sequence[Candidate]) -> List[Candidate]:
    population = []
    for backbone in backbones:
        population.extend(
            runner.cell_operator.make_seed_variants(backbone, runner.rng)
        )
    if len(population) != int(runner.cfg.population_size):
        raise RuntimeError(
            f"Stage 2 expected {runner.cfg.population_size} common seeds, "
            f"got {len(population)}"
        )
    return population


def _finalize(runner, population, archive, done: str, label: str):
    elites = select_banded(
        list(archive.items) + list(population),
        n_bins=runner.cfg.handoff_bins,
        roles=("area", "power", "knee"),
        mred_lo=runner.cfg.mred_lo,
        mred_hi=runner.cfg.mred_hi,
        delay_limit=runner.cfg.delay_limit,
    )
    _atomic_json(done, [candidate.to_dict() for candidate in elites])
    logging.info("%s Stage 2 complete: selected %d elites", label, len(elites))
    return elites


def run_stage2_cem(runner, backbones: Sequence[Candidate]) -> List[Candidate]:
    done = os.path.join(runner.run_dir, "stage2", "elites_24.json")
    if os.path.exists(done):
        logging.info("Stage 2 already complete; loading %s", done)
        return _load_candidates(done)
    searcher = CellCEMStage2Search(runner.engine, runner.cfg)
    restored = runner._restore_generation(2)
    if restored is None:
        population = _seed_population(runner, backbones)
        runner._evaluate_with_retries(population)
        if sum(candidate.evaluated for candidate in population) < len(population):
            raise RuntimeError("CEM Stage 2 common seeds have unresolved failures")
        searcher.update(population)
        seen_hashes = {candidate.cell_hash for candidate in population}
        population = environmental_select(
            population,
            runner.cfg.population_size,
            runner.cfg.delay_limit,
        )
        archive = runner._new_archive(2)
        archive.update(population)
        start_generation = 0
        runner._save_generation(
            2,
            0,
            population,
            archive,
            runner.cell_bandit,
            extra_state={
                "search_mode": "cem",
                "seen_cell_hashes": sorted(seen_hashes),
                "cem_state": searcher.state_dict(),
            },
        )
    else:
        start_generation, population, archive = restored
        extra = runner._restored_extra_state.get(2, {})
        if extra.get("search_mode") != "cem":
            raise RuntimeError("Stage 2 checkpoint was not created by cell CEM")
        seen_hashes = set(extra.get("seen_cell_hashes") or ())
        seen_hashes.update(candidate.cell_hash for candidate in population)
        seen_hashes.update(candidate.cell_hash for candidate in archive.items)
        searcher.load_state_dict(extra.get("cem_state") or {})
        logging.info(
            "resuming CEM Stage 2 after generation %d with %d seen cells",
            start_generation,
            len(seen_hashes),
        )

    for generation in range(
        int(start_generation) + 1,
        int(runner.cfg.stage2_generations) + 1,
    ):
        offspring = searcher.propose(
            backbones,
            size=runner.cfg.offspring_size,
            round_index=generation,
            excluded_hashes=seen_hashes,
        )
        runner._evaluate_with_retries(offspring)
        if sum(candidate.evaluated for candidate in offspring) < len(offspring):
            raise RuntimeError(
                f"CEM Stage 2 generation {generation} has unresolved failures"
            )
        seen_hashes.update(candidate.cell_hash for candidate in offspring)
        searcher.update(offspring)
        population = environmental_select(
            list(population) + offspring,
            runner.cfg.population_size,
            runner.cfg.delay_limit,
        )
        archive.update(offspring)
        logging.info(
            "CEM Stage 2 generation %d/%d complete: archive=%d seen=%d",
            generation,
            runner.cfg.stage2_generations,
            len(archive.items),
            len(seen_hashes),
        )
        if generation % runner.cfg.checkpoint_every == 0:
            runner._save_generation(
                2,
                generation,
                population,
                archive,
                runner.cell_bandit,
                extra_state={
                    "search_mode": "cem",
                    "seen_cell_hashes": sorted(seen_hashes),
                    "cem_state": searcher.state_dict(),
                },
            )
    return _finalize(runner, population, archive, done, "CEM")


def run_stage2_diffam_proxy(
    runner, backbones: Sequence[Candidate]
) -> List[Candidate]:
    """One-generation-stale producer hides proxy/DiffAM work under DC."""
    done = os.path.join(runner.run_dir, "stage2", "elites_24.json")
    if os.path.exists(done):
        logging.info("Stage 2 already complete; loading %s", done)
        return _load_candidates(done)
    restored = runner._restore_generation(2)
    if restored is None:
        population = _seed_population(runner, backbones)
        runner._evaluate_with_retries(population)
        if sum(candidate.evaluated for candidate in population) < len(population):
            raise RuntimeError(
                "proxy DiffAM Stage 2 common seeds have unresolved failures"
            )
        seen_hashes = {candidate.cell_hash for candidate in population}
        population = environmental_select(
            population,
            runner.cfg.population_size,
            runner.cfg.delay_limit,
        )
        archive = runner._new_archive(2)
        archive.update(population)
        start_generation = 0
        search_state = None
        pending = None
        pending_round = None
        unconsumed_observations = list(population)
    else:
        start_generation, population, archive = restored
        extra = runner._restored_extra_state.get(2, {})
        if extra.get("search_mode") != "diffam_proxy":
            raise RuntimeError(
                "Stage 2 checkpoint was not created by proxy DiffAM"
            )
        seen_hashes = set(extra.get("seen_cell_hashes") or ())
        seen_hashes.update(candidate.cell_hash for candidate in population)
        seen_hashes.update(candidate.cell_hash for candidate in archive.items)
        search_state = extra.get("diffam_proxy_state")
        pending = [
            Candidate.from_dict(payload)
            for payload in extra.get("pending_candidates") or []
        ]
        pending = pending or None
        pending_round = extra.get("pending_round")
        unconsumed_observations = [
            Candidate.from_dict(payload)
            for payload in extra.get("unconsumed_observations") or []
        ]
        logging.info(
            "resuming proxy DiffAM Stage 2 after generation %d: "
            "seen=%d pending_round=%s",
            start_generation,
            len(seen_hashes),
            pending_round,
        )

    producer = DiffAMProxyProducer(
        runner.cfg,
        runner.run_dir,
        backbones,
        state=search_state,
    )
    try:
        if (
            pending is None
            and int(start_generation) < int(runner.cfg.stage2_generations)
        ):
            next_round = int(start_generation) + 1
            producer.request(
                size=runner.cfg.offspring_size,
                round_index=next_round,
                excluded_hashes=seen_hashes,
                warm_starts=population,
                observations=unconsumed_observations,
            )
            pending, search_state = producer.receive()
            pending_round = next_round
            unconsumed_observations = []
            runner._save_generation(
                2,
                start_generation,
                population,
                archive,
                runner.cell_bandit,
                extra_state={
                    "search_mode": "diffam_proxy",
                    "seen_cell_hashes": sorted(seen_hashes),
                    "diffam_proxy_state": search_state,
                    "pending_round": pending_round,
                    "pending_candidates": [
                        candidate.to_dict() for candidate in pending
                    ],
                    "unconsumed_observations": [],
                },
            )

        for generation in range(
            int(start_generation) + 1,
            int(runner.cfg.stage2_generations) + 1,
        ):
            if pending is None or int(pending_round) != generation:
                raise RuntimeError(
                    f"proxy DiffAM pending round mismatch: "
                    f"have={pending_round} expected={generation}"
                )
            offspring = pending
            if any(candidate.cell_hash in seen_hashes for candidate in offspring):
                raise AssertionError(
                    "proxy DiffAM returned an already evaluated cell_hash"
                )

            has_next = generation < int(runner.cfg.stage2_generations)
            if has_next:
                reserved = set(seen_hashes)
                reserved.update(candidate.cell_hash for candidate in offspring)
                producer.request(
                    size=runner.cfg.offspring_size,
                    round_index=generation + 1,
                    excluded_hashes=reserved,
                    warm_starts=population,
                    observations=unconsumed_observations,
                )
                unconsumed_observations = []

            dc_start = time.monotonic()
            runner._evaluate_with_retries(offspring)
            dc_seconds = time.monotonic() - dc_start
            if sum(candidate.evaluated for candidate in offspring) < len(offspring):
                raise RuntimeError(
                    f"proxy DiffAM Stage 2 generation {generation} "
                    "has unresolved failures"
                )
            seen_hashes.update(candidate.cell_hash for candidate in offspring)
            population = environmental_select(
                list(population) + offspring,
                runner.cfg.population_size,
                runner.cfg.delay_limit,
            )
            archive.update(offspring)

            wait_seconds = 0.0
            if has_next:
                wait_start = time.monotonic()
                pending, search_state = producer.receive()
                wait_seconds = time.monotonic() - wait_start
                pending_round = generation + 1
                unconsumed_observations = list(offspring)
            else:
                pending = None
                pending_round = None
                unconsumed_observations = list(offspring)
            logging.info(
                "proxy DiffAM Stage 2 generation %d/%d complete: "
                "archive=%d seen=%d dc=%.1fs exposed_diffam_wait=%.1fs",
                generation,
                runner.cfg.stage2_generations,
                len(archive.items),
                len(seen_hashes),
                dc_seconds,
                wait_seconds,
            )
            if generation % runner.cfg.checkpoint_every == 0:
                runner._save_generation(
                    2,
                    generation,
                    population,
                    archive,
                    runner.cell_bandit,
                    extra_state={
                        "search_mode": "diffam_proxy",
                        "seen_cell_hashes": sorted(seen_hashes),
                        "diffam_proxy_state": search_state,
                        "pending_round": pending_round,
                        "pending_candidates": [
                            candidate.to_dict() for candidate in (pending or [])
                        ],
                        "unconsumed_observations": [
                            candidate.to_dict()
                            for candidate in unconsumed_observations
                        ],
                    },
                )
    finally:
        producer.close()
    return _finalize(runner, population, archive, done, "proxy DiffAM")
