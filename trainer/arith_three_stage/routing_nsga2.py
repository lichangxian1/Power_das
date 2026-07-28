"""Constraint-preserving NSGA-II variation for fixed-structure routing.

Each ``(stage, column)`` slice is a complete legal bipartite matching. Genetic
operators exchange whole slices and use a fresh Gumbel-Hungarian route as the
mutation donor, so every child remains legal without a repair heuristic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Dict, List, Sequence, Tuple

from .candidate import Candidate
from .cem import RoutingCEM, RoutingLogits
from .pareto import assign_crowding, fast_non_dominated_sort, tournament


SliceKey = Tuple[int, int]


def route_digest(connection: Sequence[tuple]) -> str:
    signature = RoutingCEM._route_signature(connection)
    raw = json.dumps(signature, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def route_blocks(connection: Sequence[tuple]) -> Dict[SliceKey, List[tuple]]:
    blocks: Dict[SliceKey, List[tuple]] = {}
    for edge in connection:
        if len(edge) != 4:
            raise ValueError("routing edge must be (src, dst, port, metadata)")
        metadata = edge[3] or {}
        key = tuple(int(x) for x in metadata.get("slice", ()))
        if len(key) != 2:
            raise ValueError("routing edge is missing a two-dimensional slice key")
        blocks.setdefault(key, []).append(edge)
    for key in blocks:
        blocks[key] = sorted(
            blocks[key], key=lambda edge: int((edge[3] or {})["flat_row"])
        )
    return blocks


def compose_route(blocks: Dict[SliceKey, Sequence[tuple]]) -> List[tuple]:
    return [
        copy.deepcopy(edge)
        for key in sorted(blocks)
        for edge in blocks[key]
    ]


class RoutingNSGA2:
    """Localized block crossover/mutation with standard NSGA-II tournaments."""

    def __init__(
        self,
        engine,
        rng: random.Random,
        *,
        crossover_probability: float = 0.70,
        crossover_blocks: int = 8,
        mutation_blocks: int = 4,
        immigrants: int = 6,
        donor_temperature: float = 1.25,
    ):
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")
        if crossover_blocks < 1 or mutation_blocks < 1:
            raise ValueError("crossover_blocks and mutation_blocks must be positive")
        if immigrants < 0:
            raise ValueError("immigrants must be non-negative")
        self.engine = engine
        self.rng = rng
        self.crossover_probability = float(crossover_probability)
        self.crossover_blocks = int(crossover_blocks)
        self.mutation_blocks = int(mutation_blocks)
        self.immigrants = int(immigrants)
        self.sampler = RoutingCEM(
            engine,
            smoothing=1.0,
            exploration=0.0,
            temperature=float(donor_temperature),
            init_mode="policy",
        )

    def initialize_logits(self, template: RoutingLogits) -> RoutingLogits:
        return self.sampler.initialize(template)

    @staticmethod
    def prepare_population(population: Sequence[Candidate], delay_limit: float) -> None:
        for front in fast_non_dominated_sort(population, delay_limit):
            assign_crowding(front)

    def _fresh_route(self, logits: RoutingLogits) -> List[tuple]:
        connection, _score = self.sampler.sample(logits)
        return connection

    @staticmethod
    def _accept(
        connection: Sequence[tuple],
        expected_keys: Sequence[SliceKey],
        seen: set,
    ) -> Tuple[bool, str]:
        blocks = route_blocks(connection)
        if sorted(blocks) != list(expected_keys):
            raise ValueError("offspring does not contain the complete slice set")
        digest = route_digest(connection)
        if digest in seen:
            return False, digest
        seen.add(digest)
        return True, digest

    def initial_population(
        self,
        logits: RoutingLogits,
        count: int,
        seen: set,
    ) -> List[dict]:
        expected_keys = sorted(logits)
        out: List[dict] = []
        attempts = 0
        while len(out) < count and attempts < max(1000, count * 200):
            attempts += 1
            connection = self._fresh_route(logits)
            accepted, digest = self._accept(connection, expected_keys, seen)
            if not accepted:
                continue
            out.append(
                {
                    "connection": connection,
                    "digest": digest,
                    "operator": "nsga2_initial",
                    "parent_ids": [],
                    "crossover_blocks": 0,
                    "mutation_blocks": len(expected_keys),
                }
            )
        if len(out) != count:
            raise RuntimeError(f"NSGA-II initialized only {len(out)}/{count} unique routes")
        return out

    def make_offspring(
        self,
        population: Sequence[Candidate],
        logits: RoutingLogits,
        count: int,
        seen: set,
        delay_limit: float,
    ) -> List[dict]:
        if len(population) < 2:
            raise ValueError("NSGA-II requires at least two evaluated parents")
        self.prepare_population(population, delay_limit)
        expected_keys = sorted(logits)
        immigrant_count = min(int(count), self.immigrants)
        genetic_target = int(count) - immigrant_count
        out: List[dict] = []
        attempts = 0
        max_attempts = max(2000, count * 400)

        while len(out) < genetic_target and attempts < max_attempts:
            attempts += 1
            parent_a = tournament(population, self.rng)
            blocks = route_blocks(parent_a.routing or [])
            parent_ids = [parent_a.candidate_id]
            crossover_count = 0
            if self.rng.random() < self.crossover_probability:
                parent_b = tournament(population, self.rng)
                blocks_b = route_blocks(parent_b.routing or [])
                crossover_count = self.rng.randint(
                    1, min(self.crossover_blocks, len(expected_keys))
                )
                for key in self.rng.sample(expected_keys, crossover_count):
                    blocks[key] = blocks_b[key]
                parent_ids.append(parent_b.candidate_id)

            donor = route_blocks(self._fresh_route(logits))
            mutation_count = self.rng.randint(
                1, min(self.mutation_blocks, len(expected_keys))
            )
            for key in self.rng.sample(expected_keys, mutation_count):
                blocks[key] = donor[key]
            connection = compose_route(blocks)
            accepted, digest = self._accept(connection, expected_keys, seen)
            if not accepted:
                continue
            out.append(
                {
                    "connection": connection,
                    "digest": digest,
                    "operator": "nsga2_genetic",
                    "parent_ids": parent_ids,
                    "crossover_blocks": crossover_count,
                    "mutation_blocks": mutation_count,
                }
            )

        while len(out) < count and attempts < max_attempts:
            attempts += 1
            connection = self._fresh_route(logits)
            accepted, digest = self._accept(connection, expected_keys, seen)
            if not accepted:
                continue
            out.append(
                {
                    "connection": connection,
                    "digest": digest,
                    "operator": "nsga2_immigrant",
                    "parent_ids": [],
                    "crossover_blocks": 0,
                    "mutation_blocks": len(expected_keys),
                }
            )

        if len(out) != count:
            raise RuntimeError(
                f"NSGA-II produced only {len(out)}/{count} globally unique routes"
            )
        return out

