"""Constrained NSGA-II primitives and a separate external archive."""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .candidate import Candidate


def constraint_violation(c: Candidate, delay_limit: float) -> float:
    if not c.valid or not c.evaluated:
        return math.inf
    return max(0.0, float(c.delay) / float(delay_limit) - 1.0)


def dominates(a: Candidate, b: Candidate, delay_limit: float) -> bool:
    va, vb = constraint_violation(a, delay_limit), constraint_violation(b, delay_limit)
    if va == 0.0 and vb > 0.0:
        return True
    if va > 0.0 and vb == 0.0:
        return False
    if va > 0.0 or vb > 0.0:
        return va < vb
    av = (float(a.area), float(a.power), float(a.mred))
    bv = (float(b.area), float(b.power), float(b.mred))
    return all(x <= y for x, y in zip(av, bv)) and any(
        x < y for x, y in zip(av, bv)
    )


def _deduplicate(items: Iterable[Candidate]) -> List[Candidate]:
    out = {}
    for c in items:
        key = c.routing_hash if c.routing is not None else c.cell_hash
        old = out.get(key)
        if old is None or (c.valid and not old.valid):
            out[key] = c
    return list(out.values())


def _objective_key(c: Candidate) -> Tuple[float, float, float]:
    """Exact measured objectives; equal keys are Pareto-equivalent."""
    return (float(c.area), float(c.power), float(c.mred))


def _candidate_preference(c: Candidate) -> Tuple[float, int, int, str]:
    """Deterministic representative preference inside an equivalent group."""
    genotype_hash = c.routing_hash if c.routing is not None else c.cell_hash
    return (
        float(c.delay),
        len(c.cells),
        len(c.routing or []),
        genotype_hash,
    )


def _routing_edges(c: Candidate):
    edges = set()
    for edge in c.routing or []:
        if len(edge) < 3:
            continue
        src, dst, port = edge[:3]
        meta = edge[3] if len(edge) > 3 else None
        edges.add((
            int(src),
            int(dst),
            int(port),
            str((meta or {}).get("src_output", "sum")),
        ))
    return edges


def _genotype_distance(a: Candidate, b: Candidate) -> int:
    """Distance used only to retain diverse Pareto-equivalent variants."""
    distance = abs(int(a.k) - int(b.k))
    for name in ("ct22", "ct32", "ct42"):
        aa, bb = getattr(a, name), getattr(b, name)
        distance += abs(len(aa) - len(bb))
        distance += sum(x != y for x, y in zip(aa, bb))
    cells_a = {tuple(x) for x in a.cells}
    cells_b = {tuple(x) for x in b.cells}
    distance += len(cells_a.symmetric_difference(cells_b))
    distance += len(_routing_edges(a).symmetric_difference(_routing_edges(b)))
    return distance


def _limit_objective_variants(
    items: Iterable[Candidate], limit: int
) -> List[Candidate]:
    """Keep at most ``limit`` diverse genotypes per exact objective tuple."""
    if int(limit) < 1:
        raise ValueError("objective variant limit must be at least 1")
    groups: Dict[Tuple[float, float, float], List[Candidate]] = {}
    for candidate in items:
        groups.setdefault(_objective_key(candidate), []).append(candidate)

    kept: List[Candidate] = []
    for key in sorted(groups):
        remaining = sorted(groups[key], key=_candidate_preference)
        if len(remaining) <= limit:
            kept.extend(remaining)
            continue
        selected = [remaining.pop(0)]
        while remaining and len(selected) < limit:
            chosen = min(
                remaining,
                key=lambda c: (
                    -min(_genotype_distance(c, old) for old in selected),
                    _candidate_preference(c),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        kept.extend(selected)
    return kept


def fast_non_dominated_sort(
    population: Sequence[Candidate], delay_limit: float
) -> List[List[Candidate]]:
    pop = list(population)
    dominated_sets = [[] for _ in pop]
    domination_counts = [0 for _ in pop]
    fronts: List[List[int]] = [[]]
    for i, a in enumerate(pop):
        for j in range(i + 1, len(pop)):
            b = pop[j]
            if dominates(a, b, delay_limit):
                dominated_sets[i].append(j)
                domination_counts[j] += 1
            elif dominates(b, a, delay_limit):
                dominated_sets[j].append(i)
                domination_counts[i] += 1
    for i, n in enumerate(domination_counts):
        if n == 0:
            pop[i].rank = 0
            fronts[0].append(i)
    k = 0
    while k < len(fronts) and fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        k += 1
    return [[pop[i] for i in f] for f in fronts if f]


def assign_crowding(front: Sequence[Candidate]) -> None:
    for c in front:
        c.crowding_distance = 0.0
    if len(front) <= 2:
        for c in front:
            c.crowding_distance = math.inf
        return
    objectives = (
        lambda c: float(c.area) if c.area is not None else math.inf,
        lambda c: float(c.power) if c.power is not None else math.inf,
        lambda c: math.log10(max(float(c.mred), 1e-15))
        if c.mred is not None else math.inf,
    )
    for value in objectives:
        ordered = sorted(front, key=value)
        ordered[0].crowding_distance = ordered[-1].crowding_distance = math.inf
        lo, hi = value(ordered[0]), value(ordered[-1])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            continue
        for i in range(1, len(ordered) - 1):
            if math.isfinite(ordered[i].crowding_distance):
                ordered[i].crowding_distance += (
                    value(ordered[i + 1]) - value(ordered[i - 1])
                ) / (hi - lo)


def environmental_select(
    candidates: Iterable[Candidate], size: int, delay_limit: float
) -> List[Candidate]:
    pool = _deduplicate(candidates)
    selected: List[Candidate] = []
    for front in fast_non_dominated_sort(pool, delay_limit):
        assign_crowding(front)
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            front = sorted(front, key=lambda c: c.crowding_distance, reverse=True)
            selected.extend(front[: size - len(selected)])
            break
    return selected


def tournament(population: Sequence[Candidate], rng: random.Random) -> Candidate:
    a, b = rng.sample(list(population), 2)
    ka = (a.rank if a.rank is not None else math.inf, -a.crowding_distance)
    kb = (b.rank if b.rank is not None else math.inf, -b.crowding_distance)
    return a if ka < kb else b


class ExternalArchive:
    def __init__(
        self,
        delay_limit: float,
        variants_per_objective: Optional[int] = None,
    ):
        self.delay_limit = float(delay_limit)
        if variants_per_objective is not None and int(variants_per_objective) < 1:
            raise ValueError("variants_per_objective must be at least 1")
        self.variants_per_objective = (
            None if variants_per_objective is None else int(variants_per_objective)
        )
        self.items: List[Candidate] = []

    def update(self, candidates: Iterable[Candidate]) -> None:
        feasible = [
            c for c in list(self.items) + list(candidates)
            if constraint_violation(c, self.delay_limit) == 0.0
        ]
        if not feasible:
            return
        deduplicated = _deduplicate(feasible)
        if self.variants_per_objective is not None:
            deduplicated = _limit_objective_variants(
                deduplicated, self.variants_per_objective
            )
        fronts = fast_non_dominated_sort(deduplicated, self.delay_limit)
        self.items = fronts[0] if fronts else []
        assign_crowding(self.items)

    def to_list(self):
        return [c.to_dict() for c in self.items]
