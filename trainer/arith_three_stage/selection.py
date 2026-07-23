"""Band-balanced handoff selection between the three search stages."""
from __future__ import annotations

import copy
import math
from typing import Iterable, List, Sequence

from .candidate import Candidate
from .pareto import assign_crowding, constraint_violation


def _norm(value, lo, hi):
    return 0.0 if hi <= lo else (value - lo) / (hi - lo)


def _knee(points: Sequence[Candidate]) -> Candidate:
    area = [float(c.area) for c in points]
    power = [float(c.power) for c in points]
    mred = [math.log10(max(float(c.mred), 1e-15)) for c in points]
    return min(
        points,
        key=lambda c: (
            _norm(float(c.area), min(area), max(area)) ** 2
            + _norm(float(c.power), min(power), max(power)) ** 2
            + _norm(math.log10(max(float(c.mred), 1e-15)), min(mred), max(mred)) ** 2
        ),
    )


def _structure_distance(a: Candidate, b: Candidate) -> int:
    return abs(a.k - b.k) + sum(
        x != y
        for aa, bb in ((a.ct22, b.ct22), (a.ct32, b.ct32), (a.ct42, b.ct42))
        for x, y in zip(aa, bb)
    )


def select_banded(
    candidates: Iterable[Candidate],
    *,
    n_bins: int,
    roles: Sequence[str],
    mred_lo: float,
    mred_hi: float,
    delay_limit: float,
) -> List[Candidate]:
    pool = [
        c for c in candidates
        if constraint_violation(c, delay_limit) == 0.0
        and mred_lo <= float(c.mred) <= mred_hi
    ]
    if not pool:
        raise RuntimeError("no feasible evaluated candidates for stage handoff")
    logs = [
        math.log10(mred_lo) + i * (math.log10(mred_hi) - math.log10(mred_lo)) / n_bins
        for i in range(n_bins + 1)
    ]
    selected: List[Candidate] = []
    selected_hashes = set()
    for band in range(n_bins):
        points = [
            c for c in pool
            if logs[band] <= math.log10(max(float(c.mred), 1e-15))
            < (logs[band + 1] if band + 1 < n_bins else logs[band + 1] + 1e-12)
        ]
        if not points:
            continue
        for role in roles:
            available = [c for c in points if c.cell_hash not in selected_hashes]
            if not available:
                continue
            if role == "area":
                chosen = min(available, key=lambda c: float(c.area))
            elif role == "power":
                chosen = min(available, key=lambda c: float(c.power))
            elif role == "knee":
                chosen = _knee(available)
            elif role == "novel":
                chosen = max(
                    available,
                    key=lambda c: min(
                        (_structure_distance(c, s) for s in selected), default=10**9
                    ),
                )
            else:
                raise ValueError(f"unknown handoff role: {role}")
            item = copy.deepcopy(chosen)
            item.metadata["selection_role"] = role
            item.metadata["mred_band"] = band
            item.metadata["mred_band_edges"] = [10 ** logs[band], 10 ** logs[band + 1]]
            selected.append(item)
            selected_hashes.add(chosen.cell_hash)

    target = n_bins * len(roles)
    assign_crowding(pool)
    for c in sorted(pool, key=lambda x: x.crowding_distance, reverse=True):
        if len(selected) >= target:
            break
        if c.cell_hash in selected_hashes:
            continue
        item = copy.deepcopy(c)
        item.metadata["selection_role"] = roles[len(selected) % len(roles)]
        item.metadata["mred_band"] = -1
        item.metadata["mred_band_edges"] = [mred_lo, mred_hi]
        selected.append(item)
        selected_hashes.add(c.cell_hash)
    if len(selected) < target:
        raise RuntimeError(f"handoff needs {target} unique candidates, only found {len(selected)}")
    return selected[:target]
