"""Scheme-A crossover and Stage-2 cell mutation operators."""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import CompressorTree

from .candidate import Candidate


CELL_ARMS = ["cell_add", "cell_remove", "cell_swap", "cell_resample", "zero_toggle"]


class CellOperator:
    def __init__(self, engine):
        self.engine = engine
        self.pp = np.asarray(engine.initial_pp, dtype=int)
        self._slot_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}

    def slots(self, c: Candidate) -> List[Tuple[int, int, int, int]]:
        cached = self._slot_cache.get(c.structure_hash)
        if cached is not None:
            return list(cached)
        self.engine._activate_trunc_profile(c.k)
        assignment = CompressorTree(
            self.pp,
            np.asarray(c.ct32),
            np.asarray(c.ct22),
            np.asarray(c.ct42),
        ).compressor_assignment_fused()
        slots = sorted(self.engine._enumerate_type_slots(assignment))
        self._slot_cache[c.structure_hash] = slots
        return list(slots)

    def _table(self, t: int):
        _head, table = self.engine._type_head_and_table(int(t))
        return table

    def _zero_types(self, t: int) -> List[int]:
        out = []
        for idx, entry in enumerate(self._table(t)):
            if idx and (
                entry.get("const_zero")
                or str(entry.get("name", "")).endswith("_zero")
            ):
                out.append(idx)
        return out

    def _nonzero_approx_types(self, t: int) -> List[int]:
        zeros = set(self._zero_types(t))
        return [i for i in range(1, len(self._table(t))) if i not in zeros]

    @staticmethod
    def _map(cells: Sequence[Sequence[int]]) -> Dict[Tuple[int, int, int, int], List[int]]:
        return {tuple(int(x) for x in e[:4]): [int(x) for x in e] for e in cells}

    def legal_arms(self, c: Candidate) -> List[str]:
        slots = self.slots(c)
        occupied = self._map(c.cells)
        free = [s for s in slots if s not in occupied]
        arms = []
        if any(self._nonzero_approx_types(s[2]) for s in free):
            arms.append("cell_add")
        if c.cells:
            arms.extend(["cell_remove", "cell_swap"])
        if slots:
            arms.append("cell_resample")
        if any(self._zero_types(s[2]) for s in free) or any(
            int(e[4]) in self._zero_types(int(e[2])) for e in c.cells
        ):
            arms.append("zero_toggle")
        return arms

    def make_seed_variants(self, backbone: Candidate, rng: random.Random) -> List[Candidate]:
        targets = [0, 1, 4, 12]
        out = []
        for idx, target in enumerate(targets):
            c = backbone.clone(stage=2)
            c.cells = []
            c.operator = ("seed_exact", "seed_low", "seed_medium", "seed_stratified")[idx]
            for _ in range(min(target, len(self.slots(c)))):
                if not self._add(c, rng, prefer_low=(idx == 3)):
                    break
            c.refresh_id()
            out.append(c)
        return out

    def crossover_a(
        self, parent_a: Candidate, parent_b: Candidate, rng: random.Random
    ) -> Candidate:
        host, donor = (parent_a, parent_b) if rng.random() < 0.5 else (parent_b, parent_a)
        child = host.clone(stage=2)
        host_map = self._map(host.cells)
        donor_map = {}
        shift = int(host.k) - int(donor.k)
        for e in donor.cells:
            moved = [int(x) for x in e]
            moved[1] += shift
            donor_map[tuple(moved[:4])] = moved
        slotset = set(self.slots(child))
        donor_map = {k: v for k, v in donor_map.items() if k in slotset}

        rel_cols = sorted({slot[1] - child.k for slot in slotset})
        donor_rel = set()
        if rel_cols:
            for _ in range(rng.randint(1, min(3, len(rel_cols)))):
                a = rng.randrange(len(rel_cols))
                b = rng.randrange(a, len(rel_cols))
                donor_rel.update(rel_cols[a : b + 1])
        cells = []
        for slot in sorted(slotset):
            source = donor_map if slot[1] - child.k in donor_rel else host_map
            allele = source.get(slot)
            if allele is not None and int(allele[4]) < len(self._table(slot[2])):
                cells.append(list(allele))
        child.cells = sorted(cells, key=lambda e: tuple(e))
        child.parent_ids = [parent_a.candidate_id, parent_b.candidate_id]
        child.metadata["crossover"] = "scheme_a"
        child.metadata["host_parent"] = host.candidate_id
        child.refresh_id()
        return child

    def mutate(self, base: Candidate, arm: str, rng: random.Random) -> Optional[Candidate]:
        child = copy.deepcopy(base)
        before = child.cell_hash
        ok = {
            "cell_add": lambda: self._add(child, rng),
            "cell_remove": lambda: self._remove(child, rng),
            "cell_swap": lambda: self._swap(child, rng),
            "cell_resample": lambda: self._resample(child, rng),
            "zero_toggle": lambda: self._zero_toggle(child, rng),
        }[arm]()
        if not ok:
            return None
        child.cells = sorted(child.cells, key=lambda e: tuple(e))
        child.operator = arm
        child.stage = 2
        child.routing = None
        child.refresh_id()
        return None if child.cell_hash == before else child

    def _add(self, c: Candidate, rng: random.Random, prefer_low: bool = False) -> bool:
        occupied = self._map(c.cells)
        free = [s for s in self.slots(c) if s not in occupied and self._nonzero_approx_types(s[2])]
        if not free:
            return False
        if prefer_low:
            lo = min(s[1] for s in free)
            near = [s for s in free if s[1] <= lo + 2]
            slot = rng.choice(near)
        else:
            slot = rng.choice(free)
        type_idx = rng.choice(self._nonzero_approx_types(slot[2]))
        c.cells.append([*slot, int(type_idx)])
        return True

    def _remove(self, c: Candidate, rng: random.Random) -> bool:
        if not c.cells:
            return False
        del c.cells[rng.randrange(len(c.cells))]
        return True

    def _swap(self, c: Candidate, rng: random.Random) -> bool:
        if not c.cells:
            return self._add(c, rng)
        idx = rng.randrange(len(c.cells))
        old = list(c.cells[idx])
        alternatives = [x for x in self._nonzero_approx_types(old[2]) if x != old[4]]
        occupied = self._map(c.cells)
        free_same_type = [
            s for s in self.slots(c)
            if s not in occupied and int(s[2]) == int(old[2])
        ]
        modes = []
        if alternatives:
            modes.append("type")
        if free_same_type:
            modes.append("move")
        if not modes:
            return False
        if rng.choice(modes) == "type":
            c.cells[idx][4] = rng.choice(alternatives)
        else:
            slot = rng.choice(free_same_type)
            c.cells[idx] = [*slot, old[4]]
        return True

    def _resample(self, c: Candidate, rng: random.Random) -> bool:
        slots = self.slots(c)
        if not slots:
            return False
        cols = sorted({s[1] for s in slots})
        width = rng.randint(2, min(5, len(cols))) if len(cols) >= 2 else 1
        start = rng.randrange(0, len(cols) - width + 1)
        chosen = set(cols[start : start + width])
        old_window = [e for e in c.cells if e[1] in chosen]
        c.cells = [e for e in c.cells if e[1] not in chosen]
        target = len(old_window)
        if target == 0:
            target = rng.randint(1, max(1, min(4, sum(s[1] in chosen for s in slots))))
        for _ in range(target):
            occupied = self._map(c.cells)
            free = [
                s for s in slots
                if s[1] in chosen and s not in occupied and self._nonzero_approx_types(s[2])
            ]
            if not free:
                break
            slot = rng.choice(free)
            c.cells.append([*slot, rng.choice(self._nonzero_approx_types(slot[2]))])
        return self._map(old_window) != self._map([e for e in c.cells if e[1] in chosen])

    def _zero_toggle(self, c: Candidate, rng: random.Random) -> bool:
        zero_existing = [
            i for i, e in enumerate(c.cells)
            if int(e[4]) in self._zero_types(int(e[2]))
        ]
        occupied = self._map(c.cells)
        zero_free = [
            s for s in self.slots(c)
            if s not in occupied and self._zero_types(s[2])
        ]
        modes = []
        if zero_existing:
            modes.append("off")
        if zero_free:
            modes.append("on")
        if not modes:
            return False
        if rng.choice(modes) == "off":
            del c.cells[rng.choice(zero_existing)]
        else:
            slot = rng.choice(zero_free)
            c.cells.append([*slot, rng.choice(self._zero_types(slot[2]))])
        return True
