"""Stage-1 structure mutations, including CT42 relocation and local repair."""
from __future__ import annotations

import copy
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils import CompressorTree

from .candidate import Candidate


CLASSIC = "classic"
CT42_LOCAL = "ct42_local"
CT42_REPAIR = "ct42_repair"
BOUNDARY_K = "boundary_k"
STRUCTURE_ARMS = [CLASSIC, CT42_LOCAL, CT42_REPAIR, BOUNDARY_K]


class StructureMutator:
    def __init__(
        self,
        engine,
        k_min: int = 2,
        k_max: int = 24,
        repair_window_min: int = 3,
        repair_window_max: int = 6,
        repair_max_edits: int = 6,
    ):
        self.engine = engine
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.window_min = int(repair_window_min)
        self.window_max = int(repair_window_max)
        self.max_edits = int(repair_max_edits)
        self.pp = np.asarray(engine.initial_pp, dtype=int)

    def _load(self, c: Candidate) -> None:
        self.engine._activate_trunc_profile(c.k)
        self.engine.state = {
            "k": int(c.k),
            "ct22": np.asarray(c.ct22, dtype=int),
            "ct32": np.asarray(c.ct32, dtype=int),
            "ct42": np.asarray(c.ct42, dtype=int),
            "cells": [],
        }

    @staticmethod
    def _store(c: Candidate, state: dict) -> None:
        c.ct22 = np.asarray(state["ct22"], dtype=int).tolist()
        c.ct32 = np.asarray(state["ct32"], dtype=int).tolist()
        c.ct42 = np.asarray(state.get("ct42"), dtype=int).tolist()

    def _masked_columns(self, c: Candidate, action_types: Iterable[int]) -> Dict[int, List[int]]:
        self._load(c)
        try:
            mask = np.asarray(self.engine.get_action_mask(), dtype=bool)
        except (ValueError, AssertionError):
            return {}
        out: Dict[int, List[int]] = {}
        for action_type in action_types:
            cols = [
                col for col in range(len(c.ct22))
                if mask[col * 6 + int(action_type)]
            ]
            if cols:
                out[int(action_type)] = cols
        return out

    def legal_arms(self, c: Candidate) -> List[str]:
        arms = []
        if self._masked_columns(c, range(4)):
            arms.append(CLASSIC)
        local = self._masked_columns(c, (4, 5))
        can_relocate = any(x > 0 for x in c.ct42) and any(
            f > 0 and h > 0 for f, h in zip(c.ct32[:-1], c.ct22[:-1])
        )
        if local or can_relocate:
            arms.append(CT42_LOCAL)
        if self._repair_targets(c):
            arms.append(CT42_REPAIR)
        if c.k > self.k_min or c.k < self.k_max:
            arms.append(BOUNDARY_K)
        return arms

    def mutate(
        self, parent: Candidate, arm: str, rng: random.Random, steps: int = 1
    ) -> Optional[Candidate]:
        child = parent.clone(stage=1)
        child.cells = []
        changed = False
        action_names = []
        for _ in range(max(1, int(steps))):
            name = self._apply_one(child, arm, rng)
            if name is None:
                break
            changed = True
            action_names.append(name)
        if not changed or not self.validate(child):
            return None
        child.operator = arm
        child.metadata["structure_actions"] = action_names
        child.refresh_id()
        return child

    def _apply_one(self, c: Candidate, arm: str, rng: random.Random) -> Optional[str]:
        if arm == CLASSIC:
            choices = self._masked_columns(c, range(4))
            if not choices:
                return None
            action_type = rng.choice(sorted(choices))
            col = rng.choice(choices[action_type])
            self._load(c)
            state = self.engine.transition(col * 6 + action_type)
            self._store(c, state or self.engine.state)
            return ("add_HA", "remove_HA", "FA_to_HA", "HA_to_FA")[action_type]
        if arm == CT42_LOCAL:
            exact = self._masked_columns(c, (4, 5))
            ops = []
            if 4 in exact:
                ops.append("promote_CT42_exact")
            if 5 in exact:
                ops.append("demote_CT42_exact")
            if any(x > 0 for x in c.ct42) and any(
                f > 0 and h > 0 for f, h in zip(c.ct32[:-1], c.ct22[:-1])
            ):
                ops.append("relocate_CT42")
            if not ops:
                return None
            op = rng.choice(ops)
            if op == "relocate_CT42":
                srcs = [i for i, x in enumerate(c.ct42[:-1]) if x > 0]
                dsts = [
                    i for i, (f, h) in enumerate(zip(c.ct32[:-1], c.ct22[:-1]))
                    if f > 0 and h > 0
                ]
                pairs = [(s, d) for s in srcs for d in dsts if s != d]
                if not pairs:
                    return None
                src, dst = rng.choice(pairs)
                c.ct42[src] -= 1; c.ct32[src] += 1; c.ct22[src] += 1
                c.ct42[dst] += 1; c.ct32[dst] -= 1; c.ct22[dst] -= 1
                return op
            action_type = 4 if op.startswith("promote") else 5
            col = rng.choice(exact[action_type])
            self._load(c)
            state = self.engine.transition(col * 6 + action_type)
            self._store(c, state or self.engine.state)
            return op
        if arm == CT42_REPAIR:
            targets = self._repair_targets(c)
            if not targets:
                return None
            delta, col = rng.choice(targets)
            repaired = self._repair_ct42(c, col, delta)
            if repaired is None:
                return None
            c.ct32, c.ct22, c.ct42 = repaired
            return "insert_CT42_repair" if delta > 0 else "delete_CT42_repair"
        if arm == BOUNDARY_K:
            dirs = []
            if c.k > self.k_min:
                dirs.append(-1)
            if c.k < self.k_max:
                dirs.append(1)
            if not dirs:
                return None
            delta = rng.choice(dirs)
            c.k += delta
            return "increase_k" if delta > 0 else "decrease_k"
        raise ValueError(f"unknown structure arm: {arm}")

    def _repair_targets(self, c: Candidate) -> List[Tuple[int, int]]:
        targets = [(-1, col) for col, n in enumerate(c.ct42[:-1]) if n > 0]
        for col in range(len(c.ct42) - 1):
            trial = copy.deepcopy(c)
            if self._repair_ct42(trial, col, 1) is not None:
                targets.append((1, col))
        return targets

    @staticmethod
    def _carry(f: int, h: int, t: int) -> int:
        return int(f) + int(h) + 2 * int(t)

    def _repair_ct42(
        self, c: Candidate, target_col: int, delta: int
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        n = len(c.ct42)
        if target_col < 0 or target_col >= n - 1:
            return None
        t = list(c.ct42)
        t[target_col] += int(delta)
        if t[target_col] < 0:
            return None
        old_f, old_h = list(c.ct32), list(c.ct22)
        for width in range(self.window_min, self.window_max + 1):
            end = min(n - 1, target_col + width - 1)
            prev = 0 if target_col == 0 else self._carry(
                old_f[target_col - 1], old_h[target_col - 1], t[target_col - 1]
            )
            states = {prev: (0.0, 0, [])}
            for col in range(target_col, end + 1):
                nxt = {}
                for carry_in, (cost, edits, path) in states.items():
                    height = int(self.pp[col]) + int(carry_in)
                    for remain in (1, 2):
                        max_f = max(0, (height - 3 * t[col] - remain) // 2)
                        for f in range(max_f + 1):
                            h = height - 3 * t[col] - 2 * f - remain
                            if h < 0:
                                continue
                            carry_out = self._carry(f, h, t[col])
                            delta_edits = abs(f - old_f[col]) + abs(h - old_h[col])
                            new_edits = edits + delta_edits
                            if new_edits > self.max_edits:
                                continue
                            changed = int(f != old_f[col] or h != old_h[col])
                            new_cost = cost + delta_edits + 0.25 * changed
                            old = nxt.get(carry_out)
                            if old is None or (new_cost, new_edits) < (old[0], old[1]):
                                nxt[carry_out] = (new_cost, new_edits, path + [(f, h)])
                states = nxt
                if not states:
                    break
            boundary = self._carry(old_f[end], old_h[end], c.ct42[end])
            best = states.get(boundary)
            if best is None:
                continue
            f_new, h_new = list(old_f), list(old_h)
            for offset, (f, h) in enumerate(best[2]):
                f_new[target_col + offset] = int(f)
                h_new[target_col + offset] = int(h)
            trial = Candidate(c.k, h_new, f_new, t)
            if self.validate(trial):
                return f_new, h_new, t
        return None

    def validate(self, c: Candidate) -> bool:
        arrays = (c.ct22, c.ct32, c.ct42)
        if any(len(a) != len(self.pp) for a in arrays):
            return False
        if any(x < 0 for a in arrays for x in a):
            return False
        if c.ct42[-1] != 0 or not (self.k_min <= c.k <= self.k_max):
            return False
        carry = 0
        for col in range(len(self.pp)):
            remain = (
                int(self.pp[col]) + carry - 2 * c.ct32[col]
                - c.ct22[col] - 3 * c.ct42[col]
            )
            if remain not in (1, 2):
                return False
            carry = self._carry(c.ct32[col], c.ct22[col], c.ct42[col])
        try:
            CompressorTree(
                self.pp,
                np.asarray(c.ct32),
                np.asarray(c.ct22),
                np.asarray(c.ct42),
            ).compressor_assignment_fused()
        except (AssertionError, ValueError, IndexError):
            return False
        return True
