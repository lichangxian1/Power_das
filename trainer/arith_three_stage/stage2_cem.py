"""Persistent categorical CEM for Stage-2 cell-only search."""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .candidate import Candidate
from .cell_ops import CellOperator
from .pareto import environmental_select


Slot = Tuple[int, int, int, int]


def _balanced_quota(remaining: int, total: int, index: int) -> int:
    left = int(total) - int(index)
    if remaining < 1 or left < 1:
        raise ValueError("invalid balanced CEM quota")
    return max(1, math.ceil(int(remaining) / left))


class CellCEMStage2Search:
    """Factorized categorical CEM updated only from real DC/Verilator labels."""

    def __init__(self, engine, config):
        self.engine = engine
        self.cfg = config
        self.cell_operator = CellOperator(engine)
        self.logits: Dict[str, List[np.ndarray]] = {}
        self.history: Dict[str, List[dict]] = {}
        self.rng = np.random.default_rng(int(config.seed) + 91_337)

    def _prior(self, backbone: Candidate) -> List[np.ndarray]:
        slots = self.cell_operator.slots(backbone)
        target = max(0.0, float(self.cfg.stage2_cem_init_approx_cells))
        approx_probability = min(0.75, target / max(len(slots), 1))
        out = []
        for slot in slots:
            choices = len(self.cell_operator._table(slot[2]))
            if choices <= 1:
                out.append(np.zeros(1, dtype=np.float64))
                continue
            probabilities = np.full(
                choices,
                approx_probability / (choices - 1),
                dtype=np.float64,
            )
            probabilities[0] = 1.0 - approx_probability
            out.append(np.log(np.maximum(probabilities, 1e-12)))
        return out

    def _ensure(self, backbone: Candidate) -> List[np.ndarray]:
        key = backbone.structure_hash
        if key not in self.logits:
            self.logits[key] = self._prior(backbone)
        return self.logits[key]

    @staticmethod
    def _choice_map(candidate: Candidate) -> Dict[Slot, int]:
        return {
            tuple(int(value) for value in entry[:4]): int(entry[4])
            for entry in candidate.cells
        }

    def _candidate(
        self,
        backbone: Candidate,
        choices: Sequence[int],
        *,
        round_index: int,
        sample_id: int,
    ) -> Candidate:
        slots = self.cell_operator.slots(backbone)
        child = backbone.clone(stage=2)
        child.cells = [
            [*slot, int(choice)]
            for slot, choice in zip(slots, choices)
            if int(choice) != 0
        ]
        child.cells.sort(key=lambda entry: tuple(entry))
        child.routing = None
        child.operator = "cell_cem"
        child.metadata.update(
            {
                "method": "cem",
                "generation": int(round_index),
                "cem_round": int(round_index),
                "cem_sample_id": int(sample_id),
            }
        )
        child.refresh_id()
        if child.structure_hash != backbone.structure_hash:
            raise AssertionError("cell CEM changed the fixed Stage-1 structure")
        return child

    def _sample_choices(
        self, backbone: Candidate, temperature: float
    ) -> List[int]:
        logits = self._ensure(backbone)
        prior = self._prior(backbone)
        exploration = float(self.cfg.stage2_cem_exploration)
        choices = []
        for current, initial in zip(logits, prior):
            scaled = current / max(float(temperature), 1e-6)
            probabilities = np.exp(scaled - np.max(scaled))
            probabilities /= probabilities.sum()
            initial_probabilities = np.exp(initial - np.max(initial))
            initial_probabilities /= initial_probabilities.sum()
            probabilities = (
                (1.0 - exploration) * probabilities
                + exploration * initial_probabilities
            )
            choices.append(int(self.rng.choice(len(probabilities), p=probabilities)))
        return choices

    def propose(
        self,
        backbones: Sequence[Candidate],
        *,
        size: int,
        round_index: int,
        excluded_hashes: Iterable[str] = (),
    ) -> List[Candidate]:
        unique = {}
        for backbone in backbones:
            unique.setdefault(backbone.structure_hash, backbone)
        backbone_list = list(unique.values())
        if not backbone_list:
            raise ValueError("cell CEM requires at least one fixed backbone")

        selected: List[Candidate] = []
        seen = set(excluded_hashes)
        for backbone_index, backbone in enumerate(backbone_list):
            if len(selected) >= size:
                break
            remaining = int(size) - len(selected)
            quota = _balanced_quota(remaining, len(backbone_list), backbone_index)
            made = 0
            attempts = 0
            while made < min(quota, remaining) and attempts < 20_000:
                attempts += 1
                # Reheating only resolves duplicate pressure; it does not alter
                # the fixed per-backbone candidate quota.
                temperature = float(self.cfg.stage2_cem_temperature) * (
                    1.0 + 0.25 * (attempts // 512)
                )
                candidate = self._candidate(
                    backbone,
                    self._sample_choices(backbone, temperature),
                    round_index=round_index,
                    sample_id=attempts - 1,
                )
                if candidate.cell_hash in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.cell_hash)
                made += 1
        if len(selected) != int(size):
            raise RuntimeError(
                f"cell CEM generated only {len(selected)}/{int(size)} unseen candidates"
            )
        return selected

    def update(self, evaluated: Sequence[Candidate]) -> None:
        """Update each structure from a bounded real-evaluated replay window."""
        cap = max(8, int(self.cfg.stage2_cem_history_per_structure))
        by_structure: Dict[str, List[Candidate]] = {}
        for candidate in evaluated:
            if candidate.evaluated and candidate.valid:
                by_structure.setdefault(candidate.structure_hash, []).append(candidate)

        for structure_hash, new_items in by_structure.items():
            old = [
                Candidate.from_dict(payload)
                for payload in self.history.get(structure_hash, [])
            ]
            merged = {}
            for candidate in old + list(new_items):
                merged[candidate.cell_hash] = candidate
            ordered = sorted(
                merged.values(),
                key=lambda candidate: (
                    int(candidate.metadata.get("generation", 0)),
                    candidate.cell_hash,
                ),
            )
            ordered = ordered[-cap:]
            self.history[structure_hash] = [
                candidate.to_dict() for candidate in ordered
            ]

            elite_count = max(
                2,
                int(math.ceil(
                    len(ordered) * float(self.cfg.stage2_cem_elite_fraction)
                )),
            )
            elites = environmental_select(
                ordered,
                min(elite_count, len(ordered)),
                float(self.cfg.delay_limit),
            )
            if not elites:
                continue
            backbone = new_items[0]
            slots = self.cell_operator.slots(backbone)
            current = self._ensure(backbone)
            prior = self._prior(backbone)
            maps = [self._choice_map(candidate) for candidate in elites]
            smoothing = float(self.cfg.stage2_cem_smoothing)
            exploration = float(self.cfg.stage2_cem_exploration)
            updated = []
            for row, slot in enumerate(slots):
                count = len(current[row])
                frequency = np.full(count, 0.25, dtype=np.float64)
                for mapping in maps:
                    choice = int(mapping.get(slot, 0))
                    if 0 <= choice < count:
                        frequency[choice] += 1.0
                frequency /= frequency.sum()
                old_probability = np.exp(current[row] - np.max(current[row]))
                old_probability /= old_probability.sum()
                prior_probability = np.exp(prior[row] - np.max(prior[row]))
                prior_probability /= prior_probability.sum()
                probability = (
                    (1.0 - smoothing) * old_probability
                    + smoothing * frequency
                )
                probability = (
                    (1.0 - exploration) * probability
                    + exploration * prior_probability
                )
                updated.append(np.log(np.maximum(probability, 1e-12)))
            self.logits[structure_hash] = updated

    def state_dict(self) -> dict:
        return {
            "logits": {
                key: [row.tolist() for row in rows]
                for key, rows in self.logits.items()
            },
            "history": copy.deepcopy(self.history),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        self.logits = {
            str(key): [np.asarray(row, dtype=np.float64) for row in rows]
            for key, rows in (state.get("logits") or {}).items()
        }
        self.history = copy.deepcopy(state.get("history") or {})
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
