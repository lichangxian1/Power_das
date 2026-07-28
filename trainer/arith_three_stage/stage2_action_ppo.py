"""Direct categorical action-PPO for Stage-2 cell-only optimization."""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .candidate import Candidate
from .cell_ops import CellOperator


_ROLES = ("area", "power", "knee")
_ROLE_WEIGHTS = {
    "area": (1.0, 0.10),
    "power": (0.10, 1.0),
    "knee": (0.50, 0.50),
}


@dataclass(frozen=True)
class CellPPOConfig:
    epochs: int = 4
    learning_rate: float = 3e-3
    clip_range: float = 0.2
    grad_clip: float = 0.5
    exploration: float = 0.05
    temperature: float = 1.0
    init_approx_cells: float = 4.0
    delay_weight: float = 5.0
    mred_weight: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("PPO epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("PPO learning rate must be positive")
        if not 0 < self.clip_range < 1:
            raise ValueError("PPO clip range must be in (0, 1)")
        if self.grad_clip <= 0:
            raise ValueError("PPO gradient clip must be positive")
        if not 0 <= self.exploration < 1:
            raise ValueError("PPO exploration must be in [0, 1)")
        if self.temperature <= 0:
            raise ValueError("PPO temperature must be positive")
        if self.delay_weight < 0 or self.mred_weight < 0:
            raise ValueError("PPO penalty weights must be non-negative")


@dataclass
class _Sample:
    key: str
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    temperature: float
    role: str
    budget: float


class CellActionPPO:
    """Factorized cell policy with the Stage-3 per-action PPO surrogate."""

    def __init__(
        self,
        engine,
        search_config,
        ppo_config: CellPPOConfig,
        backbones: Sequence[Candidate],
    ):
        ppo_config.validate()
        self.engine = engine
        self.search_config = search_config
        self.ppo_config = ppo_config
        self.cell_operator = CellOperator(engine)
        self.backbones = {
            candidate.structure_hash: candidate for candidate in backbones
        }
        self.budgets = np.geomspace(
            float(search_config.mred_lo),
            float(search_config.mred_hi),
            int(search_config.stage2_diffam_budget_count),
        ).tolist()
        self.priors: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        self.logits: Dict[str, torch.Tensor] = {}
        self.parameter_keys: List[str] = []
        self.pending: Dict[str, _Sample] = {}
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(search_config.seed) + 271_828)
        self._initialize_parameters()
        self.optimizer = torch.optim.Adam(
            [self.logits[key] for key in self.parameter_keys],
            lr=self.ppo_config.learning_rate,
        )

    @staticmethod
    def _role(round_index: int, backbone_index: int) -> str:
        return _ROLES[(int(round_index) + int(backbone_index)) % len(_ROLES)]

    @staticmethod
    def _key(structure_hash: str, role: str) -> str:
        return f"{structure_hash}|{role}"

    def _prior(
        self, backbone: Candidate
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = self.cell_operator.slots(backbone)
        max_choices = max(
            len(self.cell_operator._table(slot[2])) for slot in slots
        )
        mask = torch.zeros((len(slots), max_choices), dtype=torch.bool)
        probabilities = torch.zeros(
            (len(slots), max_choices), dtype=torch.float64
        )
        target = max(0.0, float(self.ppo_config.init_approx_cells))
        approximate_probability = min(0.75, target / max(len(slots), 1))
        for row, slot in enumerate(slots):
            choices = len(self.cell_operator._table(slot[2]))
            mask[row, :choices] = True
            if choices == 1:
                probabilities[row, 0] = 1.0
            else:
                probabilities[row, :choices] = (
                    approximate_probability / (choices - 1)
                )
                probabilities[row, 0] = 1.0 - approximate_probability
        return probabilities, mask

    def _initialize_parameters(self) -> None:
        for structure_hash in sorted(self.backbones):
            prior, mask = self._prior(self.backbones[structure_hash])
            for role in _ROLES:
                key = self._key(structure_hash, role)
                self.priors[key] = prior.clone()
                self.masks[key] = mask.clone()
                initial = torch.full_like(prior, -30.0)
                initial[mask] = torch.log(prior[mask].clamp_min(1e-12))
                self.logits[key] = initial.requires_grad_(True)
                self.parameter_keys.append(key)

    def _probabilities(self, key: str, temperature: float) -> torch.Tensor:
        mask = self.masks[key]
        scaled = self.logits[key] / max(float(temperature), 1e-6)
        scaled = scaled.masked_fill(~mask, -1e30)
        policy = torch.softmax(scaled, dim=-1).masked_fill(~mask, 0.0)
        probabilities = (
            (1.0 - self.ppo_config.exploration) * policy
            + self.ppo_config.exploration * self.priors[key]
        )
        return probabilities.masked_fill(~mask, 0.0)

    def _sample(
        self, key: str, temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = self._probabilities(key, temperature)
        actions = torch.multinomial(
            probabilities,
            1,
            replacement=True,
            generator=self.generator,
        ).squeeze(1)
        rows = torch.arange(actions.numel())
        log_probs = torch.log(
            probabilities[rows, actions].clamp_min(1e-30)
        )
        return actions, log_probs

    def _candidate(
        self,
        backbone: Candidate,
        actions: torch.Tensor,
        *,
        round_index: int,
        sample_id: int,
        role: str,
        budget: float,
    ) -> Candidate:
        child = backbone.clone(stage=2)
        child.cells = [
            [*slot, int(action)]
            for slot, action in zip(
                self.cell_operator.slots(backbone), actions.tolist()
            )
            if int(action) != 0
        ]
        child.cells.sort(key=lambda entry: tuple(entry))
        child.operator = "cell_action_ppo"
        child.metadata.update(
            {
                "method": "action_ppo",
                "generation": int(round_index),
                "ppo_round": int(round_index),
                "ppo_sample_id": int(sample_id),
                "ppo_role": role,
                "target_mred": float(budget),
            }
        )
        child.refresh_id()
        if child.structure_hash != backbone.structure_hash:
            raise AssertionError("cell PPO changed the fixed Stage-1 structure")
        return child

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
        if not backbone_list or int(size) % len(backbone_list) != 0:
            raise ValueError(
                "cell PPO requires an equal integer quota per fixed structure"
            )
        quota = int(size) // len(backbone_list)
        selected: List[Candidate] = []
        seen = set(excluded_hashes)
        self.pending = {}
        for backbone_index, backbone in enumerate(backbone_list):
            role = self._role(round_index, backbone_index)
            band = int(backbone.metadata.get("mred_band", backbone_index))
            budget = float(
                self.budgets[(band + int(round_index)) % len(self.budgets)]
            )
            key = self._key(backbone.structure_hash, role)
            made = 0
            attempts = 0
            while made < quota and attempts < 20_000:
                attempts += 1
                temperature = self.ppo_config.temperature * (
                    1.0 + 0.25 * (attempts // 512)
                )
                actions, old_log_probs = self._sample(key, temperature)
                candidate = self._candidate(
                    backbone,
                    actions,
                    round_index=round_index,
                    sample_id=attempts - 1,
                    role=role,
                    budget=budget,
                )
                if candidate.cell_hash in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.cell_hash)
                self.pending[candidate.cell_hash] = _Sample(
                    key=key,
                    actions=actions,
                    old_log_probs=old_log_probs.detach(),
                    temperature=float(temperature),
                    role=role,
                    budget=budget,
                )
                made += 1
        if len(selected) != int(size):
            raise RuntimeError(
                f"cell PPO generated only {len(selected)}/{int(size)} candidates"
            )
        return selected

    def _reward(self, candidate: Candidate, sample: _Sample) -> float:
        anchor = self.backbones[candidate.structure_hash]
        wa, wp = _ROLE_WEIGHTS[sample.role]
        ppa_cost = (
            wa
            * math.log(
                max(float(candidate.area), 1e-15)
                / max(float(anchor.area), 1e-15)
            )
            + wp
            * math.log(
                max(float(candidate.power), 1e-15)
                / max(float(anchor.power), 1e-15)
            )
        )
        delay_penalty = self.ppo_config.delay_weight * max(
            0.0,
            float(candidate.delay) / float(self.search_config.delay_limit) - 1.0,
        )
        mred_penalty = self.ppo_config.mred_weight * max(
            0.0,
            math.log(
                max(float(candidate.mred), 1e-15)
                / max(float(sample.budget), 1e-15)
            ),
        )
        return -(ppa_cost + delay_penalty + mred_penalty)

    def update(self, evaluated: Sequence[Candidate]) -> dict:
        records = []
        for candidate in evaluated:
            sample = self.pending.get(candidate.cell_hash)
            if sample is None or not candidate.evaluated or not candidate.valid:
                continue
            records.append((candidate, sample, self._reward(candidate, sample)))
        if len(records) != len(self.pending):
            raise RuntimeError(
                "cell PPO requires a complete on-policy batch: "
                f"{len(records)}/{len(self.pending)} usable"
            )

        grouped: Dict[str, list] = {}
        for record in records:
            grouped.setdefault(record[1].key, []).append(record)
        advantages: Dict[str, float] = {}
        for items in grouped.values():
            rewards = np.asarray([item[2] for item in items], dtype=np.float64)
            mean, std = float(rewards.mean()), float(rewards.std())
            for candidate, _sample, reward in items:
                advantages[candidate.cell_hash] = (
                    0.0
                    if std < 1e-12
                    else (float(reward) - mean) / (std + 1e-8)
                )

        epoch_stats = []
        for epoch in range(self.ppo_config.epochs):
            losses, ratios_all, log_ratios_all = [], [], []
            for candidate, sample, _reward in records:
                probabilities = self._probabilities(
                    sample.key, sample.temperature
                )
                rows = torch.arange(sample.actions.numel())
                new_log_probs = torch.log(
                    probabilities[rows, sample.actions].clamp_min(1e-30)
                )
                log_ratio = new_log_probs - sample.old_log_probs
                ratio = torch.exp(log_ratio)
                advantage = torch.as_tensor(
                    advantages[candidate.cell_hash], dtype=ratio.dtype
                )
                surrogate = torch.minimum(
                    ratio * advantage,
                    torch.clamp(
                        ratio,
                        1.0 - self.ppo_config.clip_range,
                        1.0 + self.ppo_config.clip_range,
                    )
                    * advantage,
                )
                losses.append(-surrogate.mean())
                ratios_all.append(ratio.detach())
                log_ratios_all.append(log_ratio.detach())
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [self.logits[key] for key in self.parameter_keys],
                self.ppo_config.grad_clip,
            )
            self.optimizer.step()
            ratios = torch.cat(ratios_all)
            log_ratios = torch.cat(log_ratios_all)
            clipped = (
                (ratios < 1.0 - self.ppo_config.clip_range)
                | (ratios > 1.0 + self.ppo_config.clip_range)
            )
            stats = {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "ratio_mean": float(ratios.mean()),
                "ratio_min": float(ratios.min()),
                "ratio_max": float(ratios.max()),
                "clip_fraction": float(clipped.float().mean()),
                "approx_kl": float((-log_ratios).mean()),
                "grad_norm": float(grad_norm),
            }
            epoch_stats.append(stats)
            logging.info(
                "Stage 2 action PPO epoch=%d/%d loss=%.6g "
                "ratio=%.4g[%.4g,%.4g] clip=%.3f approx_kl=%.4g grad=%.4g",
                stats["epoch"],
                self.ppo_config.epochs,
                stats["loss"],
                stats["ratio_mean"],
                stats["ratio_min"],
                stats["ratio_max"],
                stats["clip_fraction"],
                stats["approx_kl"],
                stats["grad_norm"],
            )

        rewards = [record[2] for record in records]
        feasible = sum(
            float(candidate.delay) <= float(self.search_config.delay_limit)
            and float(candidate.mred) <= sample.budget
            for candidate, sample, _reward in records
        )
        self.pending = {}
        return {
            "samples": len(records),
            "reward_mean": float(np.mean(rewards)),
            "reward_best": float(np.max(rewards)),
            "budget_feasible": int(feasible),
            "epochs": epoch_stats,
        }

    def state_dict(self) -> dict:
        return {
            "ppo_config": asdict(self.ppo_config),
            "logits": {
                key: self.logits[key].detach().clone()
                for key in self.parameter_keys
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("ppo_config") != asdict(self.ppo_config):
            raise ValueError("cell PPO checkpoint hyperparameters differ")
        saved = state.get("logits") or {}
        if set(saved) != set(self.parameter_keys):
            raise ValueError("cell PPO checkpoint policy keys differ")
        with torch.no_grad():
            for key in self.parameter_keys:
                if tuple(saved[key].shape) != tuple(self.logits[key].shape):
                    raise ValueError(
                        f"cell PPO checkpoint shape mismatch for {key}"
                    )
                self.logits[key].copy_(saved[key])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("generator_state") is not None:
            self.generator.set_state(state["generator_state"])
