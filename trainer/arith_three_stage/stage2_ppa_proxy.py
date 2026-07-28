"""Differentiable real-PPA surrogate used by Stage-2 proxy DiffAM."""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .candidate import Candidate
from .cell_ops import CellOperator


_TYPE_INDEX = {0: 0, 1: 1, 4: 2}


class ColumnCellEncoder:
    """Compact column/type aggregation shared by hard data and STE inputs."""

    def __init__(self, engine, backbones: Sequence[Candidate]):
        self.engine = engine
        self.cell_operator = CellOperator(engine)
        self.backbones = {
            candidate.structure_hash: candidate for candidate in backbones
        }
        columns = [
            slot[1]
            for backbone in backbones
            for slot in self.cell_operator.slots(backbone)
        ]
        self.max_column = max(columns, default=30)
        self.choice_dim = 10
        self.column_dim = 3 * self.choice_dim * 2
        self.global_dim = 4
        self.input_dim = (self.max_column + 1) * self.column_dim + self.global_dim
        self._choice_cache: Dict[int, torch.Tensor] = {}

    def _choice_features(self, cell_type: int) -> torch.Tensor:
        cell_type = int(cell_type)
        cached = self._choice_cache.get(cell_type)
        if cached is not None:
            return cached
        table = self.cell_operator._table(cell_type)
        names = ("area", "power_mw", "delay_ns", "wae", "bias", "er", "maxe")
        raw = np.asarray(
            [
                [float((entry or {}).get(name, 0.0) or 0.0) for name in names]
                for entry in table
            ],
            dtype=np.float32,
        )
        scales = np.maximum(np.max(np.abs(raw), axis=0), 1e-9)
        norm = raw / scales
        rows = []
        denominator = max(len(table) - 1, 1)
        for index in range(len(table)):
            rows.append(
                [
                    1.0,
                    float(index != 0),
                    float(index) / denominator,
                    *norm[index].tolist(),
                ]
            )
        result = torch.tensor(rows, dtype=torch.float32)
        self._choice_cache[cell_type] = result
        return result

    @staticmethod
    def _global(backbone: Candidate, device) -> torch.Tensor:
        totals = np.asarray(
            [
                sum(backbone.ct22),
                sum(backbone.ct32),
                sum(backbone.ct42),
            ],
            dtype=np.float32,
        )
        total = max(float(totals.sum()), 1.0)
        values = [float(backbone.k) / 31.0, *(totals / total).tolist()]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def candidate_features(self, candidate: Candidate) -> torch.Tensor:
        backbone = self.backbones[candidate.structure_hash]
        slots = self.cell_operator.slots(backbone)
        selected = {
            tuple(int(value) for value in entry[:4]): int(entry[4])
            for entry in candidate.cells
        }
        grid = torch.zeros(
            (self.max_column + 1, 3, self.choice_dim * 2),
            dtype=torch.float32,
        )
        max_stage = max((slot[0] for slot in slots), default=1)
        for slot in slots:
            stage, column, cell_type, _local = slot
            choice = selected.get(slot, 0)
            table = self._choice_features(cell_type)
            if not 0 <= choice < table.shape[0]:
                raise ValueError(f"invalid proxy cell choice {choice} at {slot}")
            base = table[choice]
            stage_scale = float(stage) / max(float(max_stage), 1.0)
            grid[column, _TYPE_INDEX[int(cell_type)]] += torch.cat(
                (base, base * stage_scale)
            )
        return torch.cat((grid.flatten(), self._global(backbone, "cpu")))

    def selection_features(
        self,
        backbone: Candidate,
        graph,
        slots: Sequence[Tuple[int, int, int]],
        selection: torch.Tensor,
    ) -> torch.Tensor:
        """Build features with hard forward values and soft STE gradients."""
        device = selection.device
        flat = torch.zeros(
            ((self.max_column + 1) * 3, self.choice_dim * 2),
            dtype=torch.float32,
            device=device,
        )
        max_stage = max(
            (int(graph.vertex_list[int(node)][0]) for node, _t, _c in slots),
            default=1,
        )
        indices = []
        values = []
        for row, (node, cell_type, column) in enumerate(slots):
            choices = self._choice_features(cell_type).to(device)
            weights = selection[row, : choices.shape[0]].to(torch.float32)
            base = weights @ choices
            stage = int(graph.vertex_list[int(node)][0])
            stage_scale = float(stage) / max(float(max_stage), 1.0)
            indices.append(int(column) * 3 + _TYPE_INDEX[int(cell_type)])
            values.append(torch.cat((base, base * stage_scale)))
        if values:
            index = torch.tensor(indices, dtype=torch.long, device=device)
            flat = flat.index_add(0, index, torch.stack(values))
        return torch.cat((flat.flatten(), self._global(backbone, device)))


class _PPARegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PPAProxyTrainer:
    """Small shared ensemble trained only from real Stage-2 evaluations."""

    def __init__(self, engine, backbones, config, device: str):
        self.cfg = config
        self.device = str(device)
        self.encoder = ColumnCellEncoder(engine, backbones)
        self.models = [
            _PPARegressor(self.encoder.input_dim).to(self.device)
            for _ in range(int(config.stage2_proxy_ensemble))
        ]
        self.optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=float(config.stage2_proxy_lr),
                weight_decay=float(config.stage2_proxy_weight_decay),
            )
            for model in self.models
        ]
        self.observations: Dict[str, dict] = {}
        self.anchors: Dict[str, Tuple[float, float, float]] = {}
        self.target_mean = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.target_std = torch.ones(3, dtype=torch.float32, device=self.device)
        self.trained_updates = 0
        self.rng = np.random.default_rng(int(config.seed) + 441_731)
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self.freeze()

    @property
    def ready(self) -> bool:
        return (
            self.trained_updates > 0
            and len(self.observations) >= int(self.cfg.stage2_proxy_min_samples)
        )

    def freeze(self) -> None:
        for model in self.models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _unfreeze(self, model: nn.Module) -> None:
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    def observe(self, candidates: Iterable[Candidate]) -> int:
        added = 0
        for candidate in candidates:
            if not candidate.evaluated or not candidate.valid:
                continue
            metrics = (
                float(candidate.area),
                float(candidate.power),
                float(candidate.delay),
            )
            if any(not math.isfinite(value) or value <= 0.0 for value in metrics):
                continue
            if not candidate.cells:
                self.anchors[candidate.structure_hash] = metrics
            self.observations[candidate.cell_hash] = {
                "cell_hash": candidate.cell_hash,
                "structure_hash": candidate.structure_hash,
                "cells": copy.deepcopy(candidate.cells),
                "area": metrics[0],
                "power": metrics[1],
                "delay": metrics[2],
                "generation": int(candidate.metadata.get("generation", 0)),
            }
            added += 1
        cap = max(
            int(self.cfg.stage2_proxy_min_samples),
            int(self.cfg.stage2_proxy_observation_cap),
        )
        if len(self.observations) > cap:
            anchors = {
                key for key, value in self.observations.items() if not value["cells"]
            }
            removable = sorted(
                (
                    value
                    for key, value in self.observations.items()
                    if key not in anchors
                ),
                key=lambda value: (value["generation"], value["cell_hash"]),
            )
            for value in removable[: len(self.observations) - cap]:
                self.observations.pop(value["cell_hash"], None)
                self._feature_cache.pop(value["cell_hash"], None)
        return added

    def _candidate_from_observation(self, observation: dict) -> Candidate:
        backbone = self.encoder.backbones[observation["structure_hash"]]
        candidate = backbone.clone(stage=2)
        candidate.cells = copy.deepcopy(observation["cells"])
        candidate.refresh_id()
        return candidate

    def _features(self, observation: dict) -> torch.Tensor:
        key = observation["cell_hash"]
        cached = self._feature_cache.get(key)
        if cached is None:
            cached = self.encoder.candidate_features(
                self._candidate_from_observation(observation)
            )
            self._feature_cache[key] = cached
        return cached

    def _target(self, observation: dict) -> torch.Tensor:
        anchor = self.anchors[observation["structure_hash"]]
        values = np.asarray(
            [observation["area"], observation["power"], observation["delay"]],
            dtype=np.float64,
        )
        return torch.tensor(
            np.log(values / np.asarray(anchor, dtype=np.float64)),
            dtype=torch.float32,
        )

    def _sample_observations(self) -> List[dict]:
        eligible = [
            value
            for value in self.observations.values()
            if value["structure_hash"] in self.anchors
        ]
        limit = min(len(eligible), int(self.cfg.stage2_proxy_replay_samples))
        if len(eligible) <= limit:
            return eligible
        newest = max(value["generation"] for value in eligible)
        recent = [value for value in eligible if value["generation"] >= newest - 4]
        old = [value for value in eligible if value["generation"] < newest - 4]
        recent_count = min(len(recent), limit // 2)
        old_count = min(len(old), limit - recent_count)
        if recent_count + old_count < limit:
            recent_count = min(len(recent), limit - old_count)
        recent_pick = (
            self.rng.choice(len(recent), recent_count, replace=False).tolist()
            if recent_count else []
        )
        old_pick = (
            self.rng.choice(len(old), old_count, replace=False).tolist()
            if old_count else []
        )
        return [recent[index] for index in recent_pick] + [
            old[index] for index in old_pick
        ]

    @staticmethod
    def _rank_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        structure_ids: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for structure_id in torch.unique(structure_ids):
            indices = torch.nonzero(
                structure_ids == structure_id, as_tuple=False
            ).flatten()
            if indices.numel() < 2:
                continue
            paired = torch.roll(indices, shifts=1)
            target_delta = target[indices] - target[paired]
            prediction_delta = prediction[indices] - prediction[paired]
            sign = torch.sign(target_delta)
            mask = sign != 0
            if mask.any():
                losses.append(
                    F.softplus(-sign[mask] * prediction_delta[mask]).mean()
                )
        if not losses:
            return prediction.sum() * 0.0
        return torch.stack(losses).mean()

    def fit(self) -> dict:
        if len(self.observations) < int(self.cfg.stage2_proxy_min_samples):
            self.freeze()
            return {"ready": False, "samples": len(self.observations)}
        observations = self._sample_observations()
        if len(observations) < int(self.cfg.stage2_proxy_min_samples):
            self.freeze()
            return {"ready": False, "samples": len(observations)}
        features = torch.stack([self._features(value) for value in observations])
        targets = torch.stack([self._target(value) for value in observations])
        structure_keys = sorted({value["structure_hash"] for value in observations})
        structure_index = {key: index for index, key in enumerate(structure_keys)}
        structure_ids = torch.tensor(
            [structure_index[value["structure_hash"]] for value in observations],
            dtype=torch.long,
        )
        self.target_mean = targets.mean(dim=0).to(self.device)
        self.target_std = targets.std(dim=0).clamp_min(1e-3).to(self.device)
        normalized = (
            targets.to(self.device) - self.target_mean
        ) / self.target_std
        features = features.to(self.device)
        structure_ids = structure_ids.to(self.device)
        batch_size = max(16, int(self.cfg.stage2_proxy_batch_size))
        epochs = max(1, int(self.cfg.stage2_proxy_epochs))
        losses = []
        for model_index, (model, optimizer) in enumerate(
            zip(self.models, self.optimizers)
        ):
            self._unfreeze(model)
            model_losses = []
            for _epoch in range(epochs):
                bootstrap = self.rng.integers(0, len(observations), len(observations))
                for start in range(0, len(bootstrap), batch_size):
                    indices = torch.tensor(
                        bootstrap[start : start + batch_size],
                        dtype=torch.long,
                        device=self.device,
                    )
                    prediction = model(features[indices])
                    regression = F.huber_loss(
                        prediction,
                        normalized[indices],
                        delta=1.0,
                    )
                    ranking = self._rank_loss(
                        prediction,
                        normalized[indices],
                        structure_ids[indices],
                    )
                    loss = regression + float(
                        self.cfg.stage2_proxy_rank_weight
                    ) * ranking
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    model_losses.append(float(loss.detach()))
            losses.append(float(np.mean(model_losses)))
        self.trained_updates += 1
        self.freeze()
        return {
            "ready": True,
            "samples": len(observations),
            "updates": self.trained_updates,
            "loss": float(np.mean(losses)),
        }

    def predict_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.ready:
            raise RuntimeError("PPA proxy is not trained")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        predictions = torch.stack([model(features) for model in self.models])
        mean_z = predictions.mean(dim=0)
        std_z = predictions.std(dim=0, unbiased=False)
        mean_log_ratio = mean_z * self.target_std + self.target_mean
        std_log_ratio = std_z * self.target_std.abs()
        return mean_log_ratio, std_log_ratio

    def predict_candidate(self, candidate: Candidate) -> Tuple[np.ndarray, np.ndarray]:
        features = self.encoder.candidate_features(candidate).to(self.device)
        with torch.no_grad():
            mean, std = self.predict_features(features)
        anchor = np.asarray(self.anchors[candidate.structure_hash], dtype=np.float64)
        ppa = anchor * np.exp(mean[0].cpu().numpy())
        return ppa, std[0].cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "models": [
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for model in self.models
            ],
            "observations": copy.deepcopy(self.observations),
            "anchors": copy.deepcopy(self.anchors),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "trained_updates": int(self.trained_updates),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        for model, payload in zip(self.models, state.get("models") or []):
            model.load_state_dict(payload)
            model.to(self.device)
        self.observations = copy.deepcopy(state.get("observations") or {})
        self.anchors = {
            str(key): tuple(float(value) for value in values)
            for key, values in (state.get("anchors") or {}).items()
        }
        self.target_mean = state.get(
            "target_mean", torch.zeros(3)
        ).to(self.device)
        self.target_std = state.get(
            "target_std", torch.ones(3)
        ).to(self.device)
        self.trained_updates = int(state.get("trained_updates", 0))
        if state.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.freeze()
