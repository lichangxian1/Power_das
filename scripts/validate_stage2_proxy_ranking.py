#!/usr/bin/env python3
"""Offline temporal ablation for the Stage-2 PPA ranking surrogate.

The active Stage-2 run is never modified.  A copied checkpoint supplies:

* proxy observations through generation G;
* generation G+1 in ``unconsumed_observations`` as a strict temporal test;
* the saved online proxy as E0.

E1 keeps the current aggregate feature/model and changes only grouping/loss.
E2 replaces the collision-prone aggregate with slot-level DeepSets.
E3 adds a nominal-area analytical residual to E2.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.train_three_stage import build_engine
from trainer.arith_three_stage import Candidate, ThreeStageConfig
from trainer.arith_three_stage.stage2_ppa_proxy import (
    ColumnCellEncoder,
    PPAProxyTrainer,
)


METRICS = ("area", "power", "delay")
EXACT_NATIVE_AREA = {0: 2.856, 1: 2.184, 4: 5.712}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_readonly_engine(cfg: ThreeStageConfig):
    args = SimpleNamespace(
        config=cfg.engine_config_path,
        target_delay=cfg.delay_limit,
        error_vectors=65_536,
        out="/tmp/stage2_proxy_ranking_validation",
        dc_batch=1,
        stage3_num_epochs=1,
        dc_parallelism=1,
        device="cpu",
        seed=cfg.seed,
        k_min=cfg.k_min,
        approx_col_window=cfg.approx_col_window,
        approx_lib_path=cfg.approx_lib_path,
        approx42_library_path=cfg.approx42_library_path,
        approx42_rtl_path=cfg.approx42_rtl_path,
        stage3_normalize_advantage=True,
        stage3_single_elite_index=None,
        stage3_episodes_per_elite=1,
    )
    return build_engine(args)


def record_candidate(record: dict, backbones: Dict[str, Candidate]) -> Candidate:
    candidate = backbones[record["structure_hash"]].clone(stage=2)
    candidate.cells = copy.deepcopy(record["cells"])
    candidate.refresh_id()
    return candidate


def observation_from_candidate(candidate: Candidate) -> dict:
    return {
        "cell_hash": candidate.cell_hash,
        "structure_hash": candidate.structure_hash,
        "cells": copy.deepcopy(candidate.cells),
        "area": float(candidate.area),
        "power": float(candidate.power),
        "delay": float(candidate.delay),
        "generation": int(candidate.metadata.get("generation", -1)),
    }


def target_for(record: dict, anchors: Dict[str, Sequence[float]]) -> np.ndarray:
    actual = np.asarray(
        [record["area"], record["power"], record["delay"]], dtype=np.float64
    )
    anchor = np.asarray(anchors[record["structure_hash"]], dtype=np.float64)
    return np.log(actual / anchor).astype(np.float32)


class SlotTokenEncoder:
    """Hard candidate encoder with an online-STE-compatible slot ordering."""

    def __init__(
        self,
        engine,
        backbones: Sequence[Candidate],
        choice_encoder: ColumnCellEncoder,
    ):
        self.engine = engine
        self.choice_encoder = choice_encoder
        self.backbones = {
            candidate.structure_hash: candidate for candidate in backbones
        }
        self.structure_keys = sorted(self.backbones)
        self.structure_index = {
            key: index for index, key in enumerate(self.structure_keys)
        }
        self.slots = {
            key: choice_encoder.cell_operator.slots(backbone)
            for key, backbone in self.backbones.items()
        }
        self.max_slots = max(len(value) for value in self.slots.values())
        self.position_dim = 8
        self.token_dim = choice_encoder.choice_dim + self.position_dim
        self._position_cache: Dict[str, torch.Tensor] = {}

    def _positions(self, structure_hash: str) -> torch.Tensor:
        cached = self._position_cache.get(structure_hash)
        if cached is not None:
            return cached
        backbone = self.backbones[structure_hash]
        slots = self.slots[structure_hash]
        max_stage = max((slot[0] for slot in slots), default=1)
        max_local = max((slot[3] for slot in slots), default=1)
        rows = []
        for index, (stage, column, cell_type, local) in enumerate(slots):
            type_onehot = [
                float(cell_type == 0),
                float(cell_type == 1),
                float(cell_type == 4),
            ]
            rows.append(
                [
                    float(stage) / max(max_stage, 1),
                    float(column) / max(self.choice_encoder.max_column, 1),
                    float(column - backbone.k)
                    / max(self.choice_encoder.max_column, 1),
                    float(local) / max(max_local, 1),
                    *type_onehot,
                    float(index) / max(len(slots) - 1, 1),
                ]
            )
        result = torch.tensor(rows, dtype=torch.float32)
        self._position_cache[structure_hash] = result
        return result

    def encode(
        self, candidate: Candidate
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        structure_hash = candidate.structure_hash
        backbone = self.backbones[structure_hash]
        slots = self.slots[structure_hash]
        selected = {
            tuple(int(value) for value in entry[:4]): int(entry[4])
            for entry in candidate.cells
        }
        tokens = torch.zeros((self.max_slots, self.token_dim), dtype=torch.float32)
        mask = torch.zeros(self.max_slots, dtype=torch.bool)
        nominal_area = 0.0
        exact_area = 0.0
        positions = self._positions(structure_hash)
        for row, slot in enumerate(slots):
            cell_type = int(slot[2])
            choice = selected.get(slot, 0)
            choice_features = self.choice_encoder._choice_features(cell_type)[choice]
            tokens[row] = torch.cat((choice_features, positions[row]))
            mask[row] = True
            exact = float(EXACT_NATIVE_AREA[cell_type])
            exact_area += exact
            if choice == 0:
                nominal_area += exact
            else:
                entry = self.choice_encoder.cell_operator._table(cell_type)[choice]
                raw = entry.get("area")
                nominal_area += exact if raw is None else float(raw)
        base = torch.tensor(
            [math.log(max(nominal_area / max(exact_area, 1e-12), 1e-12)), 0.0, 0.0],
            dtype=torch.float32,
        )
        global_features = self.choice_encoder._global(backbone, "cpu")
        return (
            tokens,
            mask,
            global_features,
            base,
            self.structure_index[structure_hash],
        )


class AggregateRegressor(nn.Module):
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


class SlotDeepSetsRegressor(nn.Module):
    def __init__(
        self,
        token_dim: int,
        n_structures: int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        *,
        area_residual: bool,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.structure_embedding = nn.Embedding(n_structures, 8)
        self.rho = nn.Sequential(
            nn.Linear(64 * 2 + 8 + 4, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.area_residual = bool(area_residual)
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_std", target_std.detach().clone())

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        global_features: torch.Tensor,
        baseline: torch.Tensor,
        structure_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.phi(tokens)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        summed = (hidden * weights).sum(dim=1) / torch.sqrt(count)
        mean = (hidden * weights).sum(dim=1) / count
        merged = torch.cat(
            (
                summed,
                mean,
                self.structure_embedding(structure_ids),
                global_features,
            ),
            dim=-1,
        )
        prediction = self.rho(merged)
        if self.area_residual:
            baseline_z = (baseline - self.target_mean) / self.target_std
            prediction = prediction + torch.stack(
                (
                    baseline_z[:, 0],
                    torch.zeros_like(baseline_z[:, 1]),
                    torch.zeros_like(baseline_z[:, 2]),
                ),
                dim=-1,
            )
        return prediction


def group_indices(records: Sequence[dict]) -> List[List[int]]:
    grouped: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[
            (str(record["structure_hash"]), int(record["generation"]))
        ].append(index)
    return [
        sorted(indices, key=lambda index: records[index]["cell_hash"])
        for _key, indices in sorted(grouped.items())
        if len(indices) == 4
    ]


def listwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Prediction/target are [groups, four candidates, three metrics]."""
    metric_weights = prediction.new_tensor([1.0, 1.0, 0.25])
    pred_center = prediction - prediction.mean(dim=1, keepdim=True)
    target_center = target - target.mean(dim=1, keepdim=True)
    scale = target_center.std(dim=1, unbiased=False, keepdim=True).clamp_min(0.10)
    target_probability = torch.softmax(-target_center / scale, dim=1)
    prediction_log_probability = torch.log_softmax(-pred_center / scale, dim=1)
    listnet = -(target_probability * prediction_log_probability).sum(dim=1)
    listnet = (listnet * metric_weights).sum(dim=-1) / metric_weights.sum()
    centered_huber = F.smooth_l1_loss(
        pred_center,
        target_center,
        reduction="none",
        beta=0.5,
    ).mean(dim=1)
    centered_huber = (
        centered_huber * metric_weights
    ).sum(dim=-1) / metric_weights.sum()
    absolute_huber = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=0.5,
    ).mean(dim=1)
    absolute_huber = (
        absolute_huber * metric_weights
    ).sum(dim=-1) / metric_weights.sum()
    return (
        listnet.mean()
        + 0.30 * centered_huber.mean()
        + 0.05 * absolute_huber.mean()
    )


def gather_model_inputs(
    kind: str,
    indices: torch.Tensor,
    tensors: dict,
    device: str,
) -> tuple:
    flat = indices.flatten()
    if kind == "aggregate":
        return (tensors["aggregate"][flat].to(device),)
    return (
        tensors["tokens"][flat].to(device),
        tensors["mask"][flat].to(device),
        tensors["global"][flat].to(device),
        tensors["baseline"][flat].to(device),
        tensors["structure_id"][flat].to(device),
    )


def predict_model(
    model: nn.Module,
    kind: str,
    tensors: dict,
    indices: Sequence[int],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            block = torch.tensor(
                indices[start : start + batch_size], dtype=torch.long
            )
            args = gather_model_inputs(kind, block, tensors, device)
            outputs.append(model(*args).cpu())
    return torch.cat(outputs, dim=0)


def validation_score(
    prediction: np.ndarray,
    target: np.ndarray,
    structures: Sequence[str],
) -> float:
    """Mean area/power within-structure pair accuracy."""
    scores = []
    for metric in (0, 1):
        correct = total = 0
        for structure in sorted(set(structures)):
            indices = [i for i, value in enumerate(structures) if value == structure]
            for x in range(len(indices)):
                for y in range(x):
                    delta = target[indices[x], metric] - target[indices[y], metric]
                    if delta == 0:
                        continue
                    pred_delta = (
                        prediction[indices[x], metric]
                        - prediction[indices[y], metric]
                    )
                    correct += int(np.sign(delta) == np.sign(pred_delta))
                    total += 1
        scores.append(correct / max(total, 1))
    return float(np.mean(scores))


def train_ensemble(
    *,
    name: str,
    kind: str,
    tensors: dict,
    train_records: Sequence[dict],
    validation_records: Sequence[dict],
    all_records: Sequence[dict],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    token_dim: int,
    n_structures: int,
    area_residual: bool,
    device: str,
    seeds: Sequence[int],
    epochs: int,
    patience: int,
    batch_groups: int,
) -> Tuple[List[nn.Module], dict]:
    train_groups_local = group_indices(train_records)
    validation_indices = [
        index
        for index, record in enumerate(all_records)
        if record in validation_records
    ]
    train_lookup = {
        record["cell_hash"]: index for index, record in enumerate(all_records)
    }
    train_groups = [
        [train_lookup[train_records[index]["cell_hash"]] for index in group]
        for group in train_groups_local
    ]
    refit_records = list(train_records) + list(validation_records)
    refit_groups = [
        [train_lookup[refit_records[index]["cell_hash"]] for index in group]
        for group in group_indices(refit_records)
    ]
    validation_targets = np.stack(
        [target_for(record, tensors["anchors"]) for record in validation_records]
    )
    validation_structures = [
        str(record["structure_hash"]) for record in validation_records
    ]
    models = []
    histories = []
    for member, seed in enumerate(seeds):
        seed_all(seed)
        if kind == "aggregate":
            model = AggregateRegressor(tensors["aggregate"].shape[1]).to(device)
        else:
            model = SlotDeepSetsRegressor(
                token_dim,
                n_structures,
                target_mean,
                target_std,
                area_residual=area_residual,
            ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        rng = np.random.default_rng(seed)
        best_state = None
        best_score = -math.inf
        best_epoch = 0
        stale = 0
        epoch_rows = []
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(train_groups))
            losses = []
            for start in range(0, len(order), batch_groups):
                chosen = order[start : start + batch_groups]
                index_tensor = torch.tensor(
                    [train_groups[int(index)] for index in chosen],
                    dtype=torch.long,
                )
                args = gather_model_inputs(kind, index_tensor, tensors, device)
                prediction = model(*args).reshape(len(chosen), 4, 3)
                target = tensors["target_z"][index_tensor.flatten()].to(device)
                target = target.reshape(len(chosen), 4, 3)
                loss = listwise_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            val_z = predict_model(
                model, kind, tensors, validation_indices, device
            ).numpy()
            val_prediction = val_z * target_std.numpy() + target_mean.numpy()
            score = validation_score(
                val_prediction, validation_targets, validation_structures
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_pair_area_power": score,
                }
            )
            if score > best_score + 1e-9:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError(f"{name} member {member} produced no checkpoint")
        # Select the epoch count on the held-out generation, then refit from
        # scratch on train+validation. The temporal test remains untouched,
        # and E1-E3 now see the same 8,192 labels as saved online E0.
        seed_all(seed + 500_003)
        if kind == "aggregate":
            refit_model = AggregateRegressor(
                tensors["aggregate"].shape[1]
            ).to(device)
        else:
            refit_model = SlotDeepSetsRegressor(
                token_dim,
                n_structures,
                target_mean,
                target_std,
                area_residual=area_residual,
            ).to(device)
        refit_optimizer = torch.optim.AdamW(
            refit_model.parameters(), lr=3e-4, weight_decay=1e-4
        )
        refit_rng = np.random.default_rng(seed + 500_003)
        for _epoch in range(best_epoch):
            refit_model.train()
            order = refit_rng.permutation(len(refit_groups))
            for start in range(0, len(order), batch_groups):
                chosen = order[start : start + batch_groups]
                index_tensor = torch.tensor(
                    [refit_groups[int(index)] for index in chosen],
                    dtype=torch.long,
                )
                model_args = gather_model_inputs(
                    kind, index_tensor, tensors, device
                )
                prediction = refit_model(*model_args).reshape(
                    len(chosen), 4, 3
                )
                target = tensors["target_z"][
                    index_tensor.flatten()
                ].to(device)
                target = target.reshape(len(chosen), 4, 3)
                loss = listwise_loss(prediction, target)
                refit_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    refit_model.parameters(), 2.0
                )
                refit_optimizer.step()
        refit_model.eval()
        models.append(refit_model)
        histories.append(
            {
                "member": member,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_pair_area_power": best_score,
                "epochs": epoch_rows,
            }
        )
        print(
            f"[{name}] member={member} best_epoch={best_epoch} "
            f"validation_pair={best_score:.4f}",
            flush=True,
        )
    return models, {"members": histories}


def ensemble_prediction(
    models: Sequence[nn.Module],
    kind: str,
    tensors: dict,
    indices: Sequence[int],
    device: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> np.ndarray:
    values = [
        predict_model(model, kind, tensors, indices, device).numpy()
        for model in models
    ]
    mean_z = np.mean(values, axis=0)
    return mean_z * target_std.numpy() + target_mean.numpy()


def detailed_metrics(
    prediction_log_ratio: np.ndarray,
    records: Sequence[dict],
    anchors: Dict[str, Sequence[float]],
) -> dict:
    targets = np.stack([target_for(record, anchors) for record in records])
    actual = []
    predicted = []
    structures = []
    for row, record in enumerate(records):
        anchor = np.asarray(anchors[record["structure_hash"]], dtype=np.float64)
        actual.append([record[name] for name in METRICS])
        predicted.append(anchor * np.exp(prediction_log_ratio[row]))
        structures.append(str(record["structure_hash"]))
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    result = {}
    for metric, name in enumerate(METRICS):
        a = actual[:, metric]
        p = predicted[:, metric]
        pair_correct = pair_total = top1 = groups = 0
        for structure in sorted(set(structures)):
            indices = np.asarray(
                [i for i, value in enumerate(structures) if value == structure]
            )
            if len(indices) < 2:
                continue
            groups += 1
            top1 += int(
                indices[np.argmin(a[indices])]
                == indices[np.argmin(p[indices])]
            )
            for x in range(len(indices)):
                for y in range(x):
                    delta = a[indices[x]] - a[indices[y]]
                    if delta == 0:
                        continue
                    pair_correct += int(
                        np.sign(delta)
                        == np.sign(p[indices[x]] - p[indices[y]])
                    )
                    pair_total += 1
        ape = np.abs(p - a) / a
        result[name] = {
            "mape_pct": float(ape.mean() * 100),
            "median_ape_pct": float(np.median(ape) * 100),
            "p90_ape_pct": float(np.quantile(ape, 0.9) * 100),
            "pearson": float(pearsonr(a, p).statistic),
            "spearman": float(spearmanr(a, p).statistic),
            "within_structure_pair_accuracy_pct": float(
                pair_correct / max(pair_total, 1) * 100
            ),
            "within_structure_kendall_tau": float(
                2.0 * pair_correct / max(pair_total, 1) - 1.0
            ),
            "within_structure_top1_match_pct": float(
                top1 / max(groups, 1) * 100
            ),
        }
    return result


def saved_proxy_prediction(
    engine,
    backbones: Sequence[Candidate],
    cfg: ThreeStageConfig,
    proxy_state: dict,
    records: Sequence[dict],
    device: str,
) -> np.ndarray:
    proxy = PPAProxyTrainer(engine, backbones, cfg, device)
    proxy.load_state_dict(proxy_state)
    output = []
    with torch.no_grad():
        for record in records:
            candidate = record_candidate(
                record, {value.structure_hash: value for value in backbones}
            )
            prediction, _uncertainty = proxy.predict_candidate(candidate)
            anchor = np.asarray(proxy.anchors[record["structure_hash"]])
            output.append(np.log(prediction / anchor))
    return np.asarray(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--ensemble", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.loads((run_dir / "three_stage_config.json").read_text())
    cfg = ThreeStageConfig(**raw_cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    proxy_state = state["extra_state"]["diffam_proxy_state"]["proxy"]
    anchors = {
        str(key): tuple(float(value) for value in values)
        for key, values in proxy_state["anchors"].items()
    }
    backbones = [
        Candidate.from_dict(payload)
        for payload in json.loads(Path(cfg.stage1_backbones_source).read_text())
    ]
    backbone_map = {candidate.structure_hash: candidate for candidate in backbones}
    observations = [
        copy.deepcopy(value) for value in proxy_state["observations"].values()
    ]
    temporal_test = [
        observation_from_candidate(Candidate.from_dict(payload))
        for payload in state["extra_state"]["unconsumed_observations"]
    ]
    maximum_generation = max(int(value["generation"]) for value in observations)
    validation_generation = maximum_generation
    train_records = [
        value
        for value in observations
        if int(value["generation"]) < validation_generation
    ]
    validation_records = [
        value
        for value in observations
        if int(value["generation"]) == validation_generation
    ]
    test_records = temporal_test
    if len(validation_records) != 128 or len(test_records) != 128:
        raise RuntimeError(
            "expected 128 validation and temporal-test candidates, got "
            f"{len(validation_records)} and {len(test_records)}"
        )
    all_records = train_records + validation_records + test_records
    engine = build_readonly_engine(cfg)
    aggregate_encoder = ColumnCellEncoder(engine, backbones)
    slot_encoder = SlotTokenEncoder(engine, backbones, aggregate_encoder)

    aggregate = []
    tokens = []
    masks = []
    global_features = []
    baselines = []
    structure_ids = []
    targets = []
    for index, record in enumerate(all_records):
        candidate = record_candidate(record, backbone_map)
        aggregate.append(aggregate_encoder.candidate_features(candidate))
        token, mask, global_feature, baseline, structure_id = slot_encoder.encode(
            candidate
        )
        tokens.append(token)
        masks.append(mask)
        global_features.append(global_feature)
        baselines.append(baseline)
        structure_ids.append(structure_id)
        targets.append(torch.tensor(target_for(record, anchors)))
        if (index + 1) % 1024 == 0:
            print(f"[features] {index + 1}/{len(all_records)}", flush=True)
    target = torch.stack(targets)
    train_count = len(train_records)
    target_mean = target[:train_count].mean(dim=0)
    target_std = target[:train_count].std(dim=0).clamp_min(1e-3)
    tensors = {
        "aggregate": torch.stack(aggregate),
        "tokens": torch.stack(tokens),
        "mask": torch.stack(masks),
        "global": torch.stack(global_features),
        "baseline": torch.stack(baselines),
        "structure_id": torch.tensor(structure_ids, dtype=torch.long),
        "target_z": (target - target_mean) / target_std,
        "anchors": anchors,
    }
    validation_start = len(train_records)
    test_start = validation_start + len(validation_records)
    test_indices = list(range(test_start, len(all_records)))
    seeds = [int(cfg.seed) + 90_001 + index * 101 for index in range(args.ensemble)]

    results = {
        "checkpoint_generation": int(state["generation"]),
        "proxy_observations": len(observations),
        "proxy_updates": int(proxy_state["trained_updates"]),
        "train_count": len(train_records),
        "validation_generation": validation_generation,
        "validation_count": len(validation_records),
        "test_generations": sorted(
            {int(record["generation"]) for record in test_records}
        ),
        "test_count": len(test_records),
        "models": {},
    }
    e0_prediction = saved_proxy_prediction(
        engine,
        backbones,
        cfg,
        proxy_state,
        test_records,
        args.device,
    )
    results["models"]["E0_saved_online_proxy"] = {
        "test": detailed_metrics(e0_prediction, test_records, anchors)
    }
    print("[E0] saved online proxy evaluated", flush=True)

    variants = (
        ("E1_group_listwise_aggregate", "aggregate", False),
        ("E2_group_listwise_slot_deepsets", "slot", False),
        ("E3_slot_deepsets_area_residual", "slot", True),
    )
    for name, kind, area_residual in variants:
        models, history = train_ensemble(
            name=name,
            kind=kind,
            tensors=tensors,
            train_records=train_records,
            validation_records=validation_records,
            all_records=all_records,
            target_mean=target_mean,
            target_std=target_std,
            token_dim=slot_encoder.token_dim,
            n_structures=len(slot_encoder.structure_keys),
            area_residual=area_residual,
            device=args.device,
            seeds=seeds,
            epochs=args.epochs,
            patience=args.patience,
            batch_groups=args.batch_groups,
        )
        prediction = ensemble_prediction(
            models,
            kind,
            tensors,
            test_indices,
            args.device,
            target_mean,
            target_std,
        )
        results["models"][name] = {
            "history": history,
            "test": detailed_metrics(prediction, test_records, anchors),
        }
        torch.save(
            {
                "kind": kind,
                "area_residual": area_residual,
                "models": [
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    for model in models
                ],
                "target_mean": target_mean,
                "target_std": target_std,
                "structure_keys": slot_encoder.structure_keys,
                "token_dim": slot_encoder.token_dim,
            },
            out_dir / f"{name}.pt",
        )
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(
            f"[{name}] test="
            + json.dumps(results["models"][name]["test"], ensure_ascii=False),
            flush=True,
        )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
