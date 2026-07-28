#!/usr/bin/env python3
"""Warm-start Stage-2 PPA proxy ranking-head ablation.

This experiment never calls DC and never mutates the active run.  It compares:

* E0: saved online ensemble without further updates;
* sanity: the same E0 weights continued for a fixed budget;
* dual: E0 plus an auxiliary per-candidate rank head.  Pair deltas are formed
  as r(x_i) - r(x_j), so they are antisymmetric and transitive.  DiffAM would
  continue to consume only the inherited absolute PPA head.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from trainer.arith_three_stage import Candidate, ThreeStageConfig
from trainer.arith_three_stage.stage2_ppa_proxy import ColumnCellEncoder
from scripts.validate_stage2_proxy_ranking import (
    build_readonly_engine,
    detailed_metrics,
    observation_from_candidate,
    record_candidate,
    target_for,
)


METRICS = ("area", "power", "delay")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WarmRankRegressor(nn.Module):
    """E0 backbone/absolute head plus an auxiliary scalar rank head."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.absolute_head = nn.Linear(128, 3)
        self.rank_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.rank_head[-1].weight)
        nn.init.zeros_(self.rank_head[-1].bias)

    def load_e0(self, state: dict) -> None:
        own = self.state_dict()
        mapping = {
            "backbone.0.weight": "net.0.weight",
            "backbone.0.bias": "net.0.bias",
            "backbone.1.weight": "net.1.weight",
            "backbone.1.bias": "net.1.bias",
            "backbone.4.weight": "net.4.weight",
            "backbone.4.bias": "net.4.bias",
            "absolute_head.weight": "net.6.weight",
            "absolute_head.bias": "net.6.bias",
        }
        for destination, source in mapping.items():
            own[destination] = state[source].detach().clone()
        self.load_state_dict(own)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.absolute_head(hidden), self.rank_head(hidden)


def complete_groups(records: Sequence[dict]) -> List[List[int]]:
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


def cyclic_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the current proxy's one rolled pair per sample."""
    paired_prediction = torch.roll(prediction, shifts=1, dims=1)
    paired_target = torch.roll(target, shifts=1, dims=1)
    target_delta = target - paired_target
    prediction_delta = prediction - paired_prediction
    sign = torch.sign(target_delta)
    mask = sign != 0
    if not mask.any():
        return prediction.sum() * 0.0
    return F.softplus(-sign[mask] * prediction_delta[mask]).mean()


def trusted_pair_loss(
    rank_score: torch.Tensor,
    target_z: torch.Tensor,
    target_std: torch.Tensor,
    tau_log: float,
) -> Tuple[torch.Tensor, int]:
    """All six pairs/group, excluding differences inside the dead band."""
    target_log = target_z * target_std.view(1, 1, 3)
    losses = []
    valid_count = 0
    for left in range(4):
        for right in range(left):
            delta = target_log[:, left] - target_log[:, right]
            score_delta = rank_score[:, left] - rank_score[:, right]
            mask = delta.abs() >= float(tau_log)
            sign = torch.sign(delta)
            if mask.any():
                losses.append(F.softplus(-sign[mask] * score_delta[mask]))
                valid_count += int(mask.sum())
    if not losses:
        return rank_score.sum() * 0.0, 0
    return torch.cat(losses).mean(), valid_count


def distill_rank_head(
    model: WarmRankRegressor,
    features: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    device: str,
    seed: int,
    epochs: int,
    batch_groups: int,
) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.absolute_head.parameters():
        parameter.requires_grad_(False)
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.rank_head.parameters(), lr=3e-4, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    model.eval()
    model.rank_head.train()
    for _epoch in range(epochs):
        order = rng.permutation(len(groups))
        for start in range(0, len(order), batch_groups):
            chosen = order[start : start + batch_groups]
            index = torch.tensor(
                [groups[int(value)] for value in chosen],
                dtype=torch.long,
            )
            block = features[index.flatten()].to(device)
            with torch.no_grad():
                hidden = model.backbone(block)
                teacher = model.absolute_head(hidden).reshape(
                    len(chosen), 4, 3
                )
                teacher = teacher - teacher.mean(dim=1, keepdim=True)
            student = model.rank_head(hidden).reshape(len(chosen), 4, 3)
            student = student - student.mean(dim=1, keepdim=True)
            loss = F.mse_loss(student, teacher)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def continue_training(
    model: WarmRankRegressor,
    features: torch.Tensor,
    target_z: torch.Tensor,
    groups: Sequence[Sequence[int]],
    target_std: torch.Tensor,
    *,
    dual: bool,
    tau_log: float,
    device: str,
    seed: int,
    epochs: int,
    batch_groups: int,
) -> dict:
    if dual:
        distill_rank_head(
            model,
            features,
            groups,
            device=device,
            seed=seed + 17,
            epochs=2,
            batch_groups=batch_groups,
        )
    base_parameters = list(model.backbone.parameters()) + list(
        model.absolute_head.parameters()
    )
    parameter_groups = [{"params": base_parameters, "lr": 3e-5}]
    if dual:
        parameter_groups.append(
            {"params": model.rank_head.parameters(), "lr": 3e-4}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(groups))
        absolute_losses = []
        auxiliary_losses = []
        valid_pairs = 0
        for start in range(0, len(order), batch_groups):
            chosen = order[start : start + batch_groups]
            index = torch.tensor(
                [groups[int(value)] for value in chosen],
                dtype=torch.long,
            )
            block = features[index.flatten()].to(device)
            truth = target_z[index.flatten()].to(device).reshape(
                len(chosen), 4, 3
            )
            absolute, rank_score = model(block)
            absolute = absolute.reshape(len(chosen), 4, 3)
            rank_score = rank_score.reshape(len(chosen), 4, 3)
            regression = F.huber_loss(
                absolute, truth, delta=1.0
            )
            original_rank = cyclic_rank_loss(absolute, truth)
            loss = regression + 0.30 * original_rank
            auxiliary = absolute.sum() * 0.0
            count = 0
            if dual:
                auxiliary, count = trusted_pair_loss(
                    rank_score,
                    truth,
                    target_std.to(device),
                    tau_log,
                )
                loss = loss + 0.30 * auxiliary
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute_losses.append(
                float((regression + 0.30 * original_rank).detach())
            )
            auxiliary_losses.append(float(auxiliary.detach()))
            valid_pairs += count
        rows.append(
            {
                "epoch": epoch,
                "absolute_loss": float(np.mean(absolute_losses)),
                "auxiliary_loss": float(np.mean(auxiliary_losses)),
                "trusted_pair_terms": valid_pairs,
            }
        )
    model.eval()
    return {"epochs": rows}


def predict_ensemble(
    models: Sequence[WarmRankRegressor],
    features: torch.Tensor,
    indices: Sequence[int],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str,
) -> np.ndarray:
    member_predictions = []
    with torch.no_grad():
        for model in models:
            outputs = []
            for start in range(0, len(indices), 256):
                block_index = indices[start : start + 256]
                block = features[block_index].to(device)
                absolute, _rank = model(block)
                outputs.append(absolute.cpu())
            member_predictions.append(torch.cat(outputs).numpy())
    mean_z = np.mean(member_predictions, axis=0)
    return mean_z * target_std.numpy() + target_mean.numpy()


def tie_aware_metrics(
    prediction_log_ratio: np.ndarray,
    records: Sequence[dict],
    anchors: Dict[str, Sequence[float]],
    tau_pct: float,
) -> dict:
    tau_log = float(tau_pct) / 100.0
    grouped: Dict[str, List[int]] = defaultdict(list)
    target = np.stack([target_for(record, anchors) for record in records])
    for index, record in enumerate(records):
        grouped[str(record["structure_hash"])].append(index)
    result = {}
    for metric, name in enumerate(METRICS):
        correct = total = tolerant_top1 = groups = 0
        for indices in grouped.values():
            values = target[indices, metric]
            predicted = prediction_log_ratio[indices, metric]
            groups += 1
            selected = int(np.argmin(predicted))
            tolerant_top1 += int(
                values[selected] - float(values.min()) <= tau_log
            )
            for left in range(len(indices)):
                for right in range(left):
                    delta = float(values[left] - values[right])
                    if abs(delta) < tau_log or delta == 0.0:
                        continue
                    predicted_delta = float(predicted[left] - predicted[right])
                    correct += int(np.sign(delta) == np.sign(predicted_delta))
                    total += 1
        result[name] = {
            "valid_pairs": total,
            "pair_accuracy_pct": float(correct / max(total, 1) * 100),
            "tolerant_top1_pct": float(
                tolerant_top1 / max(groups, 1) * 100
            ),
        }
    return result


def make_model(
    input_dim: int,
    e0_state: dict,
    device: str,
) -> WarmRankRegressor:
    model = WarmRankRegressor(input_dim)
    model.load_e0(e0_state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument(
        "--tie-pct",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 2.0),
    )
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
    if len(temporal_test) != 128:
        raise RuntimeError(
            f"expected 128 temporal-test candidates, got {len(temporal_test)}"
        )
    all_records = observations + temporal_test
    engine = build_readonly_engine(cfg)
    encoder = ColumnCellEncoder(engine, backbones)
    feature_rows = []
    target_rows = []
    for index, record in enumerate(all_records):
        feature_rows.append(
            encoder.candidate_features(record_candidate(record, backbone_map))
        )
        target_rows.append(torch.tensor(target_for(record, anchors)))
        if (index + 1) % 1024 == 0:
            print(f"[features] {index + 1}/{len(all_records)}", flush=True)
    features = torch.stack(feature_rows)
    target_log = torch.stack(target_rows)
    target_mean = proxy_state["target_mean"].to(torch.float32)
    target_std = proxy_state["target_std"].to(torch.float32)
    target_z = (target_log - target_mean) / target_std
    groups = complete_groups(observations)
    test_indices = list(range(len(observations), len(all_records)))
    e0_states = proxy_state["models"]
    if len(e0_states) != 3:
        raise RuntimeError(f"expected three E0 members, got {len(e0_states)}")

    results = {
        "checkpoint_generation": int(state["generation"]),
        "proxy_updates": int(proxy_state["trained_updates"]),
        "train_count": len(observations),
        "test_count": len(temporal_test),
        "joint_epochs": int(args.epochs),
        "optimizer_note": (
            "optimizer state is not stored in the Stage2 checkpoint; sanity "
            "and dual both use fresh AdamW with identical inherited E0 weights"
        ),
        "models": {},
    }

    e0_models = [
        make_model(features.shape[1], member_state, args.device)
        for member_state in e0_states
    ]
    e0_prediction = predict_ensemble(
        e0_models,
        features,
        test_indices,
        target_mean,
        target_std,
        args.device,
    )
    results["models"]["E0_saved"] = {
        "raw": detailed_metrics(e0_prediction, temporal_test, anchors),
        "tie_aware": {
            str(value): tie_aware_metrics(
                e0_prediction, temporal_test, anchors, value
            )
            for value in args.tie_pct
        },
    }
    print("[E0] evaluated", flush=True)

    sanity_models = []
    sanity_histories = []
    for member, member_state in enumerate(e0_states):
        seed = int(cfg.seed) + 200_003 + member * 101
        seed_all(seed)
        model = make_model(features.shape[1], member_state, args.device)
        history = continue_training(
            model,
            features,
            target_z,
            groups,
            target_std,
            dual=False,
            tau_log=0.0,
            device=args.device,
            seed=seed,
            epochs=args.epochs,
            batch_groups=args.batch_groups,
        )
        sanity_models.append(model)
        sanity_histories.append(history)
        print(f"[sanity] member={member} complete", flush=True)
    sanity_prediction = predict_ensemble(
        sanity_models,
        features,
        test_indices,
        target_mean,
        target_std,
        args.device,
    )
    results["models"]["warm_sanity"] = {
        "history": sanity_histories,
        "raw": detailed_metrics(sanity_prediction, temporal_test, anchors),
        "tie_aware": {
            str(value): tie_aware_metrics(
                sanity_prediction, temporal_test, anchors, value
            )
            for value in args.tie_pct
        },
    }

    for tau_pct in args.tie_pct:
        dual_models = []
        histories = []
        for member, member_state in enumerate(e0_states):
            seed = int(cfg.seed) + 300_007 + member * 101 + int(tau_pct * 1000)
            seed_all(seed)
            model = make_model(features.shape[1], member_state, args.device)
            history = continue_training(
                model,
                features,
                target_z,
                groups,
                target_std,
                dual=True,
                tau_log=float(tau_pct) / 100.0,
                device=args.device,
                seed=seed,
                epochs=args.epochs,
                batch_groups=args.batch_groups,
            )
            dual_models.append(model)
            histories.append(history)
            print(
                f"[dual tau={tau_pct:g}%] member={member} complete",
                flush=True,
            )
        prediction = predict_ensemble(
            dual_models,
            features,
            test_indices,
            target_mean,
            target_std,
            args.device,
        )
        name = f"warm_dual_tau_{tau_pct:g}pct"
        results["models"][name] = {
            "history": histories,
            "raw": detailed_metrics(prediction, temporal_test, anchors),
            "tie_aware": {
                str(value): tie_aware_metrics(
                    prediction, temporal_test, anchors, value
                )
                for value in args.tie_pct
            },
        }
        torch.save(
            {
                "tau_pct": tau_pct,
                "models": [
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                    for model in dual_models
                ],
                "target_mean": target_mean,
                "target_std": target_std,
            },
            out_dir / f"{name}.pt",
        )
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
