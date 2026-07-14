#!/usr/bin/env python3
"""Evaluate frozen 28-design proxy checkpoints on the independent XA64 set."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/power_das_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [float(center - half), float(center + half)]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    ape = np.abs(error) / actual
    return {
        "spearman": float(spearmanr(actual, predicted).statistic),
        "pearson": float(pearsonr(actual, predicted).statistic),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(100 * np.mean(ape)),
        "median_ape_percent": float(100 * np.median(ape)),
        "max_ape_percent": float(100 * np.max(ape)),
    }


def pair_metrics(frame: pd.DataFrame, column: str, floor_fraction: float) -> tuple[dict, list[dict]]:
    rows = []
    for group, group_frame in frame.groupby("group"):
        for (_, left), (_, right) in itertools.combinations(group_frame.iterrows(), 2):
            delta_xa = float(left.power_xa_mw - right.power_xa_mw)
            delta_pred = float(left[column] - right[column])
            floor = floor_fraction * 0.5 * float(left.power_xa_mw + right.power_xa_mw)
            above = abs(delta_xa) > floor
            correct = above and delta_pred != 0 and np.sign(delta_xa) == np.sign(delta_pred)
            rows.append({
                "proxy": column,
                "group": group,
                "left": left.design,
                "right": right.design,
                "delta_xa_mw": delta_xa,
                "delta_prediction": delta_pred,
                "above_floor": above,
                "sign_correct": correct,
            })
    scored = [row for row in rows if row["above_floor"] and row["delta_prediction"] != 0]
    correct = sum(bool(row["sign_correct"]) for row in scored)
    return {
        "n_pairs": len(rows),
        "n_above_floor": len(scored),
        "n_correct": correct,
        "direction_accuracy": correct / len(scored) if scored else None,
        "direction_accuracy_wilson95": wilson(correct, len(scored)),
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xa", type=Path, required=True)
    parser.add_argument("--dc", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-fraction", type=float, default=0.01)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    xa = pd.read_csv(args.xa)
    dc = pd.read_csv(args.dc)
    activity = pd.read_csv(args.activity).drop(columns=["rtl_path"], errors="ignore")
    frame = xa.merge(dc[["design", "power_dc_mw"]], on="design", how="inner")
    frame = frame.merge(activity, on="design", how="inner")
    frame = frame[
        frame["success"].astype(str).str.lower().isin(["true", "1", "1.0"])
    ].copy()
    frame["delay"] = frame["delay"].abs()
    frame["group"] = frame.design.str.rsplit("_s", n=1).str[0]

    checkpoint_names = [
        "activity_only", "area_only", "structure", "structure_activity",
        "dc_calibrated", "dc_activity",
    ]
    metrics = {
        "n_designs": int(len(frame)),
        "n_groups": int(frame.group.nunique()),
        "noise_fraction": args.noise_fraction,
        "models_are_frozen_from": str(args.checkpoints),
        "models": {},
    }
    pair_rows = []
    columns = ["power_dc_mw", "functional_toggle_total", "area_dc"]
    for name in checkpoint_names:
        payload = joblib.load(args.checkpoints / f"ridge_{name}.joblib")
        column = f"pred_{name}"
        frame[column] = payload["model"].predict(frame[payload["features"]])
        columns.append(column)

    actual = frame.power_xa_mw.to_numpy(dtype=float)
    for column in columns:
        predicted = frame[column].to_numpy(dtype=float)
        # Raw DC/toggle/area have different units; absolute errors are meaningful
        # only for frozen models trained to output XA mW.
        absolute = None if column in {"power_dc_mw", "functional_toggle_total", "area_dc"} else regression_metrics(actual, predicted)
        if absolute is not None:
            absolute["n_negative_predictions"] = int((predicted < 0).sum())
        pairwise, rows = pair_metrics(frame, column, args.noise_fraction)
        metrics["models"][column] = {
            "global_spearman": float(spearmanr(actual, predicted).statistic),
            "absolute": absolute,
            "pairwise": pairwise,
        }
        pair_rows.extend(rows)

    frame.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output / "pairs.csv", index=False)
    stratum_rows = []
    for group, group_frame in frame.groupby("group"):
        row = {
            "group": group,
            "n_designs": len(group_frame),
            "xa_min_mw": group_frame.power_xa_mw.min(),
            "xa_mean_mw": group_frame.power_xa_mw.mean(),
            "xa_max_mw": group_frame.power_xa_mw.max(),
        }
        for column in columns:
            pairwise, _ = pair_metrics(group_frame, column, args.noise_fraction)
            row[f"{column}_pair_accuracy"] = pairwise["direction_accuracy"]
            row[f"{column}_n_scored_pairs"] = pairwise["n_above_floor"]
            if column.startswith("pred_"):
                row[f"{column}_mape_percent"] = float(
                    100 * np.mean(
                        np.abs(group_frame[column] - group_frame.power_xa_mw)
                        / group_frame.power_xa_mw
                    )
                )
        stratum_rows.append(row)
    pd.DataFrame(stratum_rows).to_csv(args.output / "stratum_metrics.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for column, label in [
        ("pred_activity_only", "activity only"),
        ("pred_dc_calibrated", "DC calibrated"),
        ("pred_dc_activity", "DC + activity"),
    ]:
        axes[0].scatter(actual, frame[column], label=label, alpha=0.75)
    lo = min(actual.min(), frame[["pred_activity_only", "pred_dc_calibrated", "pred_dc_activity"]].min().min())
    hi = max(actual.max(), frame[["pred_activity_only", "pred_dc_calibrated", "pred_dc_activity"]].max().max())
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="XA power (mW)", ylabel="Frozen prediction (mW)")
    axes[0].legend()

    order = ["power_dc_mw", "functional_toggle_total", "area_dc",
             "pred_activity_only", "pred_dc_calibrated", "pred_dc_activity"]
    labels = ["DC", "toggle", "area", "activity model", "DC model", "DC+activity"]
    accuracies = [
        metrics["models"][name]["pairwise"]["direction_accuracy"]
        if metrics["models"][name]["pairwise"]["direction_accuracy"] is not None
        else np.nan
        for name in order
    ]
    axes[1].bar(labels, accuracies)
    axes[1].axhline(0.85, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Within-stratum direction accuracy", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", rotation=25)

    model_order = ["pred_activity_only", "pred_area_only", "pred_structure",
                   "pred_structure_activity", "pred_dc_calibrated", "pred_dc_activity"]
    model_labels = ["activity", "area", "structure", "struct+act", "DC", "DC+act"]
    mapes = [metrics["models"][name]["absolute"]["mape_percent"] for name in model_order]
    axes[2].bar(model_labels, mapes)
    axes[2].set(ylabel="Frozen-model MAPE (%)")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(args.output / "xa64_prediction_comparison.png", dpi=180)
    fig.savefig(args.output / "xa64_prediction_comparison.pdf")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
