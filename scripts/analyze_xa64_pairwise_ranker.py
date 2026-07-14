#!/usr/bin/env python3
"""Nested-LOGO pairwise ranking models for XA64 within-stratum power order."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/power_das_matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "area_toggle": ["area_dc", "functional_toggle_total"],
    "area_dc": ["area_dc", "power_dc_mw"],
    "toggle_dc": ["functional_toggle_total", "power_dc_mw"],
    "area_toggle_dc": ["area_dc", "functional_toggle_total", "power_dc_mw"],
    "compact_activity": [
        "area_dc", "power_dc_mw", "functional_toggle_total",
        "functional_toggle_sum_port", "functional_toggle_carry_ports",
        "functional_toggle_col_low", "functional_toggle_col_mid",
        "functional_toggle_col_high",
    ],
}


def wilson(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [float(center - half), float(center + half)]


def make_pairs(frame: pd.DataFrame, features: list[str], floor: float) -> pd.DataFrame:
    rows = []
    for group, group_frame in frame.groupby("group"):
        for (_, left), (_, right) in itertools.combinations(group_frame.iterrows(), 2):
            delta_y = float(left.power_xa_mw - right.power_xa_mw)
            threshold = floor * 0.5 * float(left.power_xa_mw + right.power_xa_mw)
            if abs(delta_y) <= threshold:
                continue
            row = {"group": group, "left": left.design, "right": right.design,
                   "delta_xa_mw": delta_y, "label": int(delta_y > 0)}
            for feature in features:
                row[feature] = float(left[feature] - right[feature])
            rows.append(row)
    return pd.DataFrame(rows)


def fit_model(train: pd.DataFrame, features: list[str], c_value: float):
    x = train[features].to_numpy(dtype=float)
    y = train.label.to_numpy(dtype=int)
    # Mirroring every comparison prevents arbitrary left/right ordering from
    # creating a class imbalance or intercept shortcut.
    x = np.concatenate([x, -x], axis=0)
    y = np.concatenate([y, 1 - y], axis=0)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, fit_intercept=False, max_iter=3000),
    )
    model.fit(x, y)
    return model


def choose_c(train: pd.DataFrame, features: list[str]) -> float:
    candidates = [0.01, 0.1, 1.0, 10.0]
    groups = sorted(train.group.unique())
    scores = {}
    for c_value in candidates:
        correct = total = 0
        for group in groups:
            inner_train = train[train.group != group]
            inner_test = train[train.group == group]
            model = fit_model(inner_train, features, c_value)
            pred = model.predict(inner_test[features].to_numpy(dtype=float))
            correct += int((pred == inner_test.label.to_numpy()).sum())
            total += len(inner_test)
        scores[c_value] = correct / total
    return max(candidates, key=lambda value: (scores[value], -value))


def evaluate_nested_logo(pairs: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    outputs = []
    chosen = {}
    for group in sorted(pairs.group.unique()):
        train = pairs[pairs.group != group]
        test = pairs[pairs.group == group].copy()
        c_value = choose_c(train, features)
        chosen[group] = c_value
        model = fit_model(train, features, c_value)
        test["prediction"] = model.predict(test[features].to_numpy(dtype=float))
        test["correct"] = test.prediction == test.label
        outputs.append(test)
    output = pd.concat(outputs, ignore_index=True)
    correct = int(output.correct.sum())
    return {
        "n_pairs": int(len(output)),
        "n_correct": correct,
        "accuracy": correct / len(output),
        "wilson95": wilson(correct, len(output)),
        "chosen_c": chosen,
        "per_group_accuracy": output.groupby("group").correct.mean().to_dict(),
    }, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xa", type=Path, required=True)
    parser.add_argument("--dc", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-fraction", type=float, default=0.01)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    xa = pd.read_csv(args.xa)
    dc = pd.read_csv(args.dc)
    activity = pd.read_csv(args.activity).drop(columns=["rtl_path"], errors="ignore")
    frame = xa.merge(dc[["design", "power_dc_mw"]], on="design").merge(activity, on="design")
    frame["group"] = frame.design.str.rsplit("_s", n=1).str[0]

    metrics = {"n_designs": len(frame), "n_groups": frame.group.nunique(), "models": {}}
    all_predictions = []
    for name, features in FEATURE_SETS.items():
        pairs = make_pairs(frame, features, args.noise_fraction)
        result, predictions = evaluate_nested_logo(pairs, features)
        metrics["models"][name] = {"features": features, **result}
        predictions.insert(0, "model", name)
        all_predictions.append(predictions)

    base_pairs = make_pairs(frame, ["area_dc", "functional_toggle_total", "power_dc_mw"], args.noise_fraction)
    signs = np.sign(base_pairs[["area_dc", "functional_toggle_total", "power_dc_mw"]].to_numpy())
    labels = base_pairs.label.to_numpy()
    for name, vote in {
        "vote_area_toggle": np.sign(signs[:, 0] + signs[:, 1]),
        "vote_area_toggle_dc": np.sign(signs.sum(axis=1)),
    }.items():
        valid = vote != 0
        prediction = (vote[valid] > 0).astype(int)
        correct = int((prediction == labels[valid]).sum())
        metrics["models"][name] = {
            "n_pairs": int(valid.sum()), "n_correct": correct,
            "accuracy": correct / int(valid.sum()), "wilson95": wilson(correct, int(valid.sum())),
        }

    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    pd.concat(all_predictions, ignore_index=True).to_csv(args.output / "pair_predictions.csv", index=False)
    frozen = json.loads(
        (args.output.parent / "frozen_model_analysis" / "metrics.json").read_text()
    )
    raw_accuracy = {}
    for proxy in ("power_dc_mw", "functional_toggle_total", "area_dc"):
        pairwise = frozen["models"][proxy]["pairwise"]
        raw_accuracy[proxy] = (
            float(pairwise["direction_accuracy"]), int(pairwise["n_above_floor"])
        )
    plot_items = [
        ("DC", *raw_accuracy["power_dc_mw"]),
        ("toggle", *raw_accuracy["functional_toggle_total"]),
        ("area", *raw_accuracy["area_dc"]),
        ("compact\nranker", metrics["models"]["compact_activity"]["accuracy"],
         metrics["models"]["compact_activity"]["n_pairs"]),
        ("area+toggle\nconsensus", metrics["models"]["vote_area_toggle"]["accuracy"],
         metrics["models"]["vote_area_toggle"]["n_pairs"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.bar([item[0] for item in plot_items], [item[1] for item in plot_items])
    ax.axhline(0.85, color="tab:red", linestyle="--", linewidth=1)
    ax.set(ylabel="Direction accuracy", ylim=(0, 1.03))
    for bar, (_, accuracy, count) in zip(bars, plot_items):
        ax.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.018,
                f"{accuracy:.1%}\nn={count}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / "pairwise_ranker_comparison.png", dpi=180)
    fig.savefig(args.output / "pairwise_ranker_comparison.pdf")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
