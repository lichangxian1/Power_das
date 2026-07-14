#!/usr/bin/env python3
"""Evaluate structure and functional-activity proxies on fixed-vector XA labels."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/power_das_matplotlib")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def group_name(design: str) -> str:
    match = re.match(r"(k\d+_b[^_]+)", design)
    return match.group(1) if match else design


def logo_predict(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    groups = df["group"].unique()
    for group in groups:
        test = (df["group"] == group).to_numpy()
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 30)),
        )
        model.fit(df.loc[train, features], df.loc[train, "power_xa_mw"])
        pred[test] = model.predict(df.loc[test, features])
    return pred


def pair_metrics(df: pd.DataFrame, column: str, noise_fraction: float) -> dict:
    measured, predicted, above = [], [], []
    rows = []
    for group, frame in df.groupby("group"):
        if len(frame) < 2:
            continue
        for (_, left), (_, right) in itertools.combinations(frame.iterrows(), 2):
            dy = float(left.power_xa_mw - right.power_xa_mw)
            dp = float(left[column] - right[column])
            floor = noise_fraction * 0.5 * float(left.power_xa_mw + right.power_xa_mw)
            measured.append(dy)
            predicted.append(dp)
            above.append(abs(dy) > floor)
            rows.append({
                "group": group,
                "left": left.design,
                "right": right.design,
                "delta_xa_mw": dy,
                f"delta_{column}": dp,
                "above_floor": abs(dy) > floor,
                "sign_correct": np.sign(dy) == np.sign(dp) and dp != 0,
            })
    measured = np.asarray(measured)
    predicted = np.asarray(predicted)
    above = np.asarray(above, dtype=bool)
    signed = above & (predicted != 0)
    correct = np.sign(measured[signed]) == np.sign(predicted[signed])
    n_signed = int(signed.sum())
    n_correct = int(correct.sum())
    if n_signed:
        # Wilson score interval is much more informative than a bare percentage
        # for the small, group-local comparison sets used in this experiment.
        z = 1.959963984540054
        phat = n_correct / n_signed
        denom = 1.0 + z * z / n_signed
        center = (phat + z * z / (2.0 * n_signed)) / denom
        half = z * np.sqrt(
            phat * (1.0 - phat) / n_signed + z * z / (4.0 * n_signed**2)
        ) / denom
        confidence_interval = [float(center - half), float(center + half)]
    else:
        confidence_interval = None
    return {
        "n_pairs": int(len(measured)),
        "n_above_floor": int(above.sum()),
        "n_scored": n_signed,
        "n_correct": n_correct,
        "spearman_all": float(spearmanr(measured, predicted).statistic),
        "sign_accuracy_above_floor": float(correct.mean()) if n_signed else None,
        "sign_accuracy_wilson95": confidence_interval,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xa", type=Path, required=True)
    ap.add_argument("--activity", type=Path, required=True)
    ap.add_argument("--dc", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--noise-fraction", type=float, default=0.01)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    xa = pd.read_csv(args.xa)
    activity = pd.read_csv(args.activity).drop(columns=["rtl_path"])
    df = xa.merge(activity, on="design", how="inner")
    df = df[df["success"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    df["delay"] = df["delay"].abs()
    df["group"] = df.design.map(group_name)
    if args.dc:
        dc = pd.read_csv(args.dc)
        dc = dc[dc["success"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        df = df.merge(dc[["design", "power_dc_mw"]], on="design", how="inner")

    feature_sets = {
        "activity_only": [
            "n_cells", "n_ct22", "n_ct32", "n_ct42",
            "functional_toggle_total", "functional_toggle_mean",
            "functional_toggle_22", "functional_toggle_32", "functional_toggle_42",
            "functional_toggle_sum_port", "functional_toggle_carry_ports",
            "functional_toggle_col_low", "functional_toggle_col_mid",
            "functional_toggle_col_high", "functional_toggle_stage_early",
            "functional_toggle_stage_middle", "functional_toggle_stage_late",
        ],
        "area_only": ["area_dc"],
        "structure": ["area_dc", "delay", "n_cells", "n_ct22", "n_ct32", "n_ct42"],
        "structure_activity": [
            "area_dc", "delay", "n_cells", "n_ct22", "n_ct32", "n_ct42",
            "functional_toggle_total", "functional_toggle_mean",
            "functional_toggle_22", "functional_toggle_32", "functional_toggle_42",
            "functional_toggle_sum_port", "functional_toggle_carry_ports",
            "fanout_weighted_toggle", "functional_toggle_col_low",
            "functional_toggle_col_mid", "functional_toggle_col_high",
            "functional_toggle_stage_early", "functional_toggle_stage_middle",
            "functional_toggle_stage_late",
            "activity_scaled_dyn_mw",
        ],
    }
    if "power_dc_mw" in df:
        feature_sets["dc_calibrated"] = ["power_dc_mw", "area_dc", "delay"]
        feature_sets["dc_activity"] = [
            "power_dc_mw", "area_dc", "delay", "n_cells", "n_ct22", "n_ct32", "n_ct42",
            "functional_toggle_total", "functional_toggle_mean",
            "functional_toggle_22", "functional_toggle_32", "functional_toggle_42",
            "functional_toggle_sum_port", "functional_toggle_carry_ports",
            "fanout_weighted_toggle", "functional_toggle_col_low",
            "functional_toggle_col_mid", "functional_toggle_col_high",
            "functional_toggle_stage_early", "functional_toggle_stage_middle",
            "functional_toggle_stage_late",
            "activity_scaled_dyn_mw",
        ]
    metrics = {
        "n_designs": len(df),
        "n_groups": int(df.group.nunique()),
        "noise_fraction": args.noise_fraction,
        "raw": {},
        "logo_ridge": {},
    }
    raw_columns = ["area_dc", "functional_toggle_total", "activity_scaled_dyn_mw"]
    if "power_dc_mw" in df:
        raw_columns.append("power_dc_mw")
    for column in raw_columns:
        metrics["raw"][column] = {
            "global_spearman": float(spearmanr(df.power_xa_mw, df[column]).statistic),
            "pairwise": pair_metrics(df, column, args.noise_fraction),
        }
    for label, features in feature_sets.items():
        column = f"pred_{label}"
        df[column] = logo_predict(df, features)
        metrics["logo_ridge"][label] = {
            "features": features,
            "global_spearman": float(spearmanr(df.power_xa_mw, df[column]).statistic),
            "mape_percent": float(np.mean(np.abs(df[column] - df.power_xa_mw) / df.power_xa_mw) * 100),
            "pairwise": pair_metrics(df, column, args.noise_fraction),
        }
        final_model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30))
        )
        final_model.fit(df[features], df["power_xa_mw"])
        joblib.dump(
            {"model": final_model, "features": features, "label": "power_xa_mw"},
            args.output / f"ridge_{label}.joblib",
        )

    def strip_rows(obj):
        if isinstance(obj, dict):
            return {k: strip_rows(v) for k, v in obj.items() if k != "rows"}
        return obj

    (args.output / "metrics.json").write_text(
        json.dumps(strip_rows(metrics), indent=2, sort_keys=True) + "\n"
    )
    df.to_csv(args.output / "predictions.csv", index=False)
    pair_rows = []
    for source, values in metrics["raw"].items():
        pair_rows.extend({"proxy": source, **row} for row in values["pairwise"]["rows"])
    for source, values in metrics["logo_ridge"].items():
        pair_rows.extend({"proxy": source, **row} for row in values["pairwise"]["rows"])
    pd.DataFrame(pair_rows).to_csv(args.output / "pairs.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(df.power_xa_mw, df.pred_structure, label="structure", alpha=0.8)
    axes[0].scatter(df.power_xa_mw, df.pred_structure_activity, label="+ activity", alpha=0.8)
    lo, hi = df.power_xa_mw.min(), df.power_xa_mw.max()
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="Fixed-vector XA power (mW)", ylabel="LOGO prediction (mW)")
    axes[0].legend()
    raw = metrics["raw"]
    names = ["area", "toggle", "activity dyn"]
    raw_keys = ["area_dc", "functional_toggle_total", "activity_scaled_dyn_mw"]
    if "power_dc_mw" in raw:
        names.append("DC power")
        raw_keys.append("power_dc_mw")
    acc = [
        raw[key]["pairwise"]["sign_accuracy_above_floor"]
        for key in raw_keys
    ]
    model_keys = ["activity_only", "structure", "structure_activity"]
    if "dc_calibrated" in metrics["logo_ridge"]:
        model_keys.extend(["dc_calibrated", "dc_activity"])
    model_acc = [
        metrics["logo_ridge"][key]["pairwise"]["sign_accuracy_above_floor"]
        for key in model_keys
    ]
    model_names = ["ridge activity", "ridge struct", "ridge + act"]
    if len(model_keys) > 2:
        model_names.extend(["ridge DC", "ridge DC+act"])
    axes[1].bar(names + model_names, acc + model_acc)
    axes[1].axhline(0.70, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Pairwise direction accuracy", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(args.output / "proxy_vs_fixed_xa.png", dpi=180)
    fig.savefig(args.output / "proxy_vs_fixed_xa.pdf")
    print(json.dumps(strip_rows(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
