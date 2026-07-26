#!/usr/bin/env python3
"""Render the routing optimizer comparison as a per-episode PNG."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SEED42_SERIES = (
    ("原版 PPO", "#4ea1ff", "2026-07-25_0232_ppo_360ep_32dc"),
    ("CEM", "#ffb224", "2026-07-25_0232_cem_360ep_32dc"),
    ("Active-block PPO", "#35d39a", "2026-07-25_0333_active_ppo_b32_360ep_32dc"),
)
SEED43_SERIES = (
    ("原版 PPO", "#4ea1ff", "2026-07-25_1144_ppo_seed43_360ep_32dc"),
    ("CEM", "#ffb224", "2026-07-25_1144_cem_seed43_360ep_32dc"),
    ("Active-block PPO", "#35d39a", "2026-07-25_1144_active_ppo_b32_seed43_360ep_32dc"),
)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode" in record and "batch_objective_min" in record:
                records.append(record)
    return sorted(records, key=lambda item: item["episode"])


def aggregate_records(
    records: list[dict], granularity: int = 1
) -> tuple[list[float], list[float]]:
    x_values, y_values = [], []
    for start in range(0, len(records) - granularity + 1, granularity):
        group = records[start : start + granularity]
        x_values.append(
            sum(float(item["episode"]) for item in group) / float(granularity)
        )
        y_values.append(
            sum(float(item["batch_objective_min"]) for item in group)
            / float(granularity)
        )
    return x_values, y_values


def render(
    experiment_root: Path,
    output: Path,
    seed43_root: Path | None,
    seed: int = 42,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "WenQuanYi Micro Hei",
                "Droid Sans Fallback",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.weight": "normal",
        }
    )

    background = "#07111f"
    panel = "#0d1b2a"
    foreground = "#e8eef7"
    muted = "#91a4ba"
    grid = "#263a50"

    fig = plt.figure(figsize=(18, 10), dpi=100, facecolor=background)
    ax = fig.add_axes((0.075, 0.14, 0.86, 0.62), facecolor=panel)

    plotted = []
    progress = {}
    if seed == 43:
        groups = [("s43", False, experiment_root, SEED43_SERIES)]
        chart_title = "布线优化器对比 · Seed 43"
    else:
        groups = [("s42", False, experiment_root, SEED42_SERIES)]
        chart_title = "布线优化器对比 · Seed 42"
        if seed43_root is not None:
            groups.append(("s43", True, seed43_root, SEED43_SERIES))
            chart_title = "布线优化器对比 · Seed 42 vs Seed 43"
    for seed, dashed, root, series in groups:
        for label, color, directory in series:
            metrics = root / directory / "routing_metrics.jsonl"
            records = load_records(metrics) if metrics.is_file() else []
            progress[(seed, label)] = len(records)
            x_values, y_values = aggregate_records(records, granularity=1)
            if not y_values:
                continue
            plotted.append((seed, dashed, label, color, records, x_values, y_values))
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=2.7,
                linestyle="--" if dashed else "-",
                marker="s" if dashed else "o",
                markersize=5.5,
                markerfacecolor=panel,
                markeredgecolor=color,
                markeredgewidth=1.8,
                solid_capstyle="round",
                zorder=3,
            )

    if not plotted:
        raise SystemExit("No records were found")

    all_y = [value for _, _, _, _, _, _, values in plotted for value in values]
    spread = max(all_y) - min(all_y)
    padding = max(spread * 0.14, 0.003)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_xlim(0, 365)
    ax.set_xticks(range(0, 361, 40))
    ax.grid(True, color=grid, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, labelsize=12, length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_color(grid)
        spine.set_linewidth(1.0)
    ax.set_xlabel("Episode（每 1 轮一个点）", color=muted, fontsize=13, labelpad=17)
    ax.set_ylabel("Per-batch best objective  ↓", color=muted, fontsize=13, labelpad=17)

    fig.text(
        0.075,
        0.925,
        chart_title,
        color=foreground,
        fontsize=27,
        fontweight="normal",
        ha="left",
        va="center",
    )
    fig.text(
        0.075,
        0.890,
        "每个点为 1 个 episode（batch）内约 32 个候选的最优 objective · 越低越好",
        color=muted,
        fontsize=13,
        ha="left",
        va="center",
    )
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
    status_lines = []
    for seed_key, _, _, series in groups:
        status = " · ".join(
            f"{label} {progress.get((seed_key, label), 0)}/360"
            for label, _, _ in series
        )
        status_lines.append(f"Seed {seed_key[1:]}：{status}")
    fig.text(
        0.935,
        0.925,
        f"生成于 {generated_at}\n" + "\n".join(status_lines),
        color=muted,
        fontsize=11,
        ha="right",
        va="center",
        linespacing=1.6,
    )

    legend_handles = []
    for seed, dashed, label, color, records, x_values, y_values in plotted:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.7,
                linestyle="--" if dashed else "-",
                marker="s" if dashed else "o",
                markersize=6,
                markerfacecolor=panel,
                label=(
                    f"[{seed}] {label}  {len(records)}/360  "
                    f"最新 {y_values[-1]:.5f}  最佳点 {min(y_values):.5f}"
                ),
            )
        )
        ax.annotate(
            f"{y_values[-1]:.5f}",
            xy=(x_values[-1], y_values[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": panel, "edgecolor": color, "alpha": 0.95},
            zorder=5,
        )

    legend = fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.855),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        handlelength=2.8,
        columnspacing=2.0,
        handletextpad=0.8,
    )
    for text in legend.get_texts():
        text.set_color(foreground)

    fig.text(
        0.075,
        0.065,
        "口径：每个点为对应 episode（batch）内约 32 条布线的最小 objective；不跨 episode 累计历史最优。",
        color=muted,
        fontsize=11,
        ha="left",
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed43-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    args = parser.parse_args()
    render(args.experiment_root, args.output, args.seed43_root, seed=args.seed)


if __name__ == "__main__":
    main()
