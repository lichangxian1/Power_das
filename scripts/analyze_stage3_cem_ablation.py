#!/usr/bin/env python3
"""Equal-budget analysis for Stage-3 CEM, PPO, frozen, and random routing."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3

import numpy as np


METRICS = ("area", "power", "delay", "mred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cem", type=Path, required=True)
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--source-elites", type=Path, required=True)
    parser.add_argument("--elite-index", type=int, default=14)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--routes-per-episode", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_results(run_dir: Path, budget: int) -> list[dict]:
    cache_path = run_dir / "evaluation_cache.sqlite"
    with sqlite3.connect(cache_path) as conn:
        rows = conn.execute(
            "SELECT rowid, cache_key, result_json "
            "FROM evaluations ORDER BY rowid LIMIT ?",
            (budget,),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    if len(rows) != budget:
        raise RuntimeError(
            f"{run_dir}: need {budget} unique evaluated routes, "
            f"found {total}. Cross-episode cache hits require an explicit "
            "episode-result trace."
        )
    out = []
    for rowid, cache_key, raw in rows:
        item = json.loads(raw)
        if not item.get("valid", False):
            raise RuntimeError(f"{run_dir}: invalid cached row {rowid}")
        item["rowid"] = int(rowid)
        item["routing_hash"] = str(cache_key).split("|")[-1]
        out.append(item)
    return out


def reward(item: dict, baseline: dict, mred_limit: float) -> float:
    da = (baseline["area"] - item["area"]) / baseline["area"]
    dp = (baseline["power"] - item["power"]) / baseline["power"]
    dm = math.log(max(baseline["mred"], 1e-15) / max(item["mred"], 1e-15))
    timing_penalty = 5.0 * max(0.0, item["delay"] / 1.5 - 1.0)
    mred_penalty = max(
        0.0, math.log(max(item["mred"], 1e-15) / max(mred_limit, 1e-15))
    )
    return 0.35 * da + 0.35 * dp + 0.30 * dm - timing_penalty - mred_penalty


def dominates_baseline(item: dict, baseline: dict) -> bool:
    no_worse = all(item[key] <= baseline[key] for key in METRICS)
    strictly_better = any(item[key] < baseline[key] for key in METRICS)
    return bool(no_worse and strictly_better)


def summarize(
    name: str,
    items: list[dict],
    baseline: dict,
    mred_limit: float,
    episodes: int,
    routes_per_episode: int,
) -> dict:
    rewards = np.asarray(
        [reward(item, baseline, mred_limit) for item in items], dtype=float
    )
    reward_matrix = rewards.reshape(episodes, routes_per_episode)
    episode_mean = reward_matrix.mean(axis=1)
    episode_best = reward_matrix.max(axis=1)
    best_idx = int(rewards.argmax())
    best_item = items[best_idx]
    dominator_indices = [
        index
        for index, item in enumerate(items)
        if dominates_baseline(item, baseline)
    ]
    best_dominator = None
    if dominator_indices:
        best_dominator_idx = max(
            dominator_indices, key=lambda index: float(rewards[index])
        )
        dominator_item = items[best_dominator_idx]
        best_dominator = {
            "reward": float(rewards[best_dominator_idx]),
            "episode": best_dominator_idx // routes_per_episode + 1,
            "route_index": best_dominator_idx % routes_per_episode + 1,
            "rowid": dominator_item["rowid"],
            "routing_hash": dominator_item["routing_hash"],
            **{key: float(dominator_item[key]) for key in METRICS},
        }
    cumulative_best = np.maximum.accumulate(rewards)
    checkpoints = {
        str(ep): float(cumulative_best[ep * routes_per_episode - 1])
        for ep in range(10, episodes + 1, 10)
    }
    result = {
        "name": name,
        "evaluations": len(items),
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "positive_reward_fraction": float(np.mean(rewards > 0.0)),
        "baseline_dominators": int(len(dominator_indices)),
        "baseline_dominator_fraction": float(
            np.mean([dominates_baseline(item, baseline) for item in items])
        ),
        "best_reward": float(rewards[best_idx]),
        "best_episode": best_idx // routes_per_episode + 1,
        "best_route_index": best_idx % routes_per_episode + 1,
        "best_route": {
            "rowid": best_item["rowid"],
            "routing_hash": best_item["routing_hash"],
            **{key: float(best_item[key]) for key in METRICS},
        },
        "best_baseline_dominator": best_dominator,
        "episode_mean_reward": episode_mean.tolist(),
        "episode_best_reward": episode_best.tolist(),
        "mean_episode_best_reward": float(episode_best.mean()),
        "first10_mean_reward": float(episode_mean[:10].mean()),
        "last10_mean_reward": float(episode_mean[-10:].mean()),
        "first10_mean_episode_best": float(episode_best[:10].mean()),
        "last10_mean_episode_best": float(episode_best[-10:].mean()),
        "cumulative_best_reward": checkpoints,
        "mean_metrics": {
            key: float(np.mean([item[key] for item in items])) for key in METRICS
        },
    }
    return result


def bootstrap_difference(
    left: list[float],
    right: list[float],
    rng: np.random.Generator,
    samples: int,
) -> dict:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    chunk = 2_000
    differences = []
    remaining = samples
    while remaining:
        n = min(chunk, remaining)
        left_index = rng.integers(0, len(left_arr), size=(n, len(left_arr)))
        right_index = rng.integers(0, len(right_arr), size=(n, len(right_arr)))
        differences.append(
            left_arr[left_index].mean(axis=1)
            - right_arr[right_index].mean(axis=1)
        )
        remaining -= n
    diff = np.concatenate(differences)
    return {
        "difference": float(left_arr.mean() - right_arr.mean()),
        "ci95": [
            float(np.quantile(diff, 0.025)),
            float(np.quantile(diff, 0.975)),
        ],
        "probability_cem_better": float(np.mean(diff > 0.0)),
        "unit": "episode",
        "bootstrap_samples": int(samples),
    }


def markdown_report(report: dict) -> str:
    baseline = report["baseline"]
    lines = [
        "# Stage-3 DOMAC-CEM equal-budget comparison",
        "",
        (
            f"Budget: {report['episodes']} episodes × "
            f"{report['routes_per_episode']} routes = "
            f"{report['budget']} true DC+Verilator evaluations per method."
        ),
        "",
        (
            "Baseline e14: "
            f"area={baseline['area']:.6f}, "
            f"power={baseline['power'] * 1000:.6f} mW, "
            f"delay={baseline['delay']:.4f} ns, "
            f"MRED={baseline['mred']:.8g}."
        ),
        "",
        "| method | mean reward | best reward | best ep | positive | "
        "dominates e14 | last-10 ep best |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("cem", "ppo", "frozen", "random"):
        item = report["methods"][name]
        lines.append(
            f"| {name} | {item['mean_reward']:.6g} | "
            f"{item['best_reward']:.6g} | {item['best_episode']} | "
            f"{item['positive_reward_fraction']:.2%} | "
            f"{item['baseline_dominators']} | "
            f"{item['last10_mean_episode_best']:.6g} |"
        )
    lines.extend(["", "## CEM best route", ""])
    best = report["methods"]["cem"]["best_route"]
    lines.append(
        f"reward={report['methods']['cem']['best_reward']:.8g}, "
        f"area={best['area']:.6f}, power={best['power'] * 1000:.6f} mW, "
        f"delay={best['delay']:.4f} ns, MRED={best['mred']:.8g}, "
        f"routing_hash={best['routing_hash']}."
    )
    dominator = report["methods"]["cem"]["best_baseline_dominator"]
    if dominator is not None:
        lines.extend(["", "## CEM best route that dominates e14", ""])
        lines.append(
            f"reward={dominator['reward']:.8g}, "
            f"area={dominator['area']:.6f}, "
            f"power={dominator['power'] * 1000:.6f} mW, "
            f"delay={dominator['delay']:.4f} ns, "
            f"MRED={dominator['mred']:.8g}, "
            f"routing_hash={dominator['routing_hash']}."
        )
    lines.extend(["", "## Episode-level bootstrap: CEM minus baseline", ""])
    for name, tests in report["bootstrap_cem_minus"].items():
        mean_test = tests["mean_reward"]
        best_test = tests["episode_best_reward"]
        lines.append(
            f"- vs {name}: mean reward Δ={mean_test['difference']:.6g}, "
            f"95% CI [{mean_test['ci95'][0]:.6g}, "
            f"{mean_test['ci95'][1]:.6g}]; episode-best "
            f"Δ={best_test['difference']:.6g}, 95% CI "
            f"[{best_test['ci95'][0]:.6g}, {best_test['ci95'][1]:.6g}]."
        )
    lines.extend(["", "## Cumulative best reward", ""])
    header = "| episode | " + " | ".join(("cem", "ppo", "frozen", "random")) + " |"
    lines.extend([header, "|---:|---:|---:|---:|---:|"])
    for episode in range(10, report["episodes"] + 1, 10):
        values = [
            report["methods"][name]["cumulative_best_reward"][str(episode)]
            for name in ("cem", "ppo", "frozen", "random")
        ]
        lines.append(
            f"| {episode} | " + " | ".join(f"{value:.6g}" for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.routes_per_episode < 1:
        raise SystemExit("episodes and routes-per-episode must be positive")
    with args.source_elites.open() as handle:
        source = json.load(handle)
    elite = source[args.elite_index]
    baseline = {key: float(elite[key]) for key in METRICS}
    band_edges = (elite.get("metadata") or {}).get("mred_band_edges")
    mred_limit = float(band_edges[1] if band_edges else 0.2)
    budget = args.episodes * args.routes_per_episode
    run_dirs = {
        "cem": args.cem,
        "ppo": args.ppo,
        "frozen": args.frozen,
        "random": args.random,
    }
    methods = {
        name: summarize(
            name,
            load_results(run_dir, budget),
            baseline,
            mred_limit,
            args.episodes,
            args.routes_per_episode,
        )
        for name, run_dir in run_dirs.items()
    }
    rng = np.random.default_rng(args.seed)
    bootstrap = {}
    for name in ("ppo", "frozen", "random"):
        bootstrap[name] = {
            "mean_reward": bootstrap_difference(
                methods["cem"]["episode_mean_reward"],
                methods[name]["episode_mean_reward"],
                rng,
                args.bootstrap,
            ),
            "episode_best_reward": bootstrap_difference(
                methods["cem"]["episode_best_reward"],
                methods[name]["episode_best_reward"],
                rng,
                args.bootstrap,
            ),
        }
    report = {
        "budget": budget,
        "episodes": args.episodes,
        "routes_per_episode": args.routes_per_episode,
        "objective": {
            "weights": {"area": 0.35, "power": 0.35, "log_mred": 0.30},
            "delay_limit_ns": 1.5,
            "mred_limit": mred_limit,
        },
        "baseline": baseline,
        "run_dirs": {name: str(path.resolve()) for name, path in run_dirs.items()},
        "methods": methods,
        "bootstrap_cem_minus": bootstrap,
    }
    output = args.output or (args.cem / "equal_budget_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(markdown_report(report))
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
