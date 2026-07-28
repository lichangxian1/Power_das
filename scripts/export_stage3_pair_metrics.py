#!/usr/bin/env python3
"""Export live, episode-level Stage-3 routing objectives."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo


ROLE_WEIGHTS = {
    "area": (0.60, 0.20, 0.20),
    "power": (0.20, 0.60, 0.20),
    "knee": (0.35, 0.35, 0.30),
}
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def reward(item: dict, baseline: dict, role: str, delay_limit: float) -> float:
    wa, wp, wm = ROLE_WEIGHTS[role]
    da = (baseline["area"] - item["area"]) / baseline["area"]
    dp = (baseline["power"] - item["power"]) / baseline["power"]
    dm = math.log(max(baseline["mred"], 1e-15) / max(item["mred"], 1e-15))
    timing_penalty = 5.0 * max(0.0, item["delay"] / delay_limit - 1.0)
    mred_penalty = max(
        0.0, math.log(max(item["mred"], 1e-15) / max(baseline["mred_hi"], 1e-15))
    )
    return wa * da + wp * dp + wm * dm - timing_penalty - mred_penalty


def load_rows(cache_path: Path) -> list[dict]:
    if not cache_path.is_file():
        return []
    uri = f"file:{cache_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10.0) as connection:
        rows = connection.execute(
            "SELECT result_json FROM evaluations ORDER BY rowid"
        ).fetchall()
    return [json.loads(raw) for (raw,) in rows]


def summarize_arm(
    name: str,
    run_dir: Path,
    baseline: dict,
    role: str,
    routes_per_episode: int,
    total_episodes: int,
    delay_limit: float,
) -> dict:
    rows = load_rows(run_dir / "evaluation_cache.sqlite")
    complete = min(total_episodes, len(rows) // routes_per_episode)
    records = []
    cumulative_best = math.inf
    best_route = None
    for episode_index in range(complete):
        group = rows[
            episode_index * routes_per_episode :
            (episode_index + 1) * routes_per_episode
        ]
        objectives = [-reward(item, baseline, role, delay_limit) for item in group]
        best_index = min(range(len(group)), key=lambda index: objectives[index])
        if objectives[best_index] < cumulative_best:
            cumulative_best = objectives[best_index]
            best_route = {
                key: float(group[best_index][key])
                for key in ("area", "power", "delay", "mred")
            }
            best_route.update(
                {"objective": float(cumulative_best), "episode": episode_index + 1}
            )
        records.append(
            {
                "episode": episode_index + 1,
                "mean_objective": float(sum(objectives) / len(objectives)),
                "batch_best_objective": float(min(objectives)),
                "cumulative_best_objective": float(cumulative_best),
            }
        )
    return {
        "name": name,
        "run_dir": str(run_dir.resolve()),
        "completed_episodes": complete,
        "total_episodes": total_episodes,
        "cache_rows": len(rows),
        "incomplete_episode_rows": len(rows) - complete * routes_per_episode,
        "records": records,
        "best_route": best_route,
    }


def summarize_logged_arm(
    name: str,
    run_dir: Path,
    total_episodes: int,
) -> dict:
    """Read generation metrics written by optimizers whose cache may deduplicate."""
    path = run_dir / "routing_metrics.jsonl"
    records = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "episode": int(item.get("episode", item["generation"])),
                    "mean_objective": float(item["mean_objective"]),
                    "batch_best_objective": float(item["batch_best_objective"]),
                    "cumulative_best_objective": float(
                        item["cumulative_best_objective"]
                    ),
                }
            )
    records = records[:total_episodes]
    return {
        "name": name,
        "run_dir": str(run_dir.resolve()),
        "completed_episodes": len(records),
        "total_episodes": total_episodes,
        "cache_rows": len(load_rows(run_dir / "evaluation_cache.sqlite")),
        "incomplete_episode_rows": 0,
        "records": records,
        "best_route": None,
    }


def load_ppo_log_means(run_dir: Path, total_episodes: int) -> dict[int, float]:
    """Recover unnormalised one-epoch PPO mean objectives from the run log."""
    path = run_dir / "launcher.log"
    if not path.is_file():
        path = run_dir / "train_three_stage.log"
    if not path.is_file():
        return {}
    pattern = re.compile(
        r"Stage 3 (\d+)/\d+ .* losses=([-+0-9.eE]+)(?:,[-+0-9.eE]+)*$"
    )
    means = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        episode = int(match.group(1))
        if 1 <= episode <= total_episodes:
            means[episode] = float(match.group(2))
    return means


def summarize_vanilla_arm(
    run_dir: Path,
    baseline: dict,
    role: str,
    routes_per_episode: int,
    total_episodes: int,
    delay_limit: float,
) -> dict:
    """Restore Vanilla PPO despite evaluation-cache route deduplication.

    This run used one PPO epoch, no auxiliary losses, and unnormalised
    advantages, so its logged PPO loss is the batch mean objective. Exact
    cache-derived means are retained until they diverge from the log; the log
    supplies the remaining means. Cumulative best is extended using discovery
    episodes stored on final-Pareto candidates.
    """
    cached = summarize_arm(
        "Vanilla PPO",
        run_dir,
        baseline,
        role,
        routes_per_episode,
        total_episodes,
        delay_limit,
    )
    logged_means = load_ppo_log_means(run_dir, total_episodes)
    if not logged_means:
        return cached

    exact_prefix = 0
    for record in cached["records"]:
        episode = int(record["episode"])
        logged = logged_means.get(episode)
        if logged is None or abs(float(record["mean_objective"]) - logged) > 2.0e-5:
            break
        exact_prefix = episode

    pareto_path = run_dir / "final" / "pareto.json"
    pareto = []
    if pareto_path.is_file():
        pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    pareto_milestones = []
    best_route = cached.get("best_route")
    for item in pareto:
        metadata = item.get("metadata") or {}
        episode = int(metadata.get("stage3_episode") or 0)
        if not 1 <= episode <= total_episodes:
            continue
        objective = -reward(item, baseline, role, delay_limit)
        pareto_milestones.append((episode, float(objective), item))
        if best_route is None or objective < float(best_route["objective"]):
            best_route = {
                key: float(item[key])
                for key in ("area", "power", "delay", "mred")
            }
            best_route.update({"objective": float(objective), "episode": episode})

    cached_by_episode = {
        int(record["episode"]): record for record in cached["records"]
    }
    records = []
    cumulative_best = math.inf
    for episode in range(1, total_episodes + 1):
        logged = logged_means.get(episode)
        if logged is None:
            break
        exact = cached_by_episode.get(episode) if episode <= exact_prefix else None
        if exact is not None:
            mean_objective = float(exact["mean_objective"])
            batch_best = float(exact["batch_best_objective"])
            cumulative_best = float(exact["cumulative_best_objective"])
        else:
            mean_objective = float(logged)
            batch_best = None
            for discovered, objective, _item in pareto_milestones:
                if discovered == episode:
                    cumulative_best = min(cumulative_best, objective)
        records.append(
            {
                "episode": episode,
                "mean_objective": mean_objective,
                "batch_best_objective": batch_best,
                "cumulative_best_objective": float(cumulative_best),
            }
        )

    return {
        "name": "Vanilla PPO",
        "run_dir": str(run_dir.resolve()),
        "completed_episodes": len(records),
        "total_episodes": total_episodes,
        "cache_rows": cached["cache_rows"],
        "incomplete_episode_rows": 0,
        "cache_exact_episodes": exact_prefix,
        "mean_source": (
            f"evaluation cache episodes 1..{exact_prefix}; "
            f"one-epoch PPO log episodes {exact_prefix + 1}..{len(records)}"
        ),
        "best_source": (
            f"evaluation cache episodes 1..{exact_prefix}; "
            "final Pareto discovery episodes thereafter"
        ),
        "records": records,
        "best_route": best_route,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_root", type=Path)
    parser.add_argument("--ppo-dir", type=Path, required=True)
    parser.add_argument("--cem-dir", type=Path, required=True)
    parser.add_argument("--reheat-dir", type=Path)
    parser.add_argument("--vanilla-dir", type=Path)
    parser.add_argument("--nsga2-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes-per-episode", type=int, default=64)
    parser.add_argument("--vanilla-routes-per-episode", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--delay-limit", type=float, default=1.5)
    args = parser.parse_args()

    source_path = args.ppo_dir / "stage2" / "source_elite.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))[0]
    metadata = source.get("metadata") or {}
    role = str(metadata.get("selection_role", "knee"))
    if role not in ROLE_WEIGHTS:
        raise SystemExit(f"unsupported Stage-3 role: {role}")
    band_edges = metadata.get("mred_band_edges") or [0.0, 0.2]
    baseline = {
        key: float(source[key]) for key in ("area", "power", "delay", "mred")
    }
    baseline["mred_hi"] = float(band_edges[1])

    arms = [
        summarize_arm(
            "PPO", args.ppo_dir, baseline, role,
            args.routes_per_episode, args.episodes, args.delay_limit,
        ),
        summarize_arm(
            "CEM", args.cem_dir, baseline, role,
            args.routes_per_episode, args.episodes, args.delay_limit,
        ),
    ]
    if args.reheat_dir is not None:
        arms.append(
            summarize_arm(
                "CEM-reheat", args.reheat_dir, baseline, role,
                args.routes_per_episode, args.episodes, args.delay_limit,
            )
        )
    if args.vanilla_dir is not None:
        arms.append(
            summarize_vanilla_arm(
                args.vanilla_dir, baseline, role,
                args.vanilla_routes_per_episode, args.episodes, args.delay_limit,
            )
        )
    if args.nsga2_dir is not None:
        arms.append(
            summarize_logged_arm("NSGA-II", args.nsga2_dir, args.episodes)
        )

    payload = {
        "generated_at": dt.datetime.now(BEIJING_TZ).isoformat(),
        "pair_root": str(args.pair_root.resolve()),
        "elite_index": 12,
        "candidate_id": source["candidate_id"],
        "selection_role": role,
        "reward_weights": dict(zip(("area", "power", "log_mred"), ROLE_WEIGHTS[role])),
        "objective_definition": "negative Stage-3 routing reward; lower is better",
        "delay_limit_ns": args.delay_limit,
        "baseline": baseline,
        "routes_per_episode": args.routes_per_episode,
        "vanilla_routes_per_episode": args.vanilla_routes_per_episode,
        "total_episodes": args.episodes,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
