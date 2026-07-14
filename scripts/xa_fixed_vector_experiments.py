#!/usr/bin/env python3
"""Measure fixed-vector XA repeatability and workload-bank sensitivity."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_power_sweep import evaluate_single_routing


PRIMARY = "uniform16_medoid_4096_v1"
VALIDATION = [
    "uniform16_validation1_4096_v1",
    "uniform16_validation2_4096_v1",
    "uniform16_validation3_4096_v1",
]


def parse_rtl(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("RTL must be LABEL=PATH")
    label, path = spec.split("=", 1)
    result = Path(path)
    if not label or not result.is_file():
        raise argparse.ArgumentTypeError(f"invalid RTL specification: {spec}")
    return label, result


def cv_percent(values: list[float]) -> float | None:
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return 100.0 * statistics.stdev(values) / statistics.mean(values)


def is_success(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def write_outputs(out: Path, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "design", "vector_set", "repeat", "success", "logic_failed",
        "power_mw", "area", "delay", "vec_cnt", "node_power_count",
        "node_timing_count", "node_toggle_count", "error",
    ]
    with (out / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    designs = sorted({row["design"] for row in rows})
    for design in designs:
        primary = [
            float(row["power_mw"])
            for row in rows
            if row["design"] == design and row["vector_set"] == PRIMARY
            and is_success(row["success"]) and row["power_mw"] != ""
        ]
        bank_means = {}
        for vector_set in [PRIMARY, *VALIDATION]:
            vals = [
                float(row["power_mw"])
                for row in rows
                if row["design"] == design and row["vector_set"] == vector_set
                and is_success(row["success"]) and row["power_mw"] != ""
            ]
            if vals:
                bank_means[vector_set] = statistics.mean(vals)
        summary[design] = {
            "primary_repeat_n": len(primary),
            "primary_mean_mw": statistics.mean(primary) if primary else None,
            "primary_stdev_mw": statistics.stdev(primary) if len(primary) >= 2 else None,
            "primary_repeat_cv_percent": cv_percent(primary),
            "bank_means_mw": bank_means,
            "cross_bank_cv_percent": cv_percent(list(bank_means.values())),
        }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", action="append", required=True, type=parse_rtl)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--primary-repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--target-delay", type=float, default=1.5)
    ap.add_argument("--primary-only", action="store_true")
    args = ap.parse_args()

    rows: list[dict] = []
    existing = args.output / "measurements.csv"
    if existing.is_file():
        with existing.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    completed = {
        (row["design"], row["vector_set"], int(row["repeat"]))
        for row in rows if is_success(row["success"])
    }

    contents = {label: path.read_text(encoding="utf-8") for label, path in args.rtl}
    schedules = [(PRIMARY, args.primary_repeats)]
    if not args.primary_only:
        schedules.extend((name, 1) for name in VALIDATION)
    next_id = 900000
    os.environ["MAX_LIMIT"] = "4096"
    os.environ["DUMP_SAIF"] = "0"

    for vector_set, repeats in schedules:
        os.environ["XA_VECTOR_SET"] = vector_set
        jobs = []
        for design, content in contents.items():
            for repeat in range(repeats):
                if (design, vector_set, repeat) not in completed:
                    jobs.append((next_id, design, repeat, content))
                    next_id += 1
        if not jobs:
            continue
        print(f"vector_set={vector_set} jobs={len(jobs)}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {
                pool.submit(
                    evaluate_single_routing, job_id, content, 16, args.target_delay
                ): (design, repeat)
                for job_id, design, repeat, content in jobs
            }
            for future in concurrent.futures.as_completed(pending):
                design, repeat = pending[future]
                try:
                    result = future.result()
                except Exception as exc:  # keep the resumable record
                    result = {"success": False, "log": repr(exc)}
                ok = bool(result.get("success"))
                row = {
                    "design": design,
                    "vector_set": vector_set,
                    "repeat": repeat,
                    "success": ok,
                    "logic_failed": result.get("logic_failed", ""),
                    "power_mw": result.get("power_mw", ""),
                    "area": result.get("area", ""),
                    "delay": result.get("delay", ""),
                    "vec_cnt": result.get("vec_cnt", ""),
                    "node_power_count": len(result.get("node_powers", {})),
                    "node_timing_count": len(result.get("node_timing", {})),
                    "node_toggle_count": len(result.get("node_toggles", {})),
                    "error": "" if ok else str(result.get("log", ""))[-1000:],
                }
                rows.append(row)
                write_outputs(args.output, rows)
                print(
                    f"{design} {vector_set} rep={repeat} "
                    f"success={ok} power_mw={row['power_mw']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
