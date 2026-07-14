#!/usr/bin/env python3
"""Incremental/resumable fixed-vector XA evaluation for k*/MUL.v datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from run_power_sweep import evaluate_single_routing
from scripts.reeval_xa_glob import wrap_to_31b


FIELDS = ["design", "med", "area_dc", "power_xa_mw", "delay", "success"]


def load_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["design"]: row for row in csv.DictReader(stream)}


def save(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-delay", type=float, default=1.5)
    parser.add_argument("--seed-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve() if args.output else base / "reeval_xa.csv"
    rows = load_rows(output)
    if args.seed_csv:
        rows.update(load_rows(args.seed_csv.resolve()))
        save(output, rows)

    jobs = []
    for directory in sorted(base.glob("k*")):
        design = directory.name
        if design in rows and str(rows[design].get("success", "")).lower() == "true":
            continue
        rtl = directory / "MUL.v"
        if not rtl.is_file():
            continue
        info_path = directory / "best_info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        med = (info.get("measured_error") or {}).get("med")
        jobs.append((design, med, wrap_to_31b(rtl.read_text())))
    print(f"XA resumable jobs={len(jobs)} completed_seed={len(rows)} workers={args.workers}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(evaluate_single_routing, 940000 + idx, rtl, 16, args.target_delay): (design, med)
            for idx, (design, med, rtl) in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(pending):
            design, med = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"success": False, "log": repr(exc)}
            row = {
                "design": design,
                "med": med,
                "area_dc": result.get("area", ""),
                "power_xa_mw": result.get("power_mw", ""),
                "delay": result.get("delay", ""),
                "success": bool(result.get("success")),
            }
            rows[design] = row
            save(output, rows)
            print(
                f"{design}: success={row['success']} area={row['area_dc']} "
                f"power_mw={row['power_xa_mw']} delay={row['delay']}",
                flush=True,
            )
    print(f"saved {len(rows)} rows -> {output}", flush=True)


if __name__ == "__main__":
    main()
