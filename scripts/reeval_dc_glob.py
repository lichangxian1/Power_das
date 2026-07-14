#!/usr/bin/env python3
"""Resumable DC-vectorless PPA evaluation for a directory of k*/MUL.v files."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_power_sweep


FIELDS = ["design", "med", "mred", "area_dc", "power_dc_mw", "delay_dc", "success", "error"]


def save(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--target-delay", type=float, default=1.5)
    args = ap.parse_args()
    args.base = args.base.resolve()
    out = args.base / "reeval_dc.csv"

    run_power_sweep.EDA_BASE_DIR = os.environ.get(
        "EDA_BASE_DIR_DC", "/home/lchangxian/sandbox/sandbox_base_dcpwr"
    )
    rows: dict[str, dict] = {}
    if out.is_file():
        with out.open(newline="", encoding="utf-8") as f:
            rows = {row["design"]: row for row in csv.DictReader(f)}
    jobs = []
    for directory in sorted(args.base.glob("k*")):
        rtl = directory / "MUL.v"
        info_path = directory / "best_info.json"
        if not rtl.is_file() or directory.name in rows and str(rows[directory.name]["success"]).lower() == "true":
            continue
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        measured = info.get("measured_error") or {}
        jobs.append((directory.name, measured.get("med"), measured.get("mred"), rtl.read_text()))
    print(f"DC jobs={len(jobs)} base={args.base}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                run_power_sweep.evaluate_single_routing,
                920000 + idx,
                rtl,
                16,
                args.target_delay,
            ): (design, med, mred)
            for idx, (design, med, mred, rtl) in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(pending):
            design, med, mred = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"success": False, "log": repr(exc)}
            row = {
                "design": design,
                "med": med,
                "mred": mred,
                "area_dc": result.get("area", ""),
                "power_dc_mw": result.get("power_mw", ""),
                "delay_dc": abs(result["delay"]) if result.get("delay") is not None else "",
                "success": bool(result.get("success")),
                "error": "" if result.get("success") else str(result.get("log", ""))[-1000:],
            }
            rows[design] = row
            save(out, rows)
            print(design, row["success"], row["area_dc"], row["power_dc_mw"], flush=True)


if __name__ == "__main__":
    main()
