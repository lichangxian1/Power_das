#!/usr/bin/env python3
"""Measure the legacy ppa_vs_error_all designs with the three-stage MRED harness.

The output is resumable: rows are reused only when the RTL SHA256 and vector
count match.  Every design sees the same deterministic vector stream used by
the arithmetic three-stage search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "ppa_vs_error_all" / "mred_measurements.csv"
HARNESS = ROOT / "verilate" / "mul_err_wrap.cpp"
VERILATOR = Path("/home/lichangxian/anaconda3/envs/vtool/bin/verilator")
VTOOL_BIN = VERILATOR.parent
ARITH_CUSTOM = Path("/home/lichangxian/Arith-DAS-CUSTOM")
FIELDS = [
    "series",
    "design",
    "rtl_path",
    "rtl_sha256",
    "vectors",
    "med",
    "bias",
    "wce_mc",
    "mred",
    "measured_at_beijing",
]


def design_jobs() -> list[tuple[str, str, Path]]:
    jobs: list[tuple[str, str, Path]] = []
    for series, rel in (
        ("v2", "outputs/2026-06-27_error_obj_v2"),
        ("trunc_arith", "outputs/2026-06-27_k_trunc_areaonly"),
    ):
        base = ARITH_CUSTOM / rel
        for k in range(2, 21, 2):
            design = f"k{k:02d}"
            jobs.append((series, design, base / design / "MUL.v"))

    base = ROOT / "Baselines" / "trunc_dadda_baseline" / "rtl"
    for k in range(1, 25):
        design = f"k{k:02d}"
        jobs.append(("trunc_dadda", design, base / f"MUL_{design}.v"))

    base = ROOT / "outputs" / "2026-06-24_evo_v2022" / "wrappers"
    for rtl in sorted(base.glob("MUL_*.v")):
        design = rtl.stem.removeprefix("MUL_")
        jobs.append(("evo", design, rtl))

    base = (
        ROOT
        / "Baselines"
        / "ELEX4_N"
        / "outputs"
        / "2026-06-24_dcvs_mine"
        / "wrappers"
    )
    for rtl in sorted(base.glob("MUL_*.v")):
        design = rtl.stem.removeprefix("MUL_")
        jobs.append(("elex", design, rtl))
    return jobs


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cached() -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not OUTPUT.exists():
        return {}
    with OUTPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (r["series"], r["design"], r["rtl_sha256"], r["vectors"]): r
        for r in rows
    }


def measure(
    series: str,
    design: str,
    rtl_path: Path,
    rtl_sha256: str,
    vectors: int,
) -> dict[str, str]:
    build_root = Path(tempfile.mkdtemp(prefix=f"ppa_mred_{series}_{design}_"))
    try:
        obj = build_root / "obj_dir"
        env = os.environ.copy()
        env["PATH"] = f"{VTOOL_BIN}:{env.get('PATH', '')}"
        cmd = [
            str(VERILATOR),
            "--cc",
            "--exe",
            "--build",
            "-j",
            "1",
            "-O3",
            "-Wno-fatal",
            "--top-module",
            "MUL",
            "--Mdir",
            str(obj),
            str(rtl_path.resolve()),
            str(HARNESS),
            "-o",
            "mul_err",
        ]
        built = subprocess.run(
            cmd,
            cwd=build_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        exe = obj / "mul_err"
        if built.returncode != 0 or not exe.exists():
            detail = (built.stderr or built.stdout)[-2000:]
            raise RuntimeError(f"Verilator build failed for {series}/{design}: {detail}")
        ran = subprocess.run(
            [str(exe), str(vectors)],
            cwd=build_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if ran.returncode != 0:
            raise RuntimeError(
                f"MRED run failed for {series}/{design}: "
                f"{(ran.stderr or ran.stdout)[-2000:]}"
            )
        values = None
        for line in ran.stdout.splitlines():
            parts = line.strip().split(",")
            if parts and parts[0] == "masked" and len(parts) >= 7:
                values = parts
                break
        if values is None:
            raise RuntimeError(
                f"No masked result for {series}/{design}: {ran.stdout[-2000:]}"
            )
        return {
            "series": series,
            "design": design,
            "rtl_path": str(rtl_path.resolve()),
            "rtl_sha256": rtl_sha256,
            "vectors": str(vectors),
            "med": values[1],
            "bias": values[2],
            "wce_mc": values[5],
            "mred": values[6],
            "measured_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                timespec="seconds"
            ),
        }
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def write_rows(rows: list[dict[str, str]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["series"], r["design"])))
    tmp.replace(OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=16_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--series",
        action="append",
        choices=["v2", "trunc_arith", "trunc_dadda", "evo", "elex"],
        help="Measure only selected series (repeatable).",
    )
    args = parser.parse_args()
    if args.vectors <= 0 or args.workers <= 0:
        parser.error("--vectors and --workers must be positive")
    if not VERILATOR.exists():
        raise SystemExit(f"Verilator not found: {VERILATOR}")

    cached = load_cached()
    rows_by_design = {(r["series"], r["design"]): r for r in cached.values()}
    pending = []
    selected = set(args.series or [])
    for series, design, rtl_path in design_jobs():
        if selected and series not in selected:
            continue
        if not rtl_path.exists():
            raise SystemExit(f"RTL not found: {rtl_path}")
        sha = file_sha256(rtl_path)
        key = (series, design, sha, str(args.vectors))
        if key in cached:
            print(f"cached {series}/{design}", flush=True)
            rows_by_design[(series, design)] = cached[key]
        else:
            pending.append((series, design, rtl_path, sha, args.vectors))

    print(
        f"measuring {len(pending)} designs with {args.workers} workers, "
        f"vectors={args.vectors}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure, *job): job[:2] for job in pending}
        for future in as_completed(futures):
            series, design = futures[future]
            row = future.result()
            rows_by_design[(series, design)] = row
            write_rows(list(rows_by_design.values()))
            print(
                f"done {series}/{design}: mred={float(row['mred']):.8g}",
                flush=True,
            )
    write_rows(list(rows_by_design.values()))
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
