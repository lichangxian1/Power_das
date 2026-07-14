#!/usr/bin/env python3
"""Check RTL compressor-node coverage in XA/SDF observables."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_power_sweep import evaluate_single_routing


NODE_RE = re.compile(r"\b(ct(?:22|32|42)_\d+)\b")
INSTANCE_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_]*\s+(ct(?:22|32|42)_\d+)\s*\(", re.MULTILINE
)


def parse_rtl(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("RTL must be LABEL=PATH")
    label, raw = spec.split("=", 1)
    path = Path(raw)
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid RTL: {spec}")
    return label, path


def normalize(keys) -> set[str]:
    result = set()
    for key in keys:
        match = NODE_RE.search(str(key))
        if match:
            result.add(match.group(1))
    return result


def coverage(expected: set[str], observed: set[str]) -> dict:
    hit = expected & observed
    return {
        "expected": len(expected),
        "observed": len(observed),
        "matched": len(hit),
        "coverage_percent": 100.0 * len(hit) / len(expected) if expected else None,
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", action="append", required=True, type=parse_rtl)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-delay", type=float, default=1.5)
    args = ap.parse_args()

    os.environ["XA_VECTOR_SET"] = "uniform16_medoid_4096_v1"
    os.environ["MAX_LIMIT"] = "4096"
    os.environ["DUMP_SAIF"] = "1"
    os.environ["EDA_RETURN_STDOUT"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)

    report = {}
    for idx, (label, path) in enumerate(args.rtl, start=910000):
        rtl = path.read_text(encoding="utf-8")
        expected = set(INSTANCE_RE.findall(rtl))
        result = evaluate_single_routing(idx, rtl, 16, args.target_delay)
        timing = normalize(result.get("node_timing", {}).keys())
        toggles = normalize(result.get("node_toggles", {}).keys())
        powers = normalize(result.get("node_powers", {}).keys())
        report[label] = {
            "success": bool(result.get("success")),
            "power_mw": result.get("power_mw"),
            "vec_cnt": result.get("vec_cnt"),
            "expected_by_type": {
                kind: sum(name.startswith(kind) for name in expected)
                for kind in ("ct22", "ct32", "ct42")
            },
            "timing": coverage(expected, timing),
            "sdf_toggle": coverage(expected, toggles),
            "node_power": coverage(expected, powers),
            "saif_status": result.get("saif_status", []),
            "stdout_tail": result.get("stdout_tail", ""),
            "error": result.get("log", "") if not result.get("success") else "",
        }
        (args.output / "mapping.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            label,
            "expected=", len(expected),
            "timing=", len(timing),
            "toggle=", len(toggles),
            "power=", len(powers),
            flush=True,
        )


if __name__ == "__main__":
    main()
