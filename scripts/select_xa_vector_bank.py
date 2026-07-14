#!/usr/bin/env python3
"""Select a representative fixed XA vector bank for an unsigned multiplier.

The primary bank is the medoid (minimum standardized discrepancy) among many
independent uniform-random candidate sequences.  The score covers input-bit and
partial-product probabilities/toggles, operand correlation, and transition
Hamming-distance distributions.  Extra fixed banks are emitted for workload
sampling sensitivity checks; they are not used to cherry-pick results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _bit_matrix(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width, dtype=np.uint64)
    return ((values[:, None] >> shifts) & 1).astype(np.float64)


def _standardized_rms(values: np.ndarray, target: float, n: int) -> float:
    se = math.sqrt(target * (1.0 - target) / max(n, 1))
    return float(np.sqrt(np.mean(np.square((values - target) / se))))


def score_sequence(x: np.ndarray, y: np.ndarray, width: int) -> dict[str, float]:
    xb = _bit_matrix(x, width)
    yb = _bit_matrix(y, width)
    bits = np.concatenate((xb, yb), axis=1)
    n = len(x)

    bit_p1 = bits.mean(axis=0)
    bit_toggle = np.not_equal(bits[1:], bits[:-1]).mean(axis=0)

    pp = (xb[:, :, None] * yb[:, None, :]).reshape(n, width * width)
    pp_p1 = pp.mean(axis=0)
    pp_toggle = np.not_equal(pp[1:], pp[:-1]).mean(axis=0)

    hd_x = np.not_equal(xb[1:], xb[:-1]).sum(axis=1).astype(np.int64)
    hd_y = np.not_equal(yb[1:], yb[:-1]).sum(axis=1).astype(np.int64)
    hd = np.concatenate((hd_x, hd_y))
    observed = np.bincount(hd, minlength=width + 1).astype(np.float64)
    expected_prob = np.array(
        [math.comb(width, k) / (2**width) for k in range(width + 1)],
        dtype=np.float64,
    )
    expected = expected_prob * len(hd)
    hd_chi2_red = float(
        np.sum(np.square(observed - expected) / np.maximum(expected, 1.0)) / width
    )

    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    operand_corr = float(np.corrcoef(xf, yf)[0, 1])
    lag_corr_x = float(np.corrcoef(xf[1:], xf[:-1])[0, 1])
    lag_corr_y = float(np.corrcoef(yf[1:], yf[:-1])[0, 1])
    corr_z = math.sqrt(n) * math.sqrt(
        (operand_corr**2 + lag_corr_x**2 + lag_corr_y**2) / 3.0
    )

    components = {
        "bit_p1_zrms": _standardized_rms(bit_p1, 0.5, n),
        "bit_toggle_zrms": _standardized_rms(bit_toggle, 0.5, n - 1),
        "pp_p1_zrms": _standardized_rms(pp_p1, 0.25, n),
        "pp_toggle_zrms": _standardized_rms(pp_toggle, 0.375, n - 1),
        "hd_chi2_reduced": hd_chi2_red,
        "correlation_zrms": corr_z,
        "operand_corr": operand_corr,
        "lag_corr_x": lag_corr_x,
        "lag_corr_y": lag_corr_y,
    }
    # Equal weight for the six independent quality families.  Squaring makes a
    # single pathological statistic harder to hide behind several good ones.
    quality = np.array(
        [
            components["bit_p1_zrms"],
            components["bit_toggle_zrms"],
            components["pp_p1_zrms"],
            components["pp_toggle_zrms"],
            math.sqrt(max(components["hd_chi2_reduced"], 0.0)),
            components["correlation_zrms"],
        ]
    )
    components["score"] = float(np.sqrt(np.mean(np.square(quality))))
    return components


def generate(seed: int, count: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    high = 1 << width
    x = rng.integers(0, high, size=count, dtype=np.uint64)
    y = rng.integers(0, high, size=count, dtype=np.uint64)
    return x, y


def write_bank(root: Path, name: str, seed: int, x: np.ndarray, y: np.ndarray,
               metrics: dict[str, float], width: int) -> None:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    hex_width = (width + 3) // 4
    (out / "x.hex").write_text(
        "".join(f"{int(v):0{hex_width}x}\n" for v in x), encoding="ascii"
    )
    (out / "y.hex").write_text(
        "".join(f"{int(v):0{hex_width}x}\n" for v in y), encoding="ascii"
    )
    (out / "golden.txt").write_text(
        "".join(f"{int(a)} * {int(b)} = {int(a) * int(b)}\n" for a, b in zip(x, y)),
        encoding="ascii",
    )
    meta = {
        "name": name,
        "generator": "numpy.random.PCG64",
        "seed": seed,
        "width": width,
        "count": len(x),
        "distribution": "independent uniform unsigned operands",
        "metrics": metrics,
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--count", type=int, default=4096)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--seed-base", type=int, default=20260713)
    ap.add_argument("--output", type=Path, default=Path("vectors/xa"))
    ap.add_argument(
        "--validation-seeds",
        type=int,
        nargs="*",
        default=[104729, 130363, 169087],
    )
    args = ap.parse_args()

    scored: list[tuple[float, int, dict[str, float]]] = []
    for offset in range(args.candidates):
        seed = args.seed_base + offset
        x, y = generate(seed, args.count, args.width)
        metrics = score_sequence(x, y, args.width)
        scored.append((metrics["score"], seed, metrics))
    scored.sort(key=lambda item: item[0])

    _, best_seed, best_metrics = scored[0]
    x, y = generate(best_seed, args.count, args.width)
    primary_name = f"uniform{args.width}_medoid_{args.count}_v1"
    write_bank(args.output, primary_name, best_seed, x, y, best_metrics, args.width)

    for index, seed in enumerate(args.validation_seeds, start=1):
        x, y = generate(seed, args.count, args.width)
        metrics = score_sequence(x, y, args.width)
        write_bank(
            args.output,
            f"uniform{args.width}_validation{index}_{args.count}_v1",
            seed,
            x,
            y,
            metrics,
            args.width,
        )

    summary = {
        "selection": "minimum standardized discrepancy among candidate sequences",
        "primary": primary_name,
        "primary_seed": best_seed,
        "candidate_count": args.candidates,
        "candidate_seed_range": [args.seed_base, args.seed_base + args.candidates - 1],
        "primary_metrics": best_metrics,
        "score_percentiles": {
            str(p): float(np.percentile([row[0] for row in scored], p))
            for p in (0, 25, 50, 75, 100)
        },
        "validation_seeds": args.validation_seeds,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
