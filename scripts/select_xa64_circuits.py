#!/usr/bin/env python3
"""Select 8 structurally/activity-diverse RTLs from each captured V5r2 stratum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from extract_rtl_activity import fixed_inputs, load_cells, simulate


FEATURES = [
    "n_cells", "n_ct22", "n_ct32", "n_ct42",
    "functional_toggle_total", "functional_toggle_mean",
    "functional_toggle_22", "functional_toggle_32", "functional_toggle_42",
    "functional_toggle_sum_port", "functional_toggle_carry_ports",
    "functional_toggle_col_low", "functional_toggle_col_mid",
    "functional_toggle_col_high", "functional_toggle_stage_early",
    "functional_toggle_stage_middle", "functional_toggle_stage_late",
]


def diverse_indices(rows: list[dict], count: int) -> list[int]:
    matrix = np.asarray([[float(row[name]) for name in FEATURES] for row in rows])
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    z = (matrix - matrix.mean(axis=0)) / scale
    centroid_distance = np.linalg.norm(z, axis=1)
    selected = [int(np.argmin(centroid_distance))]
    while len(selected) < count:
        distance = np.min(
            np.linalg.norm(z[:, None, :] - z[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("outputs/2026-07-14_xa64_v5r2"))
    parser.add_argument("--bank", type=Path, default=Path("vectors/xa/uniform16_medoid_4096_v1"))
    parser.add_argument("--per-stratum", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cells = load_cells(root)
    x, y = fixed_inputs(args.bank)
    strata = sorted(path for path in (args.study / "raw").glob("*/*") if path.is_dir())
    if len(strata) != 8:
        raise RuntimeError(f"expected 8 captured strata, found {len(strata)}")
    selected_root = args.study / "selected"
    if selected_root.exists() and any(selected_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {selected_root}")
    selected_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    selected_manifest: list[dict] = []
    global_hashes: set[str] = set()
    for group_index, stratum in enumerate(strata):
        rows = []
        for rtl in sorted(stratum.glob("MUL-*.v")):
            digest = hashlib.sha256(rtl.read_bytes()).hexdigest()
            row = simulate(rtl, x, y, cells)
            row.update({
                "stratum": stratum.relative_to(args.study).as_posix(),
                "source_rtl": str(rtl),
                "sha256": digest,
                "sample_index": int(rtl.stem.split("-")[-1]),
            })
            rows.append(row)
            all_rows.append(row)
        # Collapse exact duplicates before diversity selection. Duplicate designs
        # add EDA cost without adding evidence about proxy generalization.
        unique_rows = []
        local_hashes = set()
        for row in rows:
            if row["sha256"] not in local_hashes and row["sha256"] not in global_hashes:
                unique_rows.append(row)
                local_hashes.add(row["sha256"])
        if len(unique_rows) < args.per_stratum:
            raise RuntimeError(f"{stratum} only has {len(unique_rows)} globally unique RTLs")
        chosen = [unique_rows[idx] for idx in diverse_indices(unique_rows, args.per_stratum)]
        for within_index, row in enumerate(chosen):
            design = f"k{group_index:02d}_b{stratum.name}_s{within_index:02d}"
            destination = selected_root / design
            destination.mkdir()
            shutil.copy2(row["source_rtl"], destination / "MUL.v")
            (destination / "best_info.json").write_text(json.dumps({
                "design": design,
                "source_rtl": row["source_rtl"],
                "stratum": row["stratum"],
                "sample_index": row["sample_index"],
                "sha256": row["sha256"],
                "measured_error": {},
            }, indent=2, sort_keys=True) + "\n")
            selected_manifest.append({"design": design, **row})
            global_hashes.add(row["sha256"])

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.study / "activity_all_256.csv", all_rows)
    write_csv(args.study / "selection_manifest.csv", selected_manifest)
    print(f"selected {len(selected_manifest)} RTLs from {len(strata)} strata -> {selected_root}")


if __name__ == "__main__":
    main()
