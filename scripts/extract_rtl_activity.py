#!/usr/bin/env python3
"""Fast functional activity extraction from emitted CompressorTree RTL.

This is a zero-delay bit-vector simulator for the generated compressor network.
It intentionally excludes glitches; its output is an online-safe feature that a
later XA/SDF residual model can calibrate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(ct(?:22|32|42)_\d+)\s*\((.*?)\);\s*$"
)
PORT_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\(([^()]+)\)")
ASSIGN_RE = re.compile(
    r"^\s*assign\s+([A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)\s*=\s*"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?|1'b[01])\s*;"
)
PP_RE = re.compile(
    r"^\s*assign\s+pp_(\d+)\[(\d+)\]\s*=\s*"
    r"(?:a\[(\d+)\]\s*&\s*b\[(\d+)\]|1'b([01]))\s*;",
    re.MULTILINE,
)
NODE_META_RE = re.compile(
    r"^\s*//\s*ct(?:22|32|42)\s+node\s+\((\d+),\s*(\d+),\s*\d+,\s*\d+\)"
)


def load_cells(root: Path) -> dict[str, dict]:
    cells: dict[str, dict] = {}
    for name in ("library.json", "library42_native.json", "selected_compressors_all_substd.json"):
        path = root / "Appr_Comp" / name
        data = json.loads(path.read_text())
        cells.update(data.get("cells", {}))
    lib = json.loads((root / "Appr_Comp/library.json").read_text())
    exact22 = next(v for v in lib["cells"].values() if v.get("is_exact") and v["type"] == "22")
    exact32 = next(v for v in lib["cells"].values() if v.get("is_exact") and v["type"] == "32")
    anchors = json.loads((root / "Appr_Comp/library42_native.json").read_text())["meta"]["anchors"]
    cells["HA"] = {**exact22, "dyn_mw": float(exact22["dyn_w"]) * 1000.0}
    cells["HA_no_carry"] = cells["HA"]
    cells["FA"] = {**exact32, "dyn_mw": float(exact32["dyn_w"]) * 1000.0}
    cells["FA_no_carry"] = cells["FA"]
    cells["CT42"] = {"dyn_mw": anchors["CT42_BAL"]["dyn_mw"]}
    return cells


def fixed_inputs(bank: Path, width: int = 16) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([int(v, 16) for v in (bank / "x.hex").read_text().splitlines()], dtype=np.uint32)
    y = np.array([int(v, 16) for v in (bank / "y.hex").read_text().splitlines()], dtype=np.uint32)
    if len(x) != len(y) or not len(x):
        raise ValueError(f"invalid vector bank {bank}")
    return x & ((1 << width) - 1), y & ((1 << width) - 1)


def lut_for(module: str, cell: dict, outputs: list[str]) -> dict[str, np.ndarray]:
    if module.startswith("FA"):
        idx = np.arange(8)
        count = (idx >> 2) + ((idx >> 1) & 1) + (idx & 1)
        all_luts = {"sum": count & 1, "cout": (count >= 2).astype(np.uint8)}
    elif module.startswith("HA"):
        idx = np.arange(4)
        count = (idx >> 1) + (idx & 1)
        all_luts = {"sum": count & 1, "cout": (count >= 2).astype(np.uint8)}
    elif module == "CT42":
        idx = np.arange(16)
        a, b, c, d = (idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        w = a ^ b
        all_luts = {
            "sum": w ^ c ^ d,
            "carry": (w ^ c) & d,
            "cout": (w & c) | ((1 - w) & a),
        }
    else:
        all_luts = {"sum": np.asarray(cell["sum_lut"], dtype=np.uint8)}
        if "carry_lut" in cell:
            all_luts["carry" if module.startswith("comp42") else "cout"] = np.asarray(
                cell["carry_lut"], dtype=np.uint8
            )
        if "cout_lut" in cell:
            all_luts["cout"] = np.asarray(cell["cout_lut"], dtype=np.uint8)
    return {name: all_luts[name] for name in outputs}


def input_ports(module: str) -> list[str]:
    if module.startswith(("HA", "comp22")):
        return ["a", "cin"]
    if module.startswith(("FA", "comp32")):
        return ["a", "b", "cin"]
    if module.startswith(("CT42", "comp42")):
        return ["a", "b", "c", "d"]
    raise KeyError(module)


def toggle_rate(signal: np.ndarray) -> float:
    return float(np.not_equal(signal[1:], signal[:-1]).mean())


def nominal_toggle(luts: dict[str, np.ndarray]) -> float:
    total = 0.0
    for lut in luts.values():
        p1 = float(lut.mean())
        total += 2.0 * p1 * (1.0 - p1)
    return total


def simulate(path: Path, x: np.ndarray, y: np.ndarray, cells: dict[str, dict]) -> dict:
    src = path.read_text(encoding="utf-8")
    module_match = re.search(r"\bmodule\s+CompressorTree\b.*?;(?P<body>.*?)\bendmodule\b", src, re.S)
    if not module_match:
        raise ValueError(f"CompressorTree not found in {path}")
    body = module_match.group("body")
    n = len(x)
    signals: dict[str, np.ndarray] = {}
    for match in PP_RE.finditer(src[:module_match.start()]):
        col, index = int(match.group(1)), int(match.group(2))
        if match.group(5) is not None:
            value = np.full(n, int(match.group(5)), dtype=np.uint8)
        else:
            value = (((x >> int(match.group(3))) & 1) & ((y >> int(match.group(4))) & 1)).astype(np.uint8)
        signals[f"pp_{col}[{index}]"] = value

    operations = []
    expected_names = set()
    next_meta = None
    for line in body.splitlines():
        meta = NODE_META_RE.match(line)
        if meta:
            next_meta = (int(meta.group(1)), int(meta.group(2)))
            continue
        assign = ASSIGN_RE.match(line)
        if assign:
            operations.append(("assign", assign.group(1), assign.group(2)))
            continue
        instance = INSTANCE_RE.match(line)
        if instance:
            module, name, payload = instance.groups()
            ports = {key: value.strip() for key, value in PORT_RE.findall(payload)}
            operations.append(("cell", module, name, ports, next_meta))
            expected_names.add(name)
            next_meta = None

    uses: dict[str, int] = {}
    for operation in operations:
        if operation[0] == "assign":
            source = operation[2]
            if not source.startswith("1'b"):
                uses[source] = uses.get(source, 0) + 1
        else:
            _, module, _name, ports, _meta = operation
            for key in input_ports(module):
                source = ports.get(key)
                if source:
                    uses[source] = uses.get(source, 0) + 1

    pending = operations
    cell_rows = []
    zero = np.zeros(n, dtype=np.uint8)
    one = np.ones(n, dtype=np.uint8)
    while pending:
        remaining = []
        progressed = False
        for operation in pending:
            if operation[0] == "assign":
                _, dst, source = operation
                if source == "1'b0":
                    signals[dst] = zero
                elif source == "1'b1":
                    signals[dst] = one
                elif source in signals:
                    signals[dst] = signals[source]
                else:
                    remaining.append(operation)
                    continue
                progressed = True
                continue

            _, module, name, ports, meta = operation
            inputs = input_ports(module)
            if not all(ports.get(key) in signals for key in inputs):
                remaining.append(operation)
                continue
            index = signals[ports[inputs[0]]].astype(np.int64)
            for key in inputs[1:]:
                index = (index << 1) | signals[ports[key]]
            output_ports = [key for key in ("sum", "carry", "cout") if key in ports]
            cell = cells.get(module)
            if cell is None:
                raise KeyError(f"missing cell metadata for {module}")
            luts = lut_for(module, cell, output_ports)
            rates = {}
            for key in output_ports:
                value = luts[key][index].astype(np.uint8)
                signals[ports[key]] = value
                rates[key] = toggle_rate(value)
            actual = sum(rates.values())
            nominal = nominal_toggle(luts)
            dyn = float(cell.get("dyn_mw", float(cell.get("dyn_w", 0.0)) * 1000.0))
            cell_rows.append({
                "name": name,
                "module": module,
                "type": "42" if name.startswith("ct42") else ("32" if name.startswith("ct32") else "22"),
                "toggle": actual,
                "sum_toggle": rates.get("sum", 0.0),
                "carry_toggle": rates.get("carry", rates.get("cout", 0.0)),
                "extra_cout_toggle": rates.get("cout", 0.0) if "carry" in rates else 0.0,
                "fanout_toggle": sum(
                    rate * max(uses.get(ports[key], 0), 1) for key, rate in rates.items()
                ),
                "stage": meta[0] if meta else -1,
                "col": meta[1] if meta else -1,
                "activity_scaled_dyn_mw": dyn * actual / nominal if nominal > 0 else 0.0,
            })
            progressed = True
        if not progressed:
            sample = remaining[:8]
            raise RuntimeError(f"unresolved operations={len(remaining)} sample={sample}")
        pending = remaining

    simulated = {row["name"] for row in cell_rows}
    if simulated != expected_names:
        raise RuntimeError(f"cell coverage {len(simulated)}/{len(expected_names)}")
    result = {
        "design": path.parent.name,
        "rtl_path": str(path),
        "n_cells": len(cell_rows),
        "n_ct22": sum(row["type"] == "22" for row in cell_rows),
        "n_ct32": sum(row["type"] == "32" for row in cell_rows),
        "n_ct42": sum(row["type"] == "42" for row in cell_rows),
        "functional_toggle_total": sum(row["toggle"] for row in cell_rows),
        "functional_toggle_mean": float(np.mean([row["toggle"] for row in cell_rows])),
        "functional_toggle_sum_port": sum(row["sum_toggle"] for row in cell_rows),
        "functional_toggle_carry_ports": sum(
            row["carry_toggle"] + row["extra_cout_toggle"] for row in cell_rows
        ),
        "fanout_weighted_toggle": sum(row["fanout_toggle"] for row in cell_rows),
        "activity_scaled_dyn_mw": sum(row["activity_scaled_dyn_mw"] for row in cell_rows),
    }
    for kind in ("22", "32", "42"):
        result[f"functional_toggle_{kind}"] = sum(
            row["toggle"] for row in cell_rows if row["type"] == kind
        )
    for label, lo, hi in (("low", 0, 9), ("mid", 10, 20), ("high", 21, 10**9)):
        result[f"functional_toggle_col_{label}"] = sum(
            row["toggle"] for row in cell_rows if lo <= row["col"] <= hi
        )
    valid_stages = [row["stage"] for row in cell_rows if row["stage"] >= 0]
    max_stage = max(valid_stages) if valid_stages else 0
    for label, lo_frac, hi_frac in (
        ("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01)
    ):
        result[f"functional_toggle_stage_{label}"] = sum(
            row["toggle"] for row in cell_rows
            if row["stage"] >= 0
            and lo_frac <= row["stage"] / max(max_stage, 1) < hi_frac
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--bank", type=Path, default=Path("vectors/xa/uniform16_medoid_4096_v1"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cells = load_cells(root)
    x, y = fixed_inputs(args.bank)
    paths = []
    for item in args.paths:
        paths.extend(sorted(item.glob("k*/MUL.v")) if item.is_dir() else [item])
    rows = []
    for path in paths:
        row = simulate(path, x, y, cells)
        rows.append(row)
        print(path, row["n_cells"], row["activity_scaled_dyn_mw"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
