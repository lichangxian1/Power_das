#!/usr/bin/env python3
"""Paired exact-vs-approx comparison for identical routing/tree.

This script emits two RTLs per pair:
  - exact: same routing/tree, all FA/HA exact
  - with_cell: same routing/tree, sampled or saved approximate cells enabled

It then runs DC PPA and verilator error measurement for both.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trainer.arith_das import CompressorGraph, CompressorRouting  # noqa: E402
from utils import CompressorTree, Mul, get_initial_partial_product  # noqa: E402
import run_power_sweep  # noqa: E402


def load_cfg_kwargs(config: Path, out_dir: Path, trunc_cols: int, device: str):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    tk = dict(cfg["trainer"]["kwargs"])
    tk.update(dict(cfg["experiment"]["kwargs"]))
    tk.update(
        {
            "synth": "dc",
            "power_source": "eda",
            "use_power_proxy": False,
            "area_budget": None,
            "fixed_target_delay": 1.5,
            "delay_weight": 0.0,
            "error_gate": "verilator",
            "error_gate_vectors": 16_000_000,
            "delay_scale": 1.44,
            "area_scale": 800.0,
            "power_scale": 1.07e-2,
            "trunc_cols": trunc_cols,
            "device": device,
            "log_dir": str(out_dir / "_dummy_logs"),
            "build_dir": str(out_dir / "_dummy_build"),
            "num_samples": 1,
            "n_processing": 1,
            "n_full_target_delay_processing": 1,
        }
    )
    tk.pop("area_budgets", None)
    return cfg, tk


def setup_dummy(sel_path: str, lib_path: str):
    dummy = object.__new__(CompressorRouting)
    dummy.approx_module_src_by_name = {}
    dummy._load_approx_types(sel_path, lib_path)
    return dummy


def tuple_assignment(assignment):
    return [
        [[tuple(v) for v in col] for col in stage]
        for stage in assignment
    ]


def trunc_bits_from_log(run_dir: Path) -> dict[int, int]:
    for name in ("train_dc.log", f"{run_dir.name}.log"):
        p = run_dir / name
        if not p.exists():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            if "[trunc]" in line and "bits=" in line:
                m = re.search(r"bits=(\{.*\})", line)
                if m:
                    return {int(k): int(v) for k, v in ast.literal_eval(m.group(1)).items()}
    raise RuntimeError(f"cannot find trunc bits in {run_dir}")


def ct_from_best(best: dict, trunc_cols: int, trunc_bits: dict[int, int]):
    initial_pp = get_initial_partial_product(16, "and")
    ct = CompressorTree(
        initial_pp,
        np.asarray(best["ct"]["ct32"], dtype=int),
        np.asarray(best["ct"]["ct22"], dtype=int),
    )
    ct.trunc_cols = int(trunc_cols)
    ct.trunc_bits = dict(trunc_bits)
    return ct


def emit_pair_from_best(best_info: Path, out_dir: Path, label: str, sel_path: str, lib_path: str):
    best = json.load(open(best_info))
    run_dir = best_info.parents[2]
    trunc_cols = int(re.search(r"k(\d+)", run_dir.name).group(1))
    trunc_bits = trunc_bits_from_log(run_dir)
    dummy = setup_dummy(sel_path, lib_path)
    best_assignment = tuple_assignment(best["assignment"])
    dummy.comp_graph = CompressorGraph(get_initial_partial_product(16, "and"), best_assignment)
    cell_types = best.get("cell_types") or {}
    cell_map = dummy._cell_map_from_types(cell_types)
    ct = ct_from_best(best, trunc_cols, trunc_bits)
    mul = Mul(16, "and", ct)
    jobs = []
    for mode, cmap in [("exact", {}), ("with_cell", cell_map)]:
        assignment = dummy.emit_assignment(best["connection"], cell_map=cmap)
        rtl = out_dir / f"{label}_{mode}.v"
        extra = dummy._approx_modules_src(cmap)
        mul.emit_verilog(str(rtl), assignment=assignment, extra_modules_src=extra)
        jobs.append(
            {
                "label": label,
                "source": "saved_best",
                "mode": mode,
                "trunc_cols": trunc_cols,
                "n_approx": len(cmap),
                "cell_names": ";".join(sorted(cmap.values())),
                "rtl": str(rtl),
            }
        )
    return jobs


def load_checkpointed_exp(config: Path, run_dir: Path, save_iter: str, out_dir: Path, device: str):
    trunc_cols = int(re.search(r"k(\d+)", run_dir.name).group(1))
    _cfg, tk = load_cfg_kwargs(config, out_dir, trunc_cols, device)
    exp = CompressorRouting(**tk)
    ckpt = run_dir / "logs" / save_iter
    exp.gcn.load_state_dict(torch.load(ckpt / "gcn.pth", map_location=device))
    type_state = torch.load(ckpt / "type_heads.pth", map_location=device)
    exp.type_head_32.load_state_dict(type_state["type_head_32"])
    exp.type_head_22.load_state_dict(type_state["type_head_22"])
    if exp.approx_cardinality_logits is not None and "approx_cardinality_logits" in type_state:
        with torch.no_grad():
            exp.approx_cardinality_logits.copy_(type_state["approx_cardinality_logits"].to(exp.device))

    best = json.load(open(ckpt / "best_info.json"))
    exp.state = ct_from_best(best, trunc_cols, exp._trunc_bits)
    exp.assignment = tuple_assignment(best["assignment"])
    exp.comp_graph = CompressorGraph(exp.initial_pp, exp.assignment)
    return exp, trunc_cols


def emit_sampled_pairs(config: Path, run_dir: Path, save_iter: str, out_dir: Path, n_pairs: int, max_draws: int, device: str):
    exp, trunc_cols = load_checkpointed_exp(config, run_dir, save_iter, out_dir, device)
    jobs = []
    draws = 0
    with torch.no_grad():
        z = exp.get_Z_mat()
        while len(jobs) < n_pairs * 2 and draws < max_draws:
            draws += 1
            conn, _ = exp.sample_from_logits(z)
            cell_map, type_choices, _lp, _info = exp.sample_cell_types()
            if not cell_map:
                continue
            pair_id = len(jobs) // 2
            ct = exp.state
            ct.trunc_cols = trunc_cols
            ct.trunc_bits = dict(exp._trunc_bits)
            mul = Mul(16, "and", ct)
            for mode, cmap in [("exact", {}), ("with_cell", cell_map)]:
                assignment = exp.emit_assignment(conn, cell_map=cmap)
                rtl = out_dir / f"{run_dir.name}_sample{pair_id:02d}_{mode}.v"
                extra = exp._approx_modules_src(cmap)
                mul.emit_verilog(str(rtl), assignment=assignment, extra_modules_src=extra)
                jobs.append(
                    {
                        "label": f"{run_dir.name}_sample{pair_id:02d}",
                        "source": "sampled_checkpoint",
                        "mode": mode,
                        "trunc_cols": trunc_cols,
                        "n_approx": len(cmap),
                        "cell_names": ";".join(sorted(cmap.values())),
                        "rtl": str(rtl),
                    }
                )
    return jobs


def eval_job(job: dict, idx: int, out_dir: Path, target_delay: float, vectors: int):
    run_power_sweep.EDA_BASE_DIR = os.environ.get(
        "EDA_BASE_DIR_DC", "/home/lchangxian/sandbox/sandbox_base_dcpwr"
    )
    rtl = Path(job["rtl"])
    res = run_power_sweep.evaluate_single_routing(
        idx, rtl.read_text(), bit_width=16, target_delay=target_delay
    )
    err = CompressorRouting._measure_error_verilator(
        str(rtl), str(out_dir / f"err_{idx:03d}_{job['label']}_{job['mode']}"), vectors
    )
    row = dict(job)
    row.update(
        {
            "success": bool(res and res.get("success")),
            "area_dc": "" if not res else res.get("area"),
            "power_dc_mw": "" if not res else res.get("power_mw"),
            "delay": "" if not res else res.get("delay"),
            "med": "" if not err else err.get("med"),
            "mred": "" if not err else err.get("mred"),
            "bias": "" if not err else err.get("bias"),
            "wce_mc": "" if not err else err.get("wce_mc"),
        }
    )
    return row


def write_delta(rows: list[dict], path: Path):
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["mode"]] = r
    fields = [
        "label",
        "source",
        "trunc_cols",
        "n_approx",
        "cell_names",
        "area_exact",
        "area_with_cell",
        "delta_area",
        "power_exact_mw",
        "power_with_cell_mw",
        "delta_power_mw",
        "med_exact",
        "med_with_cell",
        "delta_med",
        "mred_exact",
        "mred_with_cell",
        "delta_mred",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, modes in sorted(by_label.items()):
            if "exact" not in modes or "with_cell" not in modes:
                continue
            e, a = modes["exact"], modes["with_cell"]
            def fl(r, k):
                try:
                    return float(r[k])
                except Exception:
                    return float("nan")
            w.writerow(
                {
                    "label": label,
                    "source": a["source"],
                    "trunc_cols": a["trunc_cols"],
                    "n_approx": a["n_approx"],
                    "cell_names": a["cell_names"],
                    "area_exact": e["area_dc"],
                    "area_with_cell": a["area_dc"],
                    "delta_area": fl(a, "area_dc") - fl(e, "area_dc"),
                    "power_exact_mw": e["power_dc_mw"],
                    "power_with_cell_mw": a["power_dc_mw"],
                    "delta_power_mw": fl(a, "power_dc_mw") - fl(e, "power_dc_mw"),
                    "med_exact": e["med"],
                    "med_with_cell": a["med"],
                    "delta_med": fl(a, "med") - fl(e, "med"),
                    "mred_exact": e["mred"],
                    "mred_with_cell": a["mred"],
                    "delta_mred": fl(a, "mred") - fl(e, "mred"),
                }
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--best-info", action="append", default=[])
    ap.add_argument("--sample-run-dir")
    ap.add_argument("--save-iter", default="save_iter19")
    ap.add_argument("--sample-pairs", type=int, default=0)
    ap.add_argument("--max-draws", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vectors", type=int, default=16_000_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-delay", type=float, default=1.5)
    ap.add_argument("--approx-lib-path", default="Appr_Comp/.bak_20260622_222332/selected_compressors.json")
    ap.add_argument("--approx-library-path", default="Appr_Comp/library.json")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in args.best_info:
        best_path = Path(p)
        label = best_path.parents[2].name + "_" + best_path.parent.name
        jobs.extend(
            emit_pair_from_best(
                best_path,
                out_dir,
                label,
                args.approx_lib_path,
                args.approx_library_path,
            )
        )
    if args.sample_run_dir and args.sample_pairs:
        jobs.extend(
            emit_sampled_pairs(
                Path(args.config),
                Path(args.sample_run_dir),
                args.save_iter,
                out_dir,
                args.sample_pairs,
                args.max_draws,
                args.device,
            )
        )
    with open(out_dir / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"emitted {len(jobs)} RTL jobs")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(eval_job, job, i, out_dir, args.target_delay, args.vectors): job
            for i, job in enumerate(jobs)
        }
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(
                row["label"],
                row["mode"],
                "area",
                row["area_dc"],
                "pwr",
                row["power_dc_mw"],
                "med",
                row["med"],
            )
    fields = [
        "label",
        "source",
        "mode",
        "trunc_cols",
        "n_approx",
        "cell_names",
        "success",
        "area_dc",
        "power_dc_mw",
        "delay",
        "med",
        "mred",
        "bias",
        "wce_mc",
        "rtl",
    ]
    with open(out_dir / "paired_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["label"], r["mode"])))
    write_delta(rows, out_dir / "paired_delta.csv")
    print("wrote", out_dir / "paired_raw.csv")
    print("wrote", out_dir / "paired_delta.csv")


if __name__ == "__main__":
    main()
