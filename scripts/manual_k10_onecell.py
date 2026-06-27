#!/usr/bin/env python3
"""Restore the 2026-06-27 pure-k k10 best RTL and emit one-cell approx variants.

This is a small, auditable experiment: keep the searched routing/tree fixed and
replace exactly one exact compressor in the first non-truncated column.
"""
import argparse
import copy
import csv
import itertools
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
APPR = os.path.join(ROOT, "Appr_Comp")
if APPR not in sys.path:
    sys.path.insert(0, APPR)

import trainer  # noqa: E402
from gen_verilog import emit_module  # noqa: E402
from trainer.arith_das import CompressorGraph, CompressorRouting  # noqa: E402
from utils import get_initial_partial_product  # noqa: E402
from utils.compressor_tree import CompressorTree  # noqa: E402
from utils.mul import Mul  # noqa: E402

CONFIG = os.path.join(ROOT, "configs/config_groups/mul_16_trunc_areaonly.yaml")
BEST = os.path.join(ROOT, "outputs/2026-06-27_k_trunc_areaonly/k10/best_info.json")
OUT = os.path.join(ROOT, "outputs/2026-06-27_k10_manual_onecell")
SEL = json.load(open(os.path.join(APPR, "selected_compressors.json")))["selected"]
LIB = json.load(open(os.path.join(APPR, "library.json")))["cells"]


def _json_to_runtime_best(info):
    info = copy.deepcopy(info)
    if isinstance(info.get("ct"), dict):
        for key in ("ct32", "ct22"):
            if key in info["ct"]:
                info["ct"][key] = np.array(info["ct"][key])
    if isinstance(info.get("assignment"), list):
        info["assignment"] = [
            [[tuple(v) for v in col] for col in stage]
            for stage in info["assignment"]
        ]
    return info


def _build_exp(trunc_cols=10):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    exp_kwargs = cfg["experiment"]["kwargs"]
    base = cfg["trainer"]["kwargs"]
    tk = copy.deepcopy(base)
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(exp_kwargs))
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
            "log_dir": None,
            "build_dir": os.path.join(OUT, "build_restore"),
            "trunc_cols": int(trunc_cols),
            "device": "cpu",
        }
    )
    exp = getattr(trainer, cfg["trainer"]["name"])(**tk)
    exp.initial_pp = get_initial_partial_product(exp.bit_width, exp.encode_type).astype(int)
    if exp.trunc_cols > 0 and not exp._trunc_bits:
        exp._setup_truncation()
    return exp


def _module_src(module_name):
    cell = LIB[module_name]
    if cell["type"] == "32":
        pats = ["".join(p) for p in itertools.product("01", repeat=3)]
        return emit_module(
            module_name,
            ["a", "b", "cin"],
            pats,
            cell["sum_lut"],
            cell["carry_lut"],
            f"{module_name} bias={cell['weighted_signed_error']:+.4f}",
        )
    pats = ["".join(p) for p in itertools.product("01", repeat=2)]
    return emit_module(
        module_name,
        ["a", "cin"],
        pats,
        cell["sum_lut"],
        cell["carry_lut"],
        f"{module_name} bias={cell['weighted_signed_error']:+.4f}",
    )


def _emit(exp, best, out_dir, name, cell_map):
    os.makedirs(out_dir, exist_ok=True)
    exp.state = copy.deepcopy(best["ct"])
    exp.assignment = copy.deepcopy(best["assignment"])
    exp.comp_graph = CompressorGraph(exp.initial_pp, exp.assignment)
    routing_assignment = exp.emit_assignment(best["connection"], cell_map=cell_map)
    ct = CompressorTree(exp.initial_pp, exp.state["ct32"], exp.state["ct22"])
    ct.trunc_cols = exp.trunc_cols
    ct.trunc_bits = exp._trunc_bits
    mul = Mul(exp.bit_width, exp.encode_type, ct)
    used = sorted(set(cell_map.values()))
    extra = ""
    if used:
        extra = "\n// ===== manual approximate compressor cells =====\n"
        extra += "".join(_module_src(m) for m in used)
    rtl = os.path.join(out_dir, f"{name}.v")
    mul.emit_verilog(rtl, assignment=routing_assignment, extra_modules_src=extra)
    return rtl


def _node_for_slot(exp, slot):
    for idx, info in enumerate(exp.comp_graph.vertex_list):
        if tuple(info) == tuple(slot):
            return idx
    raise KeyError(f"slot not found: {slot}")


def _analytic_delta(cell_alias, col):
    entry = SEL[cell_alias]
    return {
        "alias": cell_alias,
        "module": entry["name"],
        "type": entry["type"],
        "bias_delta": entry["bias"] * (1 << col),
        "wae_delta": entry["wae"] * (1 << col),
        "maxe_delta": entry["maxe"] * (1 << col),
        "cell_area": entry["area"],
        "cell_power_mw": entry["power_mw"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=int, default=2_000_000)
    ap.add_argument("--eval-error", action="store_true")
    ap.add_argument("--eval-xa", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    best = _json_to_runtime_best(json.load(open(BEST)))
    exp = _build_exp(10)
    exp.found_best_info = best
    exp.state = copy.deepcopy(best["ct"])
    exp.assignment = copy.deepcopy(best["assignment"])
    exp.comp_graph = CompressorGraph(exp.initial_pp, exp.assignment)

    variants = [
        {
            "name": "baseline_k10_exact",
            "slot": None,
            "alias": None,
            "why": "original pure-k k10 baseline",
        },
        {
            "name": "one_ha_s0_c10_i0_comp22_54",
            "slot": (0, 10, 1, 0),
            "alias": "comp22_apx_neg_1",
            "why": "small-error HA: local WAE 0.0625, saves about 4.2um2 vs exact HA",
        },
        {
            "name": "one_ha_s0_c10_i0_comp22_50",
            "slot": (0, 10, 1, 0),
            "alias": "comp22_apx_neg_2",
            "why": "aggressive low-area HA: local WAE 0.25, may save real XA power",
        },
        {
            "name": "one_ha_s0_c10_i0_comp22_e4",
            "slot": (0, 10, 1, 0),
            "alias": "comp22_apx_pos_2",
            "why": "aggressive zero-area positive HA for sign/power contrast",
        },
        {
            "name": "one_fa_s0_c10_i0_comp32_a994",
            "slot": (0, 10, 0, 0),
            "alias": "comp32_apx_neg_1",
            "why": "most conservative FA: local WAE 0.015625, saves about 1.0um2 vs exact FA",
        },
    ]
    rows = []
    for var in variants:
        cell_map = {}
        node_idx = None
        module = None
        analytic = None
        if var["slot"] is not None:
            node_idx = _node_for_slot(exp, var["slot"])
            module = SEL[var["alias"]]["name"]
            cell_map[node_idx] = module
            analytic = _analytic_delta(var["alias"], var["slot"][1])
        rtl = _emit(exp, best, OUT, var["name"], cell_map)
        row = {
            "name": var["name"],
            "rtl": rtl,
            "slot": var["slot"],
            "node_idx": node_idx,
            "alias": var["alias"],
            "module": module,
            "why": var["why"],
            "analytic": analytic,
        }
        if args.eval_error:
            row["measured_error"] = CompressorRouting._measure_error_verilator(
                rtl, os.path.join(OUT, "verilator", var["name"]), args.vectors
            )
        rows.append(row)

    if args.eval_xa:
        from run_power_sweep import evaluate_single_routing

        csv_path = os.path.join(OUT, "ppa_xa.csv")
        eval_cwd = os.path.join(OUT, "xa_eval_work")
        os.makedirs(eval_cwd, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(eval_cwd)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "design",
                        "area_dc",
                        "power_xa_mw",
                        "delay",
                        "success",
                        "logic_failed",
                    ]
                )
                for idx, row in enumerate(rows):
                    content = open(row["rtl"]).read()
                    ppa = evaluate_single_routing(idx, content, 16, 1.5)
                    row["xa_ppa"] = ppa
                    writer.writerow(
                        [
                            row["name"],
                            ppa.get("area"),
                            ppa.get("power_mw"),
                            ppa.get("delay"),
                            ppa.get("success"),
                            ppa.get("logic_failed"),
                        ]
                    )
                    print(
                        f"  XA {row['name']}: success={ppa.get('success')} "
                        f"area={ppa.get('area')} power_mw={ppa.get('power_mw')} "
                        f"delay={ppa.get('delay')}"
                    )
        finally:
            os.chdir(old_cwd)
        print(f"ppa csv -> {csv_path}")

    summary = os.path.join(OUT, "manual_onecell_summary.json")
    with open(summary, "w") as f:
        json.dump(rows, f, indent=2)
    for row in rows:
        print(f"{row['name']}: {row['rtl']}")
        if row["analytic"]:
            a = row["analytic"]
            print(
                "  "
                f"{row['slot']} node={row['node_idx']} {row['alias']}->{row['module']} "
                f"analytic wae={a['wae_delta']:.1f} bias={a['bias_delta']:+.1f} "
                f"maxe={a['maxe_delta']:.0f}"
            )
        if row.get("measured_error"):
            me = row["measured_error"]
            print(
                f"  verilator MED={me['med']:.3f} bias={me['bias']:+.3f} "
                f"WCE_MC={me['wce_mc']:.0f}"
            )
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
