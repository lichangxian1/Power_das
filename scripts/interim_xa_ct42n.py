#!/usr/bin/env python3
"""approx CT42 ablation 中期 XA 功耗对比：两组同取 save_iter59(=ep60) checkpoint，
复原 RTL 到 staging 目录（reeval_xa_glob_tmpbuild 可直接扫），供 XA 评估。
用法: python scripts/interim_xa_ct42n.py            # 只复原 RTL 到 staging
之后: python scripts/reeval_xa_glob_tmpbuild.py outputs/2026-07-08_18_med_outer_ct42n_np5/interim_ep60 5
"""
import copy
import glob
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import trainer  # noqa: E402
from utils import get_initial_partial_product  # noqa: E402

CONFIG = "configs/config_groups/mul_16_and_approx_p2p1.yaml"
CT = "outputs/2026-07-08_18_med_outer_ct42n_np5"
V1 = "outputs/2026-07-07_22_med_outer_v1_np5"
ITER = "save_iter59"
STAGE = f"{CT}/interim_ep60"
TAGS = ["k09_b1024", "k10_b1536", "k10_b3072", "k11_b5120", "k11_b8192",
        "k12_b12288", "k12_b20480", "k13_b40960", "k14_b81920", "k15_b131072"]


def build_exp(trunc_cols, use_ct42):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    exp_kwargs = cfg["experiment"]["kwargs"]
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(exp_kwargs))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "error_gate_vectors": 16_000_000,
        "delay_scale": 1.44, "area_scale": 800.0, "power_scale": 1.07e-2,
        "log_dir": None, "build_dir": "/tmp/regen_build_ct42n",
        "trunc_cols": int(trunc_cols), "device": "cpu",
    })
    if use_ct42:
        tk.update({
            "use_ct42": True,
            "approx42_library_path": "Appr_Comp/selected_compressors42_native.json",
            "approx42_rtl_path": "Appr_Comp/rtl/comp42n_lib.v",
            "approx42_max_types": 13,
        })
    exp = getattr(trainer, cfg["trainer"]["name"])(**tk)
    exp.initial_pp = get_initial_partial_product(exp.bit_width, exp.encode_type).astype(int)
    if exp.trunc_cols > 0 and not exp._trunc_bits:
        exp._setup_truncation()
    return exp


def massage(bi):
    if isinstance(bi.get("ct"), dict):
        for kk in ("ct32", "ct22", "ct42"):
            if bi["ct"].get(kk) is not None:
                bi["ct"][kk] = np.array(bi["ct"][kk])
    if isinstance(bi.get("assignment"), list):
        bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                            for stage in bi["assignment"]]
    return bi


def main():
    os.makedirs(STAGE, exist_ok=True)
    n_ok = 0
    for group, base, use_ct42 in (("ct42n", CT, True), ("noct42", V1, False)):
        for tag in TAGS:
            k = int(tag.split("_")[0][1:])
            bi_path = f"{base}/{tag}/logs/{ITER}/best_info.json"
            out_dir = f"{STAGE}/{tag}__{group}"
            try:
                bi = massage(json.load(open(bi_path)))
                exp = build_exp(k, use_ct42)
                exp.found_best_info = bi
                rtl = exp.export_best_candidate(out_dir)
                # staging 里的 best_info 供 reeval 读 measured_error
                json.dump(bi, open(f"{out_dir}/best_info.json", "w"), default=str) \
                    if not os.path.exists(f"{out_dir}/best_info.json") else None
                me = bi.get("measured_error") or {}
                n42 = int(np.sum(bi["ct"].get("ct42", 0))) if isinstance(bi.get("ct"), dict) else 0
                print(f"OK  {group:<7} {tag:<13} med={me.get('med')!s:>10} "
                      f"n_ct42={n42} -> {rtl}")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"FAIL {group} {tag}: {e}")
    print(f"\n{n_ok}/20 ep60 RTL 复原完成 -> {STAGE}")


if __name__ == "__main__":
    main()
