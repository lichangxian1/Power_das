#!/usr/bin/env python3
"""纯截断 arith 的 MRED 版基线（零重训）：纯截断下误差与结构/布线无关（objective 排序
不变），最优设计与 area-only 训练一致，唯一差别是补偿常数 C*（mred 口径 argmin E[|C-Δ|/p]）。
故直接取 warm180（缺则 06-27）各 k 的 best_info，用 error_metric=mred 的 exp 重发 RTL
（吃 mred C*），本地 verilator 16M 实测 med/mred/wce 写回 staging best_info，供 reeval。
用法: <python-with-torch> scripts/regen_trunc_mred_baseline.py [staging_dir]
之后: python3 scripts/reeval_xa_glob_tmpbuild.py <staging_dir> 10
"""
import copy
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import trainer  # noqa: E402
from trainer.arith_das import CompressorRouting  # noqa: E402
from utils import get_initial_partial_product  # noqa: E402

CONFIG = "configs/config_groups/mul_16_trunc_areaonly.yaml"
WARM = "outputs/2026-07-09_02_ktrunc_warm180_np5"
FALL = "outputs/2026-06-27_k_trunc_areaonly"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "outputs/2026-07-09_mred_trunc_baseline"
KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def build_exp(k):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    tk = copy.deepcopy(cfg["trainer"]["kwargs"])
    tk.pop("area_budgets", None)
    tk.update(copy.deepcopy(cfg["experiment"]["kwargs"]))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "error_gate_vectors": 16_000_000,
        "delay_scale": 1.44, "area_scale": 800.0, "power_scale": 1.07e-2,
        "log_dir": None, "build_dir": "/tmp/regen_mred_base_build",
        "trunc_cols": int(k), "med_budget": 1e12, "device": "cpu",
        "optim_kwargs": {"lr": float(tk["optim_kwargs"]["lr"])},
    })
    exp = getattr(trainer, cfg["trainer"]["name"])(**tk)
    # mred 口径：清缓存重算截断补偿 C*（对齐 train_dc.py 的 mred 分支）
    exp.error_metric = "mred"
    exp.mred_budget = None
    exp.mred_scale = 0.001
    exp._trunc_bits = {}
    exp._setup_truncation()
    exp.initial_pp = get_initial_partial_product(exp.bit_width, exp.encode_type).astype(int)
    return exp


def massage(bi):
    for kk in ("ct32", "ct22", "ct42"):
        if isinstance(bi.get("ct"), dict) and bi["ct"].get(kk) is not None:
            bi["ct"][kk] = np.array(bi["ct"][kk])
    if isinstance(bi.get("assignment"), list):
        bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                            for stage in bi["assignment"]]
    return bi


def main():
    os.makedirs(STAGE, exist_ok=True)
    n_ok = 0
    for k in KS:
        kk = f"k{k:02d}"
        src = f"{WARM}/{kk}/best_info.json"
        if not os.path.exists(src):
            src = f"{FALL}/{kk}/best_info.json"
        out_dir = f"{STAGE}/{kk}"
        try:
            bi = massage(json.load(open(src)))
            exp = build_exp(k)
            exp.found_best_info = bi
            rtl = exp.export_best_candidate(out_dir)
            me = CompressorRouting._measure_error_verilator(
                rtl, f"/tmp/regen_mred_base_build/verr_{kk}", 16_000_000)
            if not me or me.get("mred") is None:
                raise RuntimeError(f"verilator error measure failed: {me}")
            bi["measured_error"] = me
            bi["error_source"] = "verilator"
            json.dump(bi, open(f"{out_dir}/best_info.json", "w"), default=str)
            print(f"OK  {kk} src={'warm' if WARM in src else 'fallback'} "
                  f"med={me['med']:.4g} mred={me['mred']:.4g} wce={me.get('wce_mc')} -> {rtl}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {kk}: {e}")
    print(f"\n{n_ok}/{len(KS)} mred-C* 基线 RTL 就绪 -> {STAGE}")


if __name__ == "__main__":
    main()
