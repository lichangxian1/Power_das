#!/usr/bin/env python3
"""从 save_iter99/best_info.json 复原 ep100 的 best RTL(MUL.v)。
复用 trainer.export_best_candidate：实例化 trainer(cpu)→只跑 _start_reset 的确定性前两步
(initial_pp + _setup_truncation，不触发 get_objective 评估)→塞入保存的 found_best_info→导出。
用法: python regen_ep100_rtl.py  (遍历 outputs/2026-06-26_E_sweep_A/s* 各点)"""
import copy, glob, json, os, sys
import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import trainer  # noqa
from utils import get_initial_partial_product  # noqa

CONFIG = "configs/config_groups/mul_16_and_approx_p2p1.yaml"
D = "outputs/2026-06-26_E_sweep_A"

def build_exp(trunc_cols):
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    exp_kwargs = cfg["experiment"]["kwargs"]; base = cfg["trainer"]["kwargs"]
    tk = copy.deepcopy(base); tk.pop("area_budgets", None); tk.update(copy.deepcopy(exp_kwargs))
    tk.update({
        "synth": "dc", "power_source": "eda", "use_power_proxy": False,
        "area_budget": None, "fixed_target_delay": 1.5, "delay_weight": 0.0,
        "error_gate": "verilator", "error_gate_vectors": 16_000_000,
        "delay_scale": 1.44, "area_scale": 800.0, "power_scale": 1.07e-2,
        "log_dir": None, "build_dir": "/tmp/regen_build",
        "trunc_cols": int(trunc_cols), "device": "cpu",
    })
    exp = getattr(trainer, cfg["trainer"]["name"])(**tk)
    # 只做 _start_reset 的确定性前两步
    exp.initial_pp = get_initial_partial_product(exp.bit_width, exp.encode_type).astype(int)
    if exp.trunc_cols > 0 and not exp._trunc_bits:
        exp._setup_truncation()
    return exp

def main():
    os.chdir(ROOT)
    n_ok = 0
    for d in sorted(glob.glob(f"{D}/s0*")):
        if not os.path.isdir(d): continue
        name = os.path.basename(d)
        k = int(name.split("_k")[1].split("_")[0])
        bi_path = f"{d}/logs/save_iter99/best_info.json"
        out_dir = f"{d}/ckpt_ep100"
        try:
            bi = json.load(open(bi_path))
            if isinstance(bi.get("ct"), dict):
                for kk in ("ct32", "ct22"):
                    if kk in bi["ct"]: bi["ct"][kk] = np.array(bi["ct"][kk])
            # JSON 把 assignment 最内层 vertex_info(4-tuple)变成 list；CompressorGraph
            # 拿它当 dict 键，须还原成 tuple。assignment = [stage][col][vertex] 三层。
            if isinstance(bi.get("assignment"), list):
                bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                                    for stage in bi["assignment"]]
            exp = build_exp(k)
            exp.found_best_info = bi
            rtl = exp.export_best_candidate(out_dir)
            med = bi["measured_error"]["med"]
            print(f"OK  {name:16} k={k:<2} med={med:>10.0f} -> {rtl}")
            n_ok += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL {name}: {e}")
    print(f"\n{n_ok}/10 ep100 RTL 复原完成")

if __name__ == "__main__":
    main()
