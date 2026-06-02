"""探针: 同一份 Verilog 跑 N 次 EDA, 测 DC 综合的真实噪声.

如果 power std < 0.05% → DC 确定性, 多次平均零收益
如果 power std 0.1-0.3% → 弱随机, 平均边际收益
如果 power std > 0.5% → 显著随机, 平均能显著降噪
"""
import os
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.arith_das import CompressorRouting
from utils import CompressorTree, Mul
from send_eda import _evaluate_single_design_full


def main(bit_width=16, target_delay=2.0, n_runs=5):
    print(f"🧪 DC 确定性探针: bit_width={bit_width}, target_delay={target_delay}, n_runs={n_runs}")
    print(f"   预期 wall time ≈ 单样本 EDA 时间 (并发跑 N 次)")
    print()

    # 1) Init env (跟 generate_dataset.py 一致的参数)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CompressorRouting(
        bit_width=bit_width, encode_type="and", ct_arch="dadda",
        use_ppo_loss=False, ppo_loss_weight=0, use_delay_loss=False, delay_loss_weight=0,
        lse_gamma_val=0.01, use_rule_loss=False, rule_loss_weight=0,
        use_disc_loss=False, disc_loss_weight=0, num_episodes=0,
        num_samples=1, num_epochs=1, log_dir=None, build_dir="build_probe_dc",
        save_freq=999, log_freq=999, device=device, optim_name="Adam", optim_kwargs={"lr": 1e-4},
        scheduler_name="CosineAnnealingLR", scheduler_kwargs={"T_max": 100, "eta_min": 1e-5},
        gcn_kwargs={"input_dim": 7, "hidden_dims_list": [[64]], "output_dim": 64},
        delay_weight=1, area_weight=1, power_weight=1, delay_scale=1, area_scale=1, power_scale=1,
        clip_range=0.2, max_grad_norm=0.5, n_processing=1, reference_point=[1, 1],
        pareto_target=["delay", "area"],
        pool_size=10, rule_loss_wight_incr=0, disc_loss_weight_incr=0,
    )
    env.reset()
    Z = env.get_Z_mat()

    # 2) 采样一个 routing
    print("📋 采样一个 routing ...")
    with torch.no_grad():
        samples_connection, _ = env.sample_from_logits(Z)
        assignment = env.emit_assignment(samples_connection)

    # 3) Emit Verilog
    os.makedirs("build_probe_dc", exist_ok=True)
    ct = CompressorTree(env.initial_pp, env.state["ct32"], env.state["ct22"])
    mul = Mul(env.bit_width, env.encode_type, ct)
    v_path = "build_probe_dc/probe_dc_test.v"
    mul.emit_verilog(v_path, assignment=assignment)
    print(f"✅ Verilog 已生成: {v_path}")
    print(f"   文件大小: {os.path.getsize(v_path)} bytes")
    print()

    # 4) 并行跑 N 次同样的 EDA
    print(f"🚀 并发跑 {n_runs} 次 EDA (同一 Verilog) ...")
    t_start = time.time()

    def run_one(idx):
        t0 = time.time()
        res = _evaluate_single_design_full(v_path, target_delay, bit_width)
        elapsed = time.time() - t0
        print(f"  [run {idx}] {elapsed:.0f}s | "
              f"power={res.get('power', 'N/A')!r} | "
              f"area={res.get('area', 'N/A')!r} | "
              f"delay={res.get('delay', 'N/A')!r}", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=n_runs) as ex:
        results = list(ex.map(run_one, range(n_runs)))

    total_elapsed = time.time() - t_start
    print(f"\n⏱  全部完成耗时: {total_elapsed:.0f}s = {total_elapsed/60:.1f}min")

    # 5) 统计
    powers = [r["power"] for r in results
              if r.get("power") is not None and r["power"] != float("inf")]
    areas = [r["area"] for r in results
             if r.get("area") is not None and r["area"] != float("inf")]
    delays = [r["delay"] for r in results
              if r.get("delay") is not None and r["delay"] != float("inf")]

    print(f"\n{'=' * 60}")
    print(f"📊 结果统计 ({len(powers)} 次成功 / {n_runs} 次尝试)")
    print(f"{'=' * 60}")
    if len(powers) >= 2:
        p_m, p_s = statistics.mean(powers), statistics.stdev(powers)
        a_m, a_s = statistics.mean(areas), statistics.stdev(areas)
        d_m, d_s = statistics.mean(delays), statistics.stdev(delays)
        print(f"power: mean={p_m:.6f} mW, std={p_s:.6f} (相对 {p_s/p_m*100:.4f}%)")
        print(f"area : mean={a_m:.3f}     , std={a_s:.4f}     (相对 {a_s/a_m*100:.4f}%)")
        print(f"delay: mean={d_m:.4f} ns , std={d_s:.4f}     (相对 {d_s/d_m*100:.4f}%)")
        print()
        print(f"  全部 power: {[f'{p:.6f}' for p in powers]}")
        print(f"  全部 area : {[f'{a:.3f}'  for a in areas]}")
        print(f"  全部 delay: {[f'{d:.4f}'  for d in delays]}")
        print()

        # 判定
        p_rel = p_s / p_m * 100
        print(f"💡 判定 (power 相对 std = {p_rel:.4f}%):")
        if p_rel < 0.05:
            print(f"   ✅ DC 高度确定性 → 多次平均零收益, 不要做")
        elif p_rel < 0.3:
            print(f"   ⚠️  DC 弱随机 → 3 次平均边际收益小")
        elif p_rel < 1.0:
            print(f"   ✅ DC 显著随机 → 3 次平均能降噪 √3 = 1.7×, 上!")
        else:
            print(f"   🚨 DC 极大随机 → 5 次平均必要, 但要查 EDA 配置")
    else:
        print(f"❌ 成功次数 < 2, 无法统计 std")
        for r in results:
            print(f"   {r}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bit_width", type=int, default=16)
    p.add_argument("--target_delay", type=float, default=2.0)
    p.add_argument("--n_runs", type=int, default=5)
    args = p.parse_args()
    main(**vars(args))
