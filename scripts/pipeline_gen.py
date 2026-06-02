"""真流水线数据采集：每分钟发 N 个 sample，永不等"最慢一个"。

vs generate_dataset.py:
- generate_dataset: batch 同步, 每 batch 等所有完成才下一批 (木桶效应)
- pipeline_gen:    连续发车, 完成 1 个立刻补 1 个 (任意时刻活跃数稳定)

用法:
    python scripts/pipeline_gen.py \\
        --total_samples 5000 \\
        --rate_per_min 2 \\
        --save_path dataset/glitch_power_data_16bit_v2.pt \\
        --save_every_n 20

  rate_per_min=2 + 单样本 13.5min ⇒ 稳态活跃数 ≈ 27 个
  rate_per_min=3 ⇒ 稳态活跃 ≈ 40 个
"""
import os
import sys
import time
import argparse
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用 generate_dataset.py 的辅助函数
from scripts.generate_dataset import compute_graph_hash, extract_X_edge
from send_eda import _evaluate_single_design_full
from trainer.arith_das import CompressorRouting
from utils import CompressorTree, Mul


def _now_ts():
    return time.strftime("%H:%M:%S", time.localtime())


def main(total_samples, rate_per_min, save_path, save_every_n,
         bit_width=16, encode_type="and", target_delay=2.0,
         build_dir="build", max_active=200, tick_seconds=10,
         max_failures=0):

    # ===== 1. 断点续采 + 去重 =====
    dataset = []
    seen_hashes = set()
    if os.path.exists(save_path):
        dataset = torch.load(save_path, map_location="cpu", weights_only=False)
        for item in dataset:
            if "edge_index" in item and "edge_attr" in item:
                seen_hashes.add(compute_graph_hash(item["X"], item["edge_index"], item["edge_attr"]))
        print(f"  📂 断点续采: 已有 {len(dataset)} 样本")

    target_total = len(dataset) + total_samples
    print(f"  🎯 目标: +{total_samples} 个 ⇒ 总 {target_total}")
    print(f"  🚗 发车速率: {rate_per_min} 个/min  (tick={tick_seconds}s)")
    print(f"  ⛽ 单样本基线: 13.5min  ⇒ 稳态活跃数 ≈ {int(rate_per_min * 13.5)} 个")
    print(f"  💾 保存间隔: 每 {save_every_n} 个完成")
    print()

    # ===== 2. 初始化采样环境 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CompressorRouting(
        bit_width=bit_width, encode_type=encode_type, ct_arch="dadda",
        use_ppo_loss=False, ppo_loss_weight=0, use_delay_loss=False, delay_loss_weight=0,
        lse_gamma_val=0.01, use_rule_loss=False, rule_loss_weight=0,
        use_disc_loss=False, disc_loss_weight=0, num_episodes=0,
        num_samples=10, num_epochs=1, log_dir=None, build_dir=build_dir,
        save_freq=999, log_freq=999, device=device, optim_name="Adam", optim_kwargs={"lr": 1e-4},
        scheduler_name="CosineAnnealingLR", scheduler_kwargs={"T_max": 100, "eta_min": 1e-5},
        gcn_kwargs={"input_dim": 7, "hidden_dims_list": [[64]], "output_dim": 64},
        delay_weight=1, area_weight=1, power_weight=1, delay_scale=1, area_scale=1, power_scale=1,
        clip_range=0.2, max_grad_norm=0.5, n_processing=1, reference_point=[1, 1],
        pareto_target=["delay", "area"],
        pool_size=10, rule_loss_wight_incr=0, disc_loss_weight_incr=0,
    )
    env.reset()
    Z_mat_dict = env.get_Z_mat()
    os.makedirs(build_dir, exist_ok=True)

    # ===== 3. 单样本采样函数 =====
    def sample_one(sample_id):
        """采一个 sample，返回 {X, edge_index, edge_attr, rtl_path, graph_hash}"""
        with torch.no_grad():
            for _ in range(20):  # 最多重试 20 次找到未见过的图
                samples_connection, _ = env.sample_from_logits(Z_mat_dict)
                assignment = env.emit_assignment(samples_connection)
                X, edge_index, edge_attr = extract_X_edge(env.comp_graph, samples_connection)
                gh = compute_graph_hash(X, edge_index, edge_attr)
                if gh not in seen_hashes:
                    seen_hashes.add(gh)
                    ct = CompressorTree(env.initial_pp, env.state["ct32"], env.state["ct22"])
                    mul = Mul(env.bit_width, env.encode_type, ct)
                    rtl_path = os.path.join(build_dir, f"MUL_pipe_{sample_id}.v")
                    mul.emit_verilog(rtl_path, assignment=assignment)
                    return {
                        "X": X, "edge_index": edge_index, "edge_attr": edge_attr,
                        "rtl_path": rtl_path, "graph_hash": gh,
                    }
            return None  # 连续 20 次都是重复图，放弃

    # ===== 4. EDA 评估包装 =====
    def eval_one(sample_info):
        try:
            res = _evaluate_single_design_full(
                sample_info["rtl_path"], target_delay, bit_width,
            )
            res["sample_info"] = sample_info
            return res
        finally:
            try: os.remove(sample_info["rtl_path"])
            except OSError: pass

    # ===== 5. 主循环：发车 + 收割 =====
    executor = ThreadPoolExecutor(max_workers=max_active)
    futures = deque()
    submitted = 0
    succeeded = 0
    failed = 0
    sample_fail = 0
    last_save = 0
    tick_quota = rate_per_min * tick_seconds / 60.0  # 每 tick 应发的数量
    quota_carry = 0.0  # 小数累计
    start_time = time.time()

    def harvest_done():
        """非阻塞收割已完成的 future"""
        nonlocal succeeded, failed, last_save
        while futures and futures[0].done():
            f = futures.popleft()
            try:
                res = f.result()
            except Exception as e:
                failed += 1
                continue
            info = res["sample_info"]
            if res.get("power") != float("inf") and res.get("delay") != float("inf"):
                num_nodes = info["X"].shape[0]
                node_powers_arr = torch.zeros(num_nodes, dtype=torch.float32)
                node_power_mask = torch.zeros(num_nodes, dtype=torch.bool)
                for inst_name, p_w in (res.get("node_powers") or {}).items():
                    try:
                        idx = int(inst_name.split("_")[1])
                        if 0 <= idx < num_nodes:
                            node_powers_arr[idx] = float(p_w)
                            node_power_mask[idx] = True
                    except (ValueError, IndexError):
                        pass
                dataset.append({
                    "X": info["X"].clone(),
                    "edge_index": info["edge_index"].clone(),
                    "edge_attr": info["edge_attr"].clone(),
                    "area": res["area"], "delay": res["delay"], "power": res["power"],
                    "node_powers": node_powers_arr,
                    "node_power_mask": node_power_mask,
                })
                succeeded += 1
                # 周期 save
                if succeeded - last_save >= save_every_n:
                    torch.save(dataset, save_path)
                    last_save = succeeded
                    elapsed = (time.time() - start_time) / 60
                    rate = succeeded / max(elapsed, 0.01)
                    active = len(futures)
                    print(f"[{_now_ts()}] 💾 save: 总 {len(dataset)} | 本次成功 {succeeded} 失败 {failed} | "
                          f"活跃 {active} | 速率 {rate:.2f}/min | 已运行 {elapsed:.1f}min")
            else:
                failed += 1
                seen_hashes.discard(info.get("graph_hash"))

    print(f"[{_now_ts()}] 🚀 流水线启动")
    if max_failures > 0:
        print(f"  🛑 自动熔断: EDA 失败累计 > {max_failures} 个 → 立即停止")
    while succeeded + failed < total_samples:
        tick_start = time.time()
        # 1. 收割完成的
        harvest_done()

        # 1.5. 熔断检查
        if max_failures > 0 and failed > max_failures:
            print(f"[{_now_ts()}] 🚨 熔断: EDA 失败 {failed} > {max_failures}, 立即停止")
            torch.save(dataset, save_path)
            executor.shutdown(wait=False, cancel_futures=True)
            print(f"[{_now_ts()}] 💾 已 save {len(dataset)} 样本后退出")
            return

        # 2. 按 rate 发车 (允许小数累计)
        quota_carry += tick_quota
        n_to_submit = int(quota_carry)
        quota_carry -= n_to_submit
        for _ in range(n_to_submit):
            if submitted >= total_samples + 100:  # 上限 (留点失败 buffer)
                break
            sample = sample_one(submitted)
            if sample is None:
                sample_fail += 1
                continue
            f = executor.submit(eval_one, sample)
            futures.append(f)
            submitted += 1

        # 3. 睡到下个 tick
        elapsed = time.time() - tick_start
        if elapsed < tick_seconds:
            time.sleep(tick_seconds - elapsed)

    # ===== 6. 等所有未完成 + 最终 save =====
    print(f"[{_now_ts()}] ⏳ 等剩余 {len(futures)} 个 future 完成...")
    while futures:
        harvest_done()
        if futures:
            time.sleep(5)
    torch.save(dataset, save_path)
    executor.shutdown(wait=True)
    print(f"\n[{_now_ts()}] ✅ 完成: 总 {len(dataset)} | 本次成功 {succeeded} | EDA失败 {failed} | 采样失败 {sample_fail}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--total_samples", type=int, default=5000, help="本次目标新增样本数")
    p.add_argument("--rate_per_min", type=float, default=2.0, help="每分钟发车数")
    p.add_argument("--save_path", default="dataset/glitch_power_data_16bit_v2.pt")
    p.add_argument("--save_every_n", type=int, default=20, help="每完成多少个 save 一次")
    p.add_argument("--max_active", type=int, default=200, help="并发 worker 上限 (安全阀)")
    p.add_argument("--tick_seconds", type=int, default=10, help="主循环 tick 周期 (秒)")
    p.add_argument("--max_failures", type=int, default=0,
                   help="EDA 失败累计超过该数则自动熔断退出 (0=不熔断)")
    p.add_argument("--bit_width", type=int, default=16)
    p.add_argument("--target_delay", type=float, default=2.0)
    p.add_argument("--build_dir", default="build",
                   help="本地 verilog 文件临时目录 (并行多 bit_width 必须分开)")
    args = p.parse_args()
    main(**vars(args))
