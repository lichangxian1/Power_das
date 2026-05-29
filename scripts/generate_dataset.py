import os
import sys
import torch
import hashlib
import traceback
from tqdm import tqdm

# 将根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorRouting
from send_eda import evaluate_batch_parallel


# ========================== 特征提取 ==========================
def extract_X_edge(comp_graph, samples_connection):
    """
    将 arith-das 的图结构转化为 GNN 训练所需的稀疏图表示。

    返回:
      X:          [N, 7]  节点特征 (stage_idx, col_idx, idx, type_onehot×4)
      edge_index: [2, E]  有向边 src→dst (信号流方向)
      edge_attr:  [E, 5]  [is_sum, is_carry, port_a, port_b, port_c]
                          is_sum: src 与 dst 在同列 (sum 信号 / PP 输入)
                          is_carry: src.col + 1 = dst.col (carry 信号)
                          port_*: dst 端口 one-hot (a/b/c)

    samples_connection 由 arith_das.sample_from_logits 产生，覆盖 s=0..stage_num,
    s=0 时 src 是 PP 节点, dst 是 stage-0 节点 → 已包含真实 PP→stage0 边。
    """
    num_nodes = len(comp_graph.vertex_list)

    # -------- 节点特征 X [N, 7] --------
    x_features = []
    for vertex_idx in range(num_nodes):
        stage_idx, col_idx, type_idx, idx = comp_graph.vertex_list[vertex_idx]
        type_onehot = [0.0, 0.0, 0.0, 0.0]
        type_onehot[type_idx] = 1.0
        attr = [float(stage_idx), float(col_idx), float(idx)] + type_onehot
        x_features.append(attr)
    X = torch.tensor(x_features, dtype=torch.float32)

    # -------- 边 (edge_index + edge_attr) --------
    src_list, dst_list, attr_list = [], [], []
    for src_idx, dst_idx, dst_connec_type, _ in samples_connection:
        if not (0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes):
            continue
        src_col = comp_graph.vertex_list[src_idx][1]
        dst_col = comp_graph.vertex_list[dst_idx][1]
        is_sum = 1.0 if src_col == dst_col else 0.0
        is_carry = 1.0 if (src_col + 1) == dst_col else 0.0
        # dst_connec_type ∈ {0, 1, 2} → port a/b/c (FA 三端口; HA 用 a/c)
        port = [0.0, 0.0, 0.0]
        if 0 <= dst_connec_type <= 2:
            port[dst_connec_type] = 1.0
        src_list.append(src_idx)
        dst_list.append(dst_idx)
        attr_list.append([is_sum, is_carry] + port)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float32)
    return X, edge_index, edge_attr


# ========================== 去重工具 ==========================
def compute_graph_hash(X, edge_index, edge_attr):
    """计算图的 MD5 哈希，用于去重 (新签名: 基于稀疏边表示)"""
    # edge_index/edge_attr 顺序可能略有差异 → 排序后再 hash
    if edge_index.numel() > 0:
        keys = edge_index[0].long() * (X.shape[0] + 1) + edge_index[1].long()
        order = torch.argsort(keys)
        ei_sorted = edge_index[:, order]
        ea_sorted = edge_attr[order]
        data = X.flatten().numpy().tobytes() \
             + ei_sorted.numpy().tobytes() \
             + ea_sorted.numpy().tobytes()
    else:
        data = X.flatten().numpy().tobytes()
    return hashlib.md5(data).hexdigest()


# ========================== 主函数 ==========================
def collect_dataset(
    num_batches=100,
    samples_per_batch=16,
    bit_width=16,
    encode_type="and",
    save_path="dataset/glitch_power_data.pt",
    target_delay=2.0,
    max_eda_workers=25,
    save_every=5,
    resume=True,
    stagger_waves=1,
    stagger_delay=0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    build_dir = "build_data_gen"
    os.makedirs(build_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- 断点续采 --------
    dataset = []
    seen_hashes = set()
    start_batch = 0

    if resume and os.path.exists(save_path):
        try:
            dataset = torch.load(save_path, map_location="cpu", weights_only=False)
            # 兼容旧格式（含 P 矩阵）和新格式（含 edge_index + edge_attr）
            for item in dataset:
                if "edge_index" in item and "edge_attr" in item:
                    h = compute_graph_hash(item["X"], item["edge_index"], item["edge_attr"])
                else:
                    # 旧格式: 用 X + P 算 hash, 这些样本无法续接新格式 → 直接跳过
                    h = hashlib.md5(item["X"].numpy().tobytes()).hexdigest()
                seen_hashes.add(h)
            start_batch = len(dataset) // max(samples_per_batch // 2, 1)
            print(f"  📂 断点续采: 已有 {len(dataset)} 个样本")
        except Exception as e:
            print(f"  ⚠️ 加载已有数据失败: {e}，将从头开始")
            dataset = []
            seen_hashes = set()

    # -------- 初始化采样环境 --------
    env = CompressorRouting(
        bit_width=bit_width, encode_type=encode_type, ct_arch="dadda",
        use_ppo_loss=False, ppo_loss_weight=0, use_delay_loss=False, delay_loss_weight=0,
        lse_gamma_val=0.01, use_rule_loss=False, rule_loss_weight=0,
        use_disc_loss=False, disc_loss_weight=0, num_episodes=0,
        num_samples=samples_per_batch, num_epochs=1, log_dir=None, build_dir=build_dir,
        save_freq=999, log_freq=999, device=device, optim_name="Adam", optim_kwargs={"lr": 1e-4},
        scheduler_name="CosineAnnealingLR", scheduler_kwargs={"T_max": 100, "eta_min": 1e-5},
        gcn_kwargs={"input_dim": 7, "hidden_dims_list": [[64]], "output_dim": 64},
        delay_weight=1, area_weight=1, power_weight=1, delay_scale=1, area_scale=1, power_scale=1,
        clip_range=0.2, max_grad_norm=0.5, n_processing=1, reference_point=[1, 1],
        pareto_target=["delay", "area"],
        pool_size=10, rule_loss_wight_incr=0, disc_loss_weight_incr=0
    )

    success_count = len(dataset)
    duplicate_count = 0
    fail_count = 0
    sample_fail_count = 0

    total_planned = num_batches * samples_per_batch
    print(f"  🚀 开始离线数据采集")
    print(f"     计划: {num_batches} 批次 × {samples_per_batch} 样本 = {total_planned} 个")
    print(f"     EDA 并发数: {max_eda_workers}")
    print(f"     位宽: {bit_width}, 编码: {encode_type}, 目标延迟: {target_delay} ns")
    print(f"     已有样本: {len(dataset)}")
    print()

    for batch_idx in tqdm(range(start_batch, start_batch + num_batches), desc="Batches"):
        # 1. 环境重置
        env.reset()

        # 2. 批量采样合法连线 + 提取特征 + 导出 Verilog
        sample_info_list = []
        with torch.no_grad():
            Z_mat_dict = env.get_Z_mat()
            for sample_idx in range(samples_per_batch):
                try:
                    samples_connection, _ = env.sample_from_logits(Z_mat_dict)
                    assignment = env.emit_assignment(samples_connection)

                    # 提取张量特征 (新格式: X + edge_index + edge_attr)
                    X, edge_index, edge_attr = extract_X_edge(
                        env.comp_graph, samples_connection,
                    )

                    # 去重检查
                    graph_hash = compute_graph_hash(X, edge_index, edge_attr)
                    if graph_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(graph_hash)

                    # 导出 Verilog
                    ct = CompressorTree(env.initial_pp, env.state["ct32"], env.state["ct22"])
                    mul = Mul(env.bit_width, env.encode_type, ct)
                    rtl_path = os.path.join(build_dir, f"MUL_b{batch_idx}_s{sample_idx}.v")
                    mul.emit_verilog(rtl_path, assignment=assignment)

                    sample_info_list.append({
                        "X": X,
                        "edge_index": edge_index,
                        "edge_attr": edge_attr,
                        "rtl_path": rtl_path,
                        "graph_hash": graph_hash,
                    })
                except Exception as e:
                    sample_fail_count += 1
                    if sample_fail_count <= 10:  # 前10次打印详情
                        tqdm.write(f"  ⚠️ Batch {batch_idx} Sample {sample_idx} 采样失败: {e}")
                    continue

        if len(sample_info_list) == 0:
            continue

        # 3. 并行 EDA 评估
        verilog_files = [info["rtl_path"] for info in sample_info_list]
        actual_workers = min(max_eda_workers, len(verilog_files))
        eda_results = evaluate_batch_parallel(
            verilog_files, target_delay, bit_width, max_workers=actual_workers,
            stagger_waves=stagger_waves, stagger_delay=stagger_delay,
        )

        # 4. 结果对齐与入库
        result_dict = {res["file"]: res for res in eda_results}

        batch_success = 0
        for info in sample_info_list:
            res = result_dict.get(info["rtl_path"])
            if res and res["power"] != float('inf') and res["delay"] != float('inf'):
                # POC: 把 node_powers dict { "ct32_X": power_W } 转成 [N] tensor
                #      节点身份: ct32_X 对应 vertex_idx=X (FA), ct22_X 对应 vertex_idx=X (HA)
                num_nodes = info["X"].shape[0]
                node_powers_arr = torch.zeros(num_nodes, dtype=torch.float32)
                node_power_mask = torch.zeros(num_nodes, dtype=torch.bool)
                for inst_name, p_w in (res.get("node_powers") or {}).items():
                    try:
                        # inst_name like "ct32_45" or "ct22_27"
                        idx = int(inst_name.split("_")[1])
                        if 0 <= idx < num_nodes:
                            node_powers_arr[idx] = float(p_w)
                            node_power_mask[idx] = True
                    except (ValueError, IndexError):
                        pass

                dataset.append({
                    "X": info["X"].clone(),
                    "edge_index": info["edge_index"].clone(),
                    "edge_attr":  info["edge_attr"].clone(),
                    "area":  res["area"],
                    "delay": res["delay"],
                    "power": res["power"],
                    "node_powers": node_powers_arr,        # [N] W (PP/output 节点为 0)
                    "node_power_mask": node_power_mask,    # [N] bool (FA/HA 节点为 True)
                })
                success_count += 1
                batch_success += 1
            else:
                fail_count += 1
                # 评估失败的图允许将来重新采到
                seen_hashes.discard(info.get("graph_hash"))

        # 5. 清理临时 Verilog 文件
        for info in sample_info_list:
            try:
                if os.path.exists(info["rtl_path"]):
                    os.remove(info["rtl_path"])
            except OSError:
                pass

        # 6. 定期保存
        if (batch_idx - start_batch + 1) % save_every == 0:
            torch.save(dataset, save_path)
            tqdm.write(
                f"  💾 Batch {batch_idx+1} | 本批 {batch_success}/{len(sample_info_list)} | "
                f"总计 {success_count} 有效 | 去重 {duplicate_count} | EDA失败 {fail_count} | 采样失败 {sample_fail_count}"
            )

    # -------- 最终保存 --------
    torch.save(dataset, save_path)

    print(f"\n  {'='*60}")
    print(f"  ✅ 数据采集完成!")
    print(f"     有效样本:     {success_count}")
    print(f"     去重跳过:     {duplicate_count}")
    print(f"     EDA 失败:     {fail_count}")
    print(f"     采样异常:     {sample_fail_count}")
    print(f"     保存路径:     {save_path}")

    if len(dataset) > 0:
        powers = [d["power"] for d in dataset]
        delays = [d["delay"] for d in dataset]
        areas  = [d["area"]  for d in dataset]
        print(f"     Power 范围:   [{min(powers):.6f}, {max(powers):.6f}] mW")
        print(f"     Delay 范围:   [{min(delays):.6f}, {max(delays):.6f}] ns")
        print(f"     Area  范围:   [{min(areas):.4f}, {max(areas):.4f}]")

        # 打印特征维度供训练时确认
        sample_X = dataset[0]["X"]
        sample_ei = dataset[0]["edge_index"]
        sample_ea = dataset[0]["edge_attr"]
        print(f"     X shape:          {list(sample_X.shape)} (node_feature_dim={sample_X.shape[1]})")
        print(f"     edge_index shape: {list(sample_ei.shape)} (num_edges={sample_ei.shape[1]})")
        print(f"     edge_attr shape:  {list(sample_ea.shape)} "
              f"(dims: [is_sum, is_carry, port_a, port_b, port_c])")
        # 验证关键统计量
        edge_n = sample_ei.shape[1]
        is_sum = sample_ea[:, 0].sum().item()
        is_carry = sample_ea[:, 1].sum().item()
        print(f"     edges: sum={int(is_sum)} ({is_sum/edge_n*100:.1f}%), "
              f"carry={int(is_carry)} ({is_carry/edge_n*100:.1f}%)")

        # POC: node_powers 统计
        n_with_np = sum(1 for d in dataset if d.get("node_power_mask") is not None
                        and bool(d["node_power_mask"].any()))
        if n_with_np > 0:
            sample_d = next(d for d in dataset
                            if d.get("node_power_mask") is not None and bool(d["node_power_mask"].any()))
            mask = sample_d["node_power_mask"]
            np_arr = sample_d["node_powers"]
            valid_np = np_arr[mask]
            print(f"     node_powers 覆盖: {n_with_np}/{len(dataset)} 样本含 per-node power")
            print(f"     sample[0] FA+HA 数: {int(mask.sum())} / {len(mask)}, "
                  f"node power 范围 [{valid_np.min():.2e}, {valid_np.max():.2e}] W")
    print(f"  {'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--samples_per_batch", type=int, default=50)
    parser.add_argument("--save_path", default="dataset/glitch_power_data_16bit_v2.pt",
                        help="新数据集路径 (v2 = 修复 PP 笛卡尔积 + edge_attr)")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--stagger_waves", type=int, default=1,
                        help="把 batch 内任务切成 N 波依次启动 (避开 vcs spike). 例 2 = 分两波")
    parser.add_argument("--stagger_delay", type=int, default=0,
                        help="波之间间隔秒数. 例 900 = 15 分钟")
    args = parser.parse_args()

    collect_dataset(
        num_batches=args.num_batches,
        samples_per_batch=args.samples_per_batch,
        bit_width=16,
        encode_type="and",
        save_path=args.save_path,
        target_delay=2.0,
        max_eda_workers=args.max_workers,
        save_every=args.save_every,
        resume=True,
        stagger_waves=args.stagger_waves,
        stagger_delay=args.stagger_delay,
    )
