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
def extract_X_P(comp_graph, samples_connection):
    """
    核心特征提取器：将 arith-das 的内部图结构转化为代理模型所需的张量 X 和 P

    适配当前 proxy_mlp.py:
      - X: [N, 7]  原始节点特征 (stage_idx, col_idx, idx, type_onehot×4)
                    不做归一化，模型内部的 input_proj + LayerNorm 会处理
      - P: [N, N]  邻接矩阵, P[src, dst]=1 表示 src→dst 有边
                    proxy_mlp.py 中用 P^T @ X 来聚合前驱特征
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

    # -------- 邻接矩阵 P [N, N] --------
    # P[src, dst] = 1 表示 src 的输出连接到 dst 的输入 (src→dst)
    P = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # samples_connection 包含所有采样出的连接边
    for src_idx, dst_idx, dst_conc_type, _ in samples_connection:
        if 0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes:
            P[src_idx, dst_idx] = 1.0

    # 填入 PP 节点的固定输入边 (Partial Products → Stage 0 compressors)
    # PP 节点 (type=2) 连接到同列的 Stage 0 节点
    for src_idx in range(num_nodes):
        src_info = comp_graph.vertex_list[src_idx]
        if src_info[2] == 2:  # PP 节点
            for dst_idx in range(src_idx + 1, num_nodes):
                dst_info = comp_graph.vertex_list[dst_idx]
                if src_info[1] == dst_info[1] and dst_info[0] == 0:
                    P[src_idx, dst_idx] = 1.0

    return X, P


# ========================== 去重工具 ==========================
def compute_graph_hash(X, P):
    """计算图的 MD5 哈希，用于去重"""
    data = torch.cat([X.flatten(), P.flatten()]).numpy().tobytes()
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
            dataset = torch.load(save_path, map_location="cpu")
            for item in dataset:
                h = compute_graph_hash(item["X"], item["P"])
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

                    # 提取张量特征
                    X, P = extract_X_P(env.comp_graph, samples_connection)

                    # 去重检查
                    graph_hash = compute_graph_hash(X, P)
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
                        "P": P,
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
            verilog_files, target_delay, bit_width, max_workers=actual_workers
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
                    "P": info["P"].clone(),
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
        sample_P = dataset[0]["P"]
        print(f"     X shape:      {list(sample_X.shape)} (node_feature_dim={sample_X.shape[1]})")
        print(f"     P shape:      {list(sample_P.shape)} (num_nodes={sample_P.shape[0]})")

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
    # POC: 跑 5000 样本验证 per-node supervision (100 batch × 50/batch)
    # 输出新数据集 *_node_power.pt，不覆盖现有的 *_enriched.pt
    collect_dataset(
        num_batches=100,
        samples_per_batch=50,
        bit_width=16,
        encode_type="and",
        save_path="dataset/glitch_power_data_16bit_node_power.pt",
        target_delay=2.0,
        max_eda_workers=32,
        save_every=2,
        resume=True,
    )
