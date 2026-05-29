"""
快速验证 extract_X_edge 输出的图结构是否合理 (不跑 EDA, 只测拓扑)。

预期:
  - stage-0 FA 入度 = 3 (vs 旧版 11)
  - stage>0 FA 入度 = 3
  - HA 入度 = 2
  - PP 出度 = 1
  - sum 边占比 ~30-40%, carry 边占比 ~60-70%
  - port_a/b/c one-hot
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_initial_partial_product, CompressorTree
from trainer.arith_das import CompressorGraph, CompressorRouting
from scripts.generate_dataset import extract_X_edge


def analyze_graph(X, edge_index, edge_attr, tag=""):
    N = X.shape[0]
    E = edge_index.shape[1]
    type_idx = X[:, 3:7].argmax(dim=1)
    stage = X[:, 0]

    in_deg = torch.zeros(N, dtype=torch.long)
    in_deg.scatter_add_(0, edge_index[1], torch.ones(E, dtype=torch.long))
    out_deg = torch.zeros(N, dtype=torch.long)
    out_deg.scatter_add_(0, edge_index[0], torch.ones(E, dtype=torch.long))

    s0_fa = (type_idx == 0) & (stage == 0)
    sx_fa = (type_idx == 0) & (stage > 0)
    ha = type_idx == 1
    pp = type_idx == 2
    virt = type_idx == 3

    print(f"\n[{tag}] N={N}, E={E}")
    print(f"  节点类型: FA={int((type_idx==0).sum())}, HA={int(ha.sum())}, "
          f"PP={int(pp.sum())}, virtual={int(virt.sum())}")
    if s0_fa.any():
        print(f"  stage-0 FA 入度: mean={in_deg[s0_fa].float().mean():.2f}, "
              f"max={in_deg[s0_fa].max()}, min={in_deg[s0_fa].min()} (期望=3)")
    if sx_fa.any():
        print(f"  stage>0 FA 入度: mean={in_deg[sx_fa].float().mean():.2f}, "
              f"max={in_deg[sx_fa].max()} (期望=3)")
    if ha.any():
        print(f"  HA      入度: mean={in_deg[ha].float().mean():.2f}, "
              f"max={in_deg[ha].max()} (期望=2)")
    if pp.any():
        print(f"  PP      出度: mean={out_deg[pp].float().mean():.2f}, "
              f"max={out_deg[pp].max()}, min={out_deg[pp].min()} (期望=1)")

    # edge_attr 分布
    if E > 0:
        is_sum = edge_attr[:, 0].sum().item()
        is_carry = edge_attr[:, 1].sum().item()
        port_a = edge_attr[:, 2].sum().item()
        port_b = edge_attr[:, 3].sum().item()
        port_c = edge_attr[:, 4].sum().item()
        print(f"  edge_attr:")
        print(f"    is_sum   = {int(is_sum):4d} ({is_sum/E*100:.1f}%)")
        print(f"    is_carry = {int(is_carry):4d} ({is_carry/E*100:.1f}%)")
        print(f"    port_a   = {int(port_a):4d}")
        print(f"    port_b   = {int(port_b):4d}")
        print(f"    port_c   = {int(port_c):4d}")
        # 健康检查
        assert int(is_sum + is_carry) == E, f"is_sum+is_carry={int(is_sum+is_carry)} != E={E}"
        assert int(port_a + port_b + port_c) == E, "port one-hot 总和异常"


def main():
    pp = get_initial_partial_product(16, "and")
    ct = CompressorTree.dadda(pp)
    assignment = ct.compressor_assignment_fused()
    comp_graph = CompressorGraph(pp, assignment)
    print(f"  CompressorGraph: stage_num={comp_graph.stage_num}, col_num={comp_graph.col_num}")
    print(f"  vertex_list 长度: {len(comp_graph.vertex_list)}")

    # 用 CompressorRouting 采样一次 samples_connection (复用 generate_dataset 的方式)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CompressorRouting(
        bit_width=16, encode_type="and", ct_arch="dadda",
        use_ppo_loss=False, ppo_loss_weight=0, use_delay_loss=False, delay_loss_weight=0,
        lse_gamma_val=0.01, use_rule_loss=False, rule_loss_weight=0,
        use_disc_loss=False, disc_loss_weight=0, num_episodes=0,
        num_samples=3, num_epochs=1, log_dir=None, build_dir="build_validate",
        save_freq=999, log_freq=999, device=device, optim_name="Adam", optim_kwargs={"lr": 1e-4},
        scheduler_name="CosineAnnealingLR", scheduler_kwargs={"T_max": 100, "eta_min": 1e-5},
        gcn_kwargs={"input_dim": 7, "hidden_dims_list": [[64]], "output_dim": 64},
        delay_weight=1, area_weight=1, power_weight=1, delay_scale=1, area_scale=1, power_scale=1,
        clip_range=0.2, max_grad_norm=0.5, n_processing=1, reference_point=[1, 1],
        pareto_target=["delay", "area"],
        pool_size=10, rule_loss_wight_incr=0, disc_loss_weight_incr=0
    )
    env.reset()

    with torch.no_grad():
        Z_mat_dict = env.get_Z_mat()
        for k in range(3):
            samples_connection, _ = env.sample_from_logits(Z_mat_dict)
            X, edge_index, edge_attr = extract_X_edge(env.comp_graph, samples_connection)
            analyze_graph(X, edge_index, edge_attr, tag=f"sample {k}")


if __name__ == "__main__":
    main()
