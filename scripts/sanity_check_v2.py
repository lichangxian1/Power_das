"""
v2 数据 sanity check: 验证新格式的图结构和 edge_attr 是否正确。

期望:
  - 字段齐全: X, edge_index, edge_attr, power, node_powers, node_power_mask
  - edge_attr shape = [E, 5]
  - stage-0 FA 入度 = 3 (不再是 11)
  - stage>0 FA 入度 = 3
  - HA 入度 = 2
  - PP 出度 = 1
  - is_sum + is_carry = 1 (per edge), port_a + port_b + port_c = 1 (per edge)
  - graph_hash 去重正常 (相邻样本不应是同一 hash)
"""
import os
import sys
import torch
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_dataset import compute_graph_hash


def check(d, label):
    print(f"\n=== {label}: {len(d)} samples ===")
    keys = set(d[0].keys())
    print(f"  字段: {sorted(keys)}")
    required = {"X", "edge_index", "edge_attr", "power", "area", "delay",
                "node_powers", "node_power_mask"}
    missing = required - keys
    extras = keys - required
    if missing:
        print(f"  ❌ 缺少字段: {missing}")
    if "P" in extras:
        print(f"  ❌ 仍存在旧 P 字段 (应已删除)")
    if not (missing or "P" in extras):
        print(f"  ✅ 字段齐全且无旧 P")

    # edge_attr shape
    ea0 = d[0]["edge_attr"]
    print(f"  edge_attr[0] shape: {list(ea0.shape)} (期望 [E, 5])")
    assert ea0.shape[1] == 5, "edge_attr 维度异常"

    # 统计所有样本
    n_samples = min(50, len(d))
    s0_fa_in = []
    sx_fa_in = []
    ha_in = []
    pp_out = []
    n_dups = 0
    seen = set()
    is_sum_pct, is_carry_pct = [], []
    port_a_pct, port_b_pct, port_c_pct = [], [], []
    onehot_violations = 0

    for it in d[:n_samples]:
        X = it["X"]; ei = it["edge_index"]; ea = it["edge_attr"]
        N, E = X.shape[0], ei.shape[1]
        if E == 0:
            continue
        type_idx = X[:, 3:7].argmax(dim=1)
        stage = X[:, 0]

        in_deg = torch.zeros(N, dtype=torch.long).scatter_add_(0, ei[1], torch.ones(E, dtype=torch.long))
        out_deg = torch.zeros(N, dtype=torch.long).scatter_add_(0, ei[0], torch.ones(E, dtype=torch.long))

        m_s0 = (type_idx == 0) & (stage == 0)
        m_sx = (type_idx == 0) & (stage > 0)
        m_ha = type_idx == 1
        m_pp = type_idx == 2

        if m_s0.any():
            s0_fa_in.extend(in_deg[m_s0].tolist())
        if m_sx.any():
            sx_fa_in.extend(in_deg[m_sx].tolist())
        if m_ha.any():
            ha_in.extend(in_deg[m_ha].tolist())
        if m_pp.any():
            pp_out.extend(out_deg[m_pp].tolist())

        # edge_attr 比例
        is_sum_pct.append(ea[:, 0].mean().item())
        is_carry_pct.append(ea[:, 1].mean().item())
        port_a_pct.append(ea[:, 2].mean().item())
        port_b_pct.append(ea[:, 3].mean().item())
        port_c_pct.append(ea[:, 4].mean().item())

        # one-hot 自洽
        sum_carry = (ea[:, 0] + ea[:, 1])
        ports = (ea[:, 2] + ea[:, 3] + ea[:, 4])
        if not torch.allclose(sum_carry, torch.ones_like(sum_carry)):
            onehot_violations += 1
        if not torch.allclose(ports, torch.ones_like(ports)):
            onehot_violations += 1

        # hash 去重
        h = compute_graph_hash(X, ei, ea)
        if h in seen:
            n_dups += 1
        seen.add(h)

    def fmt(arr, target):
        if not arr:
            return "(无)"
        t = torch.tensor(arr, dtype=torch.float32)
        return f"mean={t.mean():.2f}, min={int(t.min())}, max={int(t.max())} (期望 {target})"

    print(f"\n  入度/出度 (前 {n_samples} 样本聚合):")
    print(f"    stage-0 FA 入度: {fmt(s0_fa_in, 3)}")
    print(f"    stage>0 FA 入度: {fmt(sx_fa_in, 3)}")
    print(f"    HA      入度:    {fmt(ha_in, 2)}")
    print(f"    PP      出度:    {fmt(pp_out, 1)}")

    print(f"\n  edge_attr 平均占比:")
    print(f"    is_sum:   {sum(is_sum_pct)/len(is_sum_pct)*100:.1f}%")
    print(f"    is_carry: {sum(is_carry_pct)/len(is_carry_pct)*100:.1f}%")
    print(f"    port_a:   {sum(port_a_pct)/len(port_a_pct)*100:.1f}%")
    print(f"    port_b:   {sum(port_b_pct)/len(port_b_pct)*100:.1f}%")
    print(f"    port_c:   {sum(port_c_pct)/len(port_c_pct)*100:.1f}%")

    print(f"\n  one-hot 违规: {onehot_violations} (应=0)")
    print(f"  hash 去重: 重复 {n_dups}/{n_samples}")

    # power 范围
    powers = torch.tensor([it["power"] for it in d])
    print(f"\n  power: mean={powers.mean():.4f}, std={powers.std():.4f}, "
          f"min={powers.min():.4f}, max={powers.max():.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/glitch_power_data_16bit_v2.pt"
    print(f"加载: {path}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    check(d, label=os.path.basename(path))
