"""
基于 GNN 的 Arith 功耗代理模型

架构改进 (相比 PFP-MLP)：
1. 异构节点投影 (HeterogeneousProjection)
   - FA/HA/PP/output 每种类型独立的输入 Linear 权重
   - 捕获类型特定的特征语义
2. 双向消息传递 (BidirectionalGNNLayer)
   - 同时聚合前驱（src→dst）和后继（dst→src）
   - 前驱: 模拟输入信号的影响（产生 glitch）
   - 后继: 模拟 glitch 向下游的传播
3. 多层 GNN backbone (默认 3 层)
   - 每个节点能看到 3-hop 邻居
   - 残差 + LayerNorm 保证训练稳定
4. 图级双池化修正 (mean + max pooling)
   - 提供 intensive 修正项，补充节点求和的不足

输出 = 节点级 sum (extensive) + 图级修正 (intensive)
"""

import torch
import torch.nn as nn


class HeterogeneousProjection(nn.Module):
    """每种节点类型独立的输入投影"""

    def __init__(self, in_dim, out_dim, num_types=4):
        super().__init__()
        self.num_types = num_types
        self.out_dim = out_dim
        self.projs = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_types)
        ])
        self.act = nn.GELU()

    def forward(self, x_flat, types_flat, mask_flat):
        """
        x_flat:     [B*N, F]
        types_flat: [B*N]   long, 节点类型 (0~num_types-1)
        mask_flat:  [B*N]   bool, True = 真实节点
        Returns:    [B*N, out_dim]
        """
        out = torch.zeros(x_flat.size(0), self.out_dim,
                          device=x_flat.device, dtype=x_flat.dtype)
        for t in range(self.num_types):
            sel = (types_flat == t) & mask_flat
            if sel.any():
                out[sel] = self.projs[t](x_flat[sel])
        return self.act(out)


class BidirectionalGNNLayer(nn.Module):
    """单层 GNN: 聚合前驱 + 后继 + 自身 → 残差 + LayerNorm

    支持：
      - mean aggregation (按 in/out degree 归一化)，避免 sum 让 hidden 数量级被 degree 主导
      - 边特征调制 (例如 arrival_skew)：edge_msg = edge_proj(edge_feat) 加到源节点消息上
    """

    def __init__(self, hidden_dim, dropout=0.1, edge_feat_dim=0, use_mean_agg=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.use_mean_agg = use_mean_agg

        if edge_feat_dim > 0:
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_proj = None

        # 三路特征拼接：自身 + 前驱聚合 + 后继聚合
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_feat=None):
        """
        h:          [B*N, H]
        edge_index: [2, E]   edge_index[0]=src, edge_index[1]=dst
        edge_feat:  [E, edge_feat_dim] 或 None
        """
        H = self.hidden_dim
        src = edge_index[0]
        dst = edge_index[1]
        E = src.size(0)
        N_total = h.size(0)

        # 边消息：在 src/dst 节点 hidden 上叠加边特征投影
        if self.edge_proj is not None and edge_feat is not None:
            edge_msg = self.edge_proj(edge_feat)   # [E, H]
            msg_fwd = h[src] + edge_msg            # dst 接收的消息
            msg_bwd = h[dst] + edge_msg            # src 接收的反向消息（共享边特征）
        else:
            msg_fwd = h[src]
            msg_bwd = h[dst]

        # 1) 聚合前驱: dst ← src
        h_pred = torch.zeros_like(h)
        h_pred.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), msg_fwd)

        # 2) 聚合后继: src ← dst (反向边)
        h_succ = torch.zeros_like(h)
        h_succ.scatter_add_(0, src.unsqueeze(1).expand(-1, H), msg_bwd)

        # 3) Mean aggregation：除以 in/out degree，防止 hidden 数量级被 degree 主导
        if self.use_mean_agg:
            ones = torch.ones(E, device=h.device, dtype=h.dtype)
            in_deg = torch.zeros(N_total, device=h.device, dtype=h.dtype).scatter_add_(0, dst, ones)
            out_deg = torch.zeros(N_total, device=h.device, dtype=h.dtype).scatter_add_(0, src, ones)
            h_pred = h_pred / in_deg.clamp_min(1.0).unsqueeze(-1)
            h_succ = h_succ / out_deg.clamp_min(1.0).unsqueeze(-1)

        # 4) 拼接 + 变换
        h_cat = torch.cat([h, h_pred, h_succ], dim=-1)  # [B*N, 3H]
        h_new = self.msg_mlp(h_cat)

        # 5) 残差 + 归一化
        return self.norm(h + h_new)


class ArithProxyGNN(nn.Module):
    """基于 GNN 的功耗预测代理

    可配置开关 (默认全开)：
      - use_mean_agg:  GNN 用 mean 聚合（按 degree 归一化）替代 sum
      - use_edge_feat: 用 arrival_time 差分作为边特征调制消息
      - arrival_idx:   X 中 arrival_time 所在列；若 X 维度不足则自动降级
    """

    def __init__(
        self,
        node_feature_dim=9,
        hidden_dim=64,
        num_gnn_layers=3,
        num_node_types=4,
        dropout=0.1,
        use_mean_agg=True,
        use_edge_feat=True,
        arrival_idx=7,
        external_edge_attr_dim=0,    # 外部 edge_attr 的维度 (v2 数据: 5 [is_sum, is_carry, port_a/b/c])
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_node_types = num_node_types

        # 当 X 维度不足以提供 arrival_time 时自动关掉 arrival-based edge_feat
        self.use_edge_feat = use_edge_feat and (node_feature_dim > arrival_idx)
        self.arrival_idx = arrival_idx
        self.use_mean_agg = use_mean_agg
        self.external_edge_attr_dim = external_edge_attr_dim

        # edge_feat = [arrival_skew, arrival_src] (2) + external_edge_attr (5: sum/carry/port_a/b/c)
        arrival_feat_dim = 2 if self.use_edge_feat else 0
        edge_feat_dim = arrival_feat_dim + external_edge_attr_dim

        # 1) 异构输入投影
        self.input_proj = HeterogeneousProjection(
            node_feature_dim, hidden_dim, num_types=num_node_types
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 2) GNN 主干
        self.gnn_layers = nn.ModuleList([
            BidirectionalGNNLayer(
                hidden_dim, dropout,
                edge_feat_dim=edge_feat_dim,
                use_mean_agg=use_mean_agg,
            )
            for _ in range(num_gnn_layers)
        ])

        # 3) 节点级预测头 (extensive: sum 出全图功耗的主体)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 4) 图级修正头 (intensive: mean + max 池化做 correction)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 5) 节点级辅助预测头 (Route D: per-FA power supervision)
        # 独立于 node_head，避免节点级标签的尺度强行扭曲 global sum
        self.node_aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )


    def forward(self, x_node, edge_index, mask, edge_attr=None, return_nodes=False):
        """
        Args:
            x_node:     [B, N, F]
            edge_index: [2, E]   含 batch offset 的全局边索引
            mask:       [B, N]   True = 真实节点
            edge_attr:  [E, external_edge_attr_dim] 或 None
                          v2 数据: [is_sum, is_carry, port_a, port_b, port_c]
            return_nodes: 若 True 额外返回 node_power [B, N]
        Returns:
            global_power: [B]
            (若 return_nodes) node_power: [B, N]
        """
        B, N, _ = x_node.shape
        H = self.hidden_dim

        # 摊平 batch
        x_flat = x_node.reshape(B * N, -1)            # [B*N, F]
        mask_flat = mask.reshape(B * N)               # [B*N]
        types_flat = x_flat[:, 3:7].argmax(dim=-1)    # [B*N]

        # 异构输入投影
        h = self.input_proj(x_flat, types_flat, mask_flat)  # [B*N, H]
        h = self.input_norm(h)

        # 边特征 = arrival 相关 (内部算) + external edge_attr (来自 dataset)
        edge_feat_parts = []
        if self.use_edge_feat:
            src = edge_index[0]
            dst = edge_index[1]
            arrival_flat = x_flat[:, self.arrival_idx]            # [B*N]
            edge_skew = arrival_flat[src] - arrival_flat[dst]     # [E]
            edge_src_at = arrival_flat[src]                       # [E]
            edge_feat_parts.append(torch.stack([edge_skew, edge_src_at], dim=-1))
        if self.external_edge_attr_dim > 0 and edge_attr is not None:
            edge_feat_parts.append(edge_attr)
        edge_feat = torch.cat(edge_feat_parts, dim=-1) if edge_feat_parts else None

        # GNN 主干
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_feat)

        # ===== 节点级预测 (extensive part) =====
        mask_f = mask_flat.unsqueeze(-1).float()      # [B*N, 1]
        h_masked = h * mask_f                          # 屏蔽 padded
        node_power_flat = self.node_head(h_masked).squeeze(-1)  # [B*N]
        node_power = node_power_flat.reshape(B, N) * mask.float()
        sum_power = node_power.sum(dim=1)              # [B]

        # ===== 图级修正 (intensive part) =====
        h_3d = h_masked.reshape(B, N, H)
        n_real = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)  # [B, 1]
        mean_pool = h_3d.sum(dim=1) / n_real           # [B, H]

        # max pooling 需要把 padded 位置压到 -inf
        h_for_max = h_3d.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pool = h_for_max.max(dim=1)[0]             # [B, H]
        # 防止极端情况 (整个 batch 都被 masked) 出 inf
        max_pool = torch.where(
            torch.isinf(max_pool),
            torch.zeros_like(max_pool),
            max_pool,
        )

        graph_feat = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]
        global_corr = self.global_head(graph_feat).squeeze(-1)  # [B]

        if return_nodes:
            # 用独立的 aux head 预测 per-node power（不影响 global 求和路径）
            node_aux_flat = self.node_aux_head(h_masked).squeeze(-1)  # [B*N]
            node_aux = node_aux_flat.reshape(B, N) * mask.float()
            return sum_power + global_corr, node_aux
        return sum_power + global_corr


# ========== 向下兼容：保留旧的类名 ==========
# 这样 train_proxy.py 用 `from proxy_mlp import ArithProxyMLP` 还能跑
ArithProxyMLP = ArithProxyGNN