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
    """单层 GNN: 聚合前驱 + 后继 + 自身 → 残差 + LayerNorm"""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
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

    def forward(self, h, edge_index):
        """
        h:          [B*N, H]
        edge_index: [2, E]   edge_index[0]=src, edge_index[1]=dst
        """
        H = self.hidden_dim
        src = edge_index[0]
        dst = edge_index[1]

        # 1) 聚合前驱: dst ← src
        h_pred = torch.zeros_like(h)
        h_pred.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, H),
            h[src],
        )

        # 2) 聚合后继: src ← dst (反向边)
        h_succ = torch.zeros_like(h)
        h_succ.scatter_add_(
            0,
            src.unsqueeze(1).expand(-1, H),
            h[dst],
        )

        # 3) 拼接 + 变换
        h_cat = torch.cat([h, h_pred, h_succ], dim=-1)  # [B*N, 3H]
        h_new = self.msg_mlp(h_cat)

        # 4) 残差 + 归一化
        return self.norm(h + h_new)


class ArithProxyGNN(nn.Module):
    """基于 GNN 的功耗预测代理"""

    def __init__(
        self,
        node_feature_dim=9,
        hidden_dim=64,
        num_gnn_layers=3,
        num_node_types=4,
        dropout=0.1,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_node_types = num_node_types

        # 1) 异构输入投影
        self.input_proj = HeterogeneousProjection(
            node_feature_dim, hidden_dim, num_types=num_node_types
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 2) GNN 主干
        self.gnn_layers = nn.ModuleList([
            BidirectionalGNNLayer(hidden_dim, dropout)
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

    def forward(self, x_node, edge_index, mask):
        """
        Args:
            x_node:     [B, N, F]
            edge_index: [2, E]   含 batch offset 的全局边索引
            mask:       [B, N]   True = 真实节点
        Returns:
            global_power: [B]
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

        # GNN 主干
        for layer in self.gnn_layers:
            h = layer(h, edge_index)

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

        return sum_power + global_corr


# ========== 向下兼容：保留旧的类名 ==========
# 这样 train_proxy.py 用 `from proxy_mlp import ArithProxyMLP` 还能跑
ArithProxyMLP = ArithProxyGNN