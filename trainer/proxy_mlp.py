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
      - mean aggregation (按 in/out degree 归一化)
      - 边特征调制: edge_msg = edge_proj(edge_feat) 加到源节点消息上
      - 方案 3: 边类型化 — 6 类边 (2 sum/carry × 3 ports) 各有 type embedding,
        让 sum/carry 链路和 a/b/c 端口走不同的消息空间
    """

    def __init__(self, hidden_dim, dropout=0.1, edge_feat_dim=0, use_mean_agg=True,
                 use_typed_edges=False, num_edge_types=6,
                 type_id_idx=(1, 2, 5)):
        """
        Args:
            use_typed_edges: 启用方案 3 (RGCN 思想 — type embedding 加到 message 上)
            num_edge_types:  边类型总数 (默认 6 = 2 × 3)
            type_id_idx:     (carry_col, port_start_col, port_end_col) 在 edge_feat 中提取 type
                              默认 (1, 2, 5) 对应 [is_sum, is_carry, port_a, port_b, port_c, ...]
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.use_mean_agg = use_mean_agg
        self.use_typed_edges = use_typed_edges
        self.type_id_idx = type_id_idx

        if edge_feat_dim > 0:
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_proj = None

        # 方案 3: 边类型 embedding (sum/carry × port a/b/c = 6 类)
        if use_typed_edges:
            self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)

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

    def _compute_type_id(self, edge_feat):
        """从 edge_feat 提取 type_id ∈ [0, 5]
        type_id = is_carry * 3 + port_idx (其中 port_idx = argmax of port_a/b/c)
        """
        carry_idx, port_start, port_end = self.type_id_idx
        is_carry = edge_feat[:, carry_idx].long()                       # [E]
        port_idx = edge_feat[:, port_start:port_end].argmax(dim=-1)     # [E]
        return is_carry * 3 + port_idx                                  # [E]

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
            # 方案 3: 边类型 embedding 进一步调制 (RGCN 思想)
            if self.use_typed_edges:
                type_id = self._compute_type_id(edge_feat)             # [E]
                edge_msg = edge_msg + self.edge_type_emb(type_id)      # [E, H]
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


class BidirectionalGINLayer(nn.Module):
    """双向 GIN: 将 GIN 的核心思想 (sum agg + 可学 ε + MLP 变换) 拓展到 DAG 双向场景

    GIN 原版: h_new = MLP((1+ε) * h + Σ_{u∈N(v)} h_u)
    本实现: 在前驱和后继两个方向各做一次 GIN-style 更新, 然后融合.
    保留 sum aggregation (GIN 表达力关键), 但允许 edge_feat 调制消息.

    与 BidirectionalGNNLayer 区别:
      - sum 聚合 (GIN 表达力达到 WL test 上界, mean/max 不行)
      - 学 ε 让自身和邻居权重可调 (vs 这里之前的简单拼接)
      - 边特征仍然支持 (h_u + edge_proj(edge_attr))
      - 无 typed-edge embedding (GIN 是纯结构主义, 不用 RGCN 类思想)
    """

    def __init__(self, hidden_dim, dropout=0.1, edge_feat_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim

        if edge_feat_dim > 0:
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.edge_proj = None

        # GIN 的可学 ε (前驱/后继各一个, 让模型自适应两方向权重)
        self.eps_fwd = nn.Parameter(torch.zeros(1))
        self.eps_bwd = nn.Parameter(torch.zeros(1))

        # GIN MLP — 关键: 必须 2 层 + 非线性 (论文证明 1 层 Linear 退化为 GCN)
        # 输入 2H = 前向 GIN 输出 || 后向 GIN 输出
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_feat=None):
        H = self.hidden_dim
        src = edge_index[0]
        dst = edge_index[1]

        # 边消息 (与原 BidirectionalGNNLayer 一致)
        if self.edge_proj is not None and edge_feat is not None:
            edge_msg = self.edge_proj(edge_feat)
            msg_fwd = h[src] + edge_msg
            msg_bwd = h[dst] + edge_msg
        else:
            msg_fwd = h[src]
            msg_bwd = h[dst]

        # GIN sum aggregation (前驱方向: dst ← Σ src)
        h_fwd_agg = torch.zeros_like(h)
        h_fwd_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), msg_fwd)
        # GIN sum aggregation (后继方向: src ← Σ dst)
        h_bwd_agg = torch.zeros_like(h)
        h_bwd_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, H), msg_bwd)

        # GIN 核心更新: (1 + ε) * h + Σ_neighbors
        h_fwd = (1 + self.eps_fwd) * h + h_fwd_agg
        h_bwd = (1 + self.eps_bwd) * h + h_bwd_agg

        # 双向融合 + MLP 变换 (GIN 必须有非线性 MLP 才能达到 WL 表达力)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        h_new = self.mlp(h_cat)

        # 残差 + LayerNorm (训练稳定)
        return self.norm(h + h_new)


class PureGCNLayer(nn.Module):
    """纯 GCN 单层 (Kipf & Welling 2017):
        h^(l+1) = ReLU(D^(-1/2) Ã D^(-1/2) h^(l) W^(l) + b)
    其中 Ã = A + I (自环), D 是 Ã 的度数矩阵.
    DAG 视为无向图 (添加反向边).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h, edge_index):
        n = h.size(0)
        H = h.size(1)

        # 构造无向边 + 自环
        self_loops = torch.arange(n, device=h.device).unsqueeze(0).expand(2, -1)
        rev_edges = edge_index.flip(0)
        edges = torch.cat([edge_index, rev_edges, self_loops], dim=1)
        src, dst = edges[0], edges[1]

        # 度数 (Ã 的度, 含自环和反向边)
        deg = torch.zeros(n, device=h.device, dtype=h.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(src, dtype=h.dtype))
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]  # 对称归一化系数

        # 消息: norm[e] * h[src[e]]
        msg = norm.unsqueeze(-1) * h[src]

        # 聚合
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), msg)

        # 线性变换 + ReLU (论文风格)
        return torch.relu(self.linear(out) + self.bias)


class PureGCN(nn.Module):
    """纯 GCN (Kipf & Welling 2017) — 用于跟 PureGIN 对照.

    架构:
      x → Linear(F, H)
        → GCN layer × K (对称归一化 + 单 Linear + ReLU)
        → mean readout → MLP head → scalar

    与 PureGIN 对比:
      - GCN: D^(-1/2) Ã D^(-1/2) 加权 (mean-style 归一化), 表达力 < WL
      - GIN: sum 聚合 + 可学 ε, 表达力 = WL
      - GCN 视为无向图 (加反向边), GIN 用前驱方向
    """

    def __init__(
        self,
        node_feature_dim=4,
        hidden_dim=96,
        num_gnn_layers=4,
        dropout=0.0,
        **unused_kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            PureGCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_gnn_layers)
        ])

        # Mean readout + 2 层 MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, edge_index, mask, edge_attr=None,
                return_nodes=False, return_aux=False):
        # 忽略 edge_attr / return_nodes / return_aux (GCN 不用边特征)
        B, N, _ = x_node.shape
        H = self.hidden_dim

        x_flat = x_node.reshape(B * N, -1)
        mask_flat = mask.reshape(B * N).float()

        h = self.input_proj(x_flat) * mask_flat.unsqueeze(-1)

        for layer in self.gcn_layers:
            h = layer(h, edge_index)
            h = h * mask_flat.unsqueeze(-1)  # 每层 re-mask

        # Mean pooling readout
        h_3d = h.reshape(B, N, H)
        n_real = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        graph_repr = h_3d.sum(dim=1) / n_real

        return self.head(graph_repr).squeeze(-1)


class PureGINLayer(nn.Module):
    """纯 GIN 单层 (Xu et al. 2019): MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)
    无边特征 / 无双向 / 无残差 / 无归一化 (论文最简形式).
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = nn.Parameter(torch.zeros(1))
        # 论文风格 MLP: 2 层 Linear + ReLU (无 BN 因 batched 含 padding 会失真)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h, edge_index):
        H = self.hidden_dim
        src = edge_index[0]
        dst = edge_index[1]
        # 单向 sum agg: dst ← Σ src
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), h[src])
        # GIN 核心: (1+ε)·h_v + Σ h_u
        out = (1 + self.eps) * h + h_agg
        return self.mlp(out)


class PureGIN(nn.Module):
    """纯 GIN (Xu et al. 2019) — 完全无任何工程加强项, 用于对照实验.

    架构:
      x → Linear(F, H)
        → GIN layer × K (单向 sum + ε + MLP)
        → Multi-scale READOUT (每层 sum, concat 含输入层)
        → MLP head → scalar

    缺失项 (相对当前 ArithProxyGNN):
      ❌ 异构输入投影
      ❌ 边特征 (edge_attr / arrival)
      ❌ 双向消息
      ❌ 残差 + LayerNorm
      ❌ 双池化 (mean+max)
      ❌ extensive(node_head sum) + intensive(global_head) 解耦
      ❌ node_aux_head
    """

    def __init__(
        self,
        node_feature_dim=13,
        hidden_dim=96,
        num_gnn_layers=4,
        dropout=0.0,
        **unused_kwargs,  # 吞掉 ArithProxyGNN 的多余参数避免上层报错
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers

        # 单一共享输入投影 (无异构)
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # K 个 GIN 层
        self.gin_layers = nn.ModuleList([
            PureGINLayer(hidden_dim, dropout) for _ in range(num_gnn_layers)
        ])

        # 多尺度 sum readout: (K+1) × H 维 (含输入层 h_0)
        readout_dim = (num_gnn_layers + 1) * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, edge_index, mask, edge_attr=None,
                return_nodes=False, return_aux=False):
        # edge_attr / return_nodes / return_aux 全部忽略 (纯 GIN 不支持)
        B, N, _ = x_node.shape
        H = self.hidden_dim

        x_flat = x_node.reshape(B * N, -1)
        mask_flat = mask.reshape(B * N).float()

        # 输入投影 + mask padded
        h = self.input_proj(x_flat) * mask_flat.unsqueeze(-1)

        # 收集每层 sum (含 h_0)
        h_per_layer = [h.reshape(B, N, H).sum(dim=1)]  # [B, H]

        for layer in self.gin_layers:
            h = layer(h, edge_index)
            h = h * mask_flat.unsqueeze(-1)  # 每层后 re-mask
            h_per_layer.append(h.reshape(B, N, H).sum(dim=1))

        # Concat multi-scale readout
        graph_repr = torch.cat(h_per_layer, dim=-1)  # [B, (K+1)*H]
        pred = self.head(graph_repr).squeeze(-1)     # [B]
        return pred


class OneHotGINLayer(nn.Module):
    """GIN layer with the original single-direction sum aggregation."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h, edge_index):
        src, dst = edge_index
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.hidden_dim), h[src])
        return self.mlp((1.0 + self.eps) * h + h_agg)


class OneHotGIN(nn.Module):
    """Pure GIN power proxy that only consumes node type one-hot features.

    The input tensor may contain rich features, but this model slices only
    X[:, :, onehot_start:onehot_start + onehot_dim]. It ignores stage/column,
    arrival, edge_attr, physics features, node labels, and auxiliary heads.
    """

    def __init__(
        self,
        node_feature_dim=13,
        hidden_dim=96,
        num_gnn_layers=4,
        dropout=0.0,
        onehot_start=3,
        onehot_dim=4,
        **unused_kwargs,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.onehot_start = onehot_start
        self.onehot_dim = onehot_dim
        self.dropout = dropout

        self.input_proj = nn.Linear(onehot_dim, hidden_dim)
        self.gin_layers = nn.ModuleList([
            OneHotGINLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        readout_dim = (num_gnn_layers + 1) * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, edge_index, mask, edge_attr=None,
                return_nodes=False, return_aux=False):
        del edge_attr, return_nodes, return_aux
        B, N, _ = x_node.shape
        H = self.hidden_dim

        x_onehot = x_node[:, :, self.onehot_start:self.onehot_start + self.onehot_dim]
        x_flat = x_onehot.reshape(B * N, self.onehot_dim)
        mask_flat = mask.reshape(B * N).float()

        h = self.input_proj(x_flat) * mask_flat.unsqueeze(-1)
        h_per_layer = [h.reshape(B, N, H).sum(dim=1)]

        for layer in self.gin_layers:
            h = layer(h, edge_index)
            h = h * mask_flat.unsqueeze(-1)
            h_per_layer.append(h.reshape(B, N, H).sum(dim=1))

        graph_repr = torch.cat(h_per_layer, dim=-1)
        return self.head(graph_repr).squeeze(-1)


class ArithProxyGNN(nn.Module):
    """基于 GNN 的功耗预测代理

    可配置开关 (默认全开)：
      - use_mean_agg:  GNN 用 mean 聚合（按 degree 归一化）替代 sum
      - use_edge_feat: 用 arrival_time 差分作为边特征调制消息
      - arrival_idx:   X 中 arrival_time 所在列；若 X 维度不足则自动降级
      - use_gin:       用 GIN 替代 BidirectionalGNNLayer (sum agg + 学 ε + 纯 MLP)
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
        external_edge_attr_dim=0,    # 外部 edge_attr 的维度 (v2: 5, 方案1后: 10)
        use_typed_edges=False,       # 方案 3: RGCN-style 边类型 embedding
        use_multitask=False,         # 方案 2: 多任务 area+delay head
        use_jk_pool=False,           # 方案 4: JK-Net 多尺度池化 (跨层 h 聚合)
        use_gin=False,               # 替换 backbone 为 BidirectionalGIN
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
        self.use_typed_edges = use_typed_edges
        self.use_multitask = use_multitask
        self.use_jk_pool = use_jk_pool
        self.use_gin = use_gin

        # edge_feat = [arrival_skew, arrival_src] (2) + external_edge_attr (5: sum/carry/port_a/b/c)
        arrival_feat_dim = 2 if self.use_edge_feat else 0
        edge_feat_dim = arrival_feat_dim + external_edge_attr_dim

        # 1) 异构输入投影
        self.input_proj = HeterogeneousProjection(
            node_feature_dim, hidden_dim, num_types=num_node_types
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 2) GNN 主干 — GIN or BidirectionalGNN
        # 方案 3: 边类型化需要知道 edge_attr 中 [is_sum, is_carry, port_a/b/c] 的位置
        # external_edge_attr_dim 拼在 arrival_feat 后面, 所以偏移 arrival_feat_dim
        type_id_idx = (arrival_feat_dim + 1, arrival_feat_dim + 2, arrival_feat_dim + 5)
        if use_gin:
            # GIN 不支持 typed_edges (GIN 是纯结构主义), warn 一下
            if use_typed_edges:
                print("  ⚠️  use_gin=True 与 use_typed_edges=True 同开: GIN 层会忽略 typed_edges")
            self.gnn_layers = nn.ModuleList([
                BidirectionalGINLayer(
                    hidden_dim, dropout,
                    edge_feat_dim=edge_feat_dim,
                )
                for _ in range(num_gnn_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                BidirectionalGNNLayer(
                    hidden_dim, dropout,
                    edge_feat_dim=edge_feat_dim,
                    use_mean_agg=use_mean_agg,
                    use_typed_edges=use_typed_edges and external_edge_attr_dim >= 5,
                    type_id_idx=type_id_idx,
                )
            for _ in range(num_gnn_layers)
        ])

        # 方案 4: JK-Net 多尺度池化 - 学一个 attention 权重融合每层 h
        if use_jk_pool:
            self.jk_attention = nn.Linear(hidden_dim, 1)  # 学一个 layer score


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

        # 方案 2: 多任务 area + delay head (共享 backbone, 不同 head)
        if use_multitask:
            self.area_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.delay_head = nn.Sequential(
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


    def forward(self, x_node, edge_index, mask, edge_attr=None,
                return_nodes=False, return_aux=False):
        """
        Args:
            x_node:     [B, N, F]
            edge_index: [2, E]   含 batch offset 的全局边索引
            mask:       [B, N]   True = 真实节点
            edge_attr:  [E, external_edge_attr_dim] 或 None
            return_nodes: 若 True 额外返回 node_power [B, N]
            return_aux:   方案 2 — 若 True 且 use_multitask 启用, 额外返回 area, delay
        Returns:
            global_power: [B]
            (若 return_nodes) node_power: [B, N]
            (若 return_aux 且 use_multitask) area: [B], delay: [B]
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

        # GNN 主干 (方案 4: JK-Net 收集每层 h 跨层池化)
        h_layers = []
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_feat)
            if self.use_jk_pool:
                h_layers.append(h)

        # 方案 4: JK-Net 多尺度池化 - 用 attention 加权融合每层 h
        if self.use_jk_pool and len(h_layers) > 0:
            h_stacked = torch.stack(h_layers, dim=1)   # [B*N, L, H]
            scores = self.jk_attention(h_stacked).squeeze(-1)  # [B*N, L]
            attn_w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B*N, L, 1]
            h = (h_stacked * attn_w).sum(dim=1)        # [B*N, H]

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
        pred_power = sum_power + global_corr                    # [B]

        # 方案 2: 多任务 area + delay (共享 graph_feat)
        area_pred = None
        delay_pred = None
        if self.use_multitask and return_aux:
            area_pred = self.area_head(graph_feat).squeeze(-1)  # [B]
            delay_pred = self.delay_head(graph_feat).squeeze(-1)  # [B]

        if return_nodes:
            node_aux_flat = self.node_aux_head(h_masked).squeeze(-1)  # [B*N]
            node_aux = node_aux_flat.reshape(B, N) * mask.float()
            if return_aux and self.use_multitask:
                return pred_power, node_aux, area_pred, delay_pred
            return pred_power, node_aux
        if return_aux and self.use_multitask:
            return pred_power, area_pred, delay_pred
        return pred_power


# ========== 向下兼容：保留旧的类名 ==========
# 这样 train_proxy.py 用 `from proxy_mlp import ArithProxyMLP` 还能跑
ArithProxyMLP = ArithProxyGNN