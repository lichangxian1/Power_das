"""GCN 策略网络：多通道残差 GCN（sum/carry/type 三套边各一路卷积）。"""
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ConfigurableGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: List[int],
        out_channels: int,
        activation: Optional[str] = "relu",
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.activation = getattr(F, activation) if activation is not None else None
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        dims = [in_channels] + hidden_dims + [out_channels]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i + 1]))
            if use_layernorm and i < len(dims) - 2:
                self.norms.append(nn.LayerNorm(dims[i + 1]))
            else:
                self.norms.append(None)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                if self.use_layernorm and self.norms[i] is not None:
                    x = self.norms[i](x)
                if self.activation is not None:
                    x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MultiChannelResGCNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        super(MultiChannelResGCNBlock, self).__init__()
        self.gcn_a = ConfigurableGCN(
            input_dim, hidden_dims, output_dim, activation, dropout, use_layernorm
        )
        self.gcn_b = ConfigurableGCN(
            input_dim, hidden_dims, output_dim, activation, dropout, use_layernorm
        )
        self.gcn_c = ConfigurableGCN(
            input_dim, hidden_dims, output_dim, activation, dropout, use_layernorm
        )

        self.dropout = dropout
        self.activation = getattr(F, activation) if activation is not None else None
        self.use_layernorm = use_layernorm

        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else None
        self.linear = nn.Linear(output_dim * 3, output_dim)

        self.res_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x, edge_index_a, edge_index_b, edge_index_c):
        out_a = self.gcn_a(x, edge_index_a)
        out_b = self.gcn_b(x, edge_index_b)
        out_c = self.gcn_c(x, edge_index_c)

        out = torch.cat([out_a, out_b, out_c], dim=-1)
        out = self.linear(out)

        if self.use_layernorm:
            out = self.layernorm(out)

        if self.activation is not None:
            out = self.activation(out)

        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out + self.res_proj(x)


class MultiChannelResGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims_list: List[List[int]],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "tanh",
        use_layernorm: bool = False,
    ):
        super(MultiChannelResGCN, self).__init__()

        self.blocks = nn.ModuleList()
        in_dim = input_dim

        for hidden_dims in hidden_dims_list:
            out_dim = hidden_dims[-1] if hidden_dims else in_dim
            block = MultiChannelResGCNBlock(
                input_dim=in_dim,
                hidden_dims=hidden_dims,
                output_dim=out_dim,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )
            self.blocks.append(block)
            in_dim = out_dim

        # 主干输出维度（最后一个 block 的 out_dim），近似类型头会挂在这上面
        self.embedding_dim = in_dim

        self.fc_a = nn.Linear(in_dim, output_dim)
        self.fc_b = nn.Linear(in_dim, output_dim)
        self.fc_c = nn.Linear(in_dim, output_dim)
        self.fc_d = nn.Linear(in_dim, output_dim)

        self.fc_sum = nn.Linear(in_dim, output_dim)
        self.fc_carry = nn.Linear(in_dim, output_dim)

    def embed(self, x, edge_index_a, edge_index_b, edge_index_c) -> torch.Tensor:
        """跑完主干 block，返回逐节点嵌入（不含 5 个投影头）。"""
        for block in self.blocks:
            x = block(x, edge_index_a, edge_index_b, edge_index_c)
        return x

    def forward(
        self,
        x,
        edge_index_a,
        edge_index_b,
        edge_index_c,
        return_embedding=False,
        return_port_d=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # return_embedding 默认 False -> 返回值与改前完全一致（零回归）
        x = self.embed(x, edge_index_a, edge_index_b, edge_index_c)
        out_a = self.fc_a(x)
        out_b = self.fc_b(x)
        out_c = self.fc_c(x)
        out_d = self.fc_d(x)

        out_sum = self.fc_sum(x)
        out_carry = self.fc_carry(x)
        if return_port_d:
            if return_embedding:
                return out_a, out_b, out_c, out_d, out_sum, out_carry, x
            return out_a, out_b, out_c, out_d, out_sum, out_carry
        if return_embedding:
            return out_a, out_b, out_c, out_sum, out_carry, x
        return out_a, out_b, out_c, out_sum, out_carry
