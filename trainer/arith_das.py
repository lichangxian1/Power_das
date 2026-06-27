import os
import sys
from typing import Dict, List, Tuple, Set, Any, Tuple, Optional, Callable
import random
import copy
import time
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm
import networkx as nx
import multiprocessing
from pygmo import hypervolume
from paretoset import paretoset
import numpy as np
import matplotlib.pyplot as plt


from utils import (
    get_initial_partial_product,
    CompressorTree,
    Mul,
    get_full_target_delay,
    get_target_delay,
    lse_gamma,
    convert_to_serializable,
    BoundedParetoPool,
)

try:
    from .power_proxy import PowerProxyPredictor
except ImportError:
    from power_proxy import PowerProxyPredictor


def get_masked_logits(logits: torch.Tensor, mask: torch.Tensor):
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    return masked_logits


def masked_column_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    logits_masked = logits.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(logits_masked, dim=dim)

    probs = torch.where(mask.any(dim=dim, keepdim=True), probs, torch.zeros_like(probs))

    return probs


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

        self.fc_sum = nn.Linear(in_dim, output_dim)
        self.fc_carry = nn.Linear(in_dim, output_dim)

    def embed(self, x, edge_index_a, edge_index_b, edge_index_c) -> torch.Tensor:
        """跑完主干 block，返回逐节点嵌入（不含 5 个投影头）。"""
        for block in self.blocks:
            x = block(x, edge_index_a, edge_index_b, edge_index_c)
        return x

    def forward(
        self, x, edge_index_a, edge_index_b, edge_index_c, return_embedding=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # return_embedding 默认 False -> 返回值与改前完全一致（零回归）
        x = self.embed(x, edge_index_a, edge_index_b, edge_index_c)
        out_a = self.fc_a(x)
        out_b = self.fc_b(x)
        out_c = self.fc_c(x)

        out_sum = self.fc_sum(x)
        out_carry = self.fc_carry(x)
        if return_embedding:
            return out_a, out_b, out_c, out_sum, out_carry, x
        return out_a, out_b, out_c, out_sum, out_carry


class CompressorGraph:
    def __init__(
        self,
        pp: np.ndarray,
        assignment: List[List[Tuple]],
    ):
        self.assignment = assignment
        self.pp = pp

        self.stage_num = len(assignment)
        self.col_num = len(assignment[0])
        self.vertex_list = []
        self.indice_map = {}

        remain_pp = np.zeros_like(pp, dtype=int)
        ct32 = np.zeros_like(pp, dtype=int)
        ct22 = np.zeros_like(pp, dtype=int)
        dec_ct32 = np.zeros((self.stage_num, self.col_num), dtype=int)
        dec_ct22 = np.zeros((self.stage_num, self.col_num), dtype=int)

        for s in range(self.stage_num):
            for c in range(self.col_num):
                for vertex_info in assignment[s][c]:
                    _, _, type_idx, _ = vertex_info
                    if type_idx == 0:
                        ct32[c] += 1
                        dec_ct32[s, c] += 1
                    elif type_idx == 1:
                        ct22[c] += 1
                        dec_ct22[s, c] += 1
                    else:
                        raise ValueError
        carry_num = 0
        for c in range(self.col_num):
            remain_pp[c] = pp[c] + carry_num - 2 * ct32[c] - ct22[c]
            carry_num = ct32[c] + ct22[c]
        logging.info(f"remain_pp\n: {remain_pp}")

        self.remain_pp = remain_pp
        self.dec_ct32 = dec_ct32
        self.dec_ct22 = dec_ct22
        self.ct32 = ct32
        self.ct22 = ct22
        self.slice_size = np.zeros((self.stage_num + 1, self.col_num), dtype=int)
        self.slice_size[0, :] = pp
        for s in range(1, self.stage_num + 1):
            self.slice_size[s, 0] = (
                self.slice_size[s - 1, 0] - dec_ct32[s - 1, 0] * 2 - dec_ct22[s - 1, 0]
            )
            for c in range(1, self.col_num):
                self.slice_size[s, c] = (
                    self.slice_size[s - 1, c]
                    - dec_ct32[s - 1, c] * 2
                    - dec_ct22[s - 1, c]
                    + dec_ct32[s - 1, c - 1]
                    + dec_ct22[s - 1, c - 1]
                )
        self.port_size = np.zeros((self.stage_num + 1, self.col_num), dtype=int)
        for s in range(self.stage_num):
            for c in range(self.col_num):
                self.port_size[s, c] = 3 * dec_ct32[s, c] + 2 * dec_ct22[s, c]
        self.virtual_node_num = self.slice_size - self.port_size

        self.pp_indices = []
        self.col_offset_map = {}
        self.col_stage_offset_map = {}

        self.slice_indice_map: Dict[Tuple, List] = {}
        vertex_idx = 0
        for c in range(self.col_num):
            self.col_offset_map[c] = vertex_idx
            self.slice_indice_map[(-1, c)] = []
            for pp_idx in range(pp[c]):
                vertex_info = (-1, c, 2, pp_idx)
                self.vertex_list.append(vertex_info)
                self.indice_map[vertex_info] = vertex_idx
                self.pp_indices.append(vertex_idx)
                self.slice_indice_map[(-1, c)].append(vertex_idx)
                vertex_idx += 1
            for s in range(self.stage_num + 1):
                self.slice_indice_map[(s, c)] = []
                self.col_stage_offset_map[(s, c)] = vertex_idx
                if s < self.stage_num:
                    for vertex_info in assignment[s][c]:
                        self.vertex_list.append(vertex_info)
                        self.indice_map[vertex_info] = vertex_idx
                        self.slice_indice_map[(s, c)].append(vertex_idx)
                        vertex_idx += 1
                for visual_idx in range(self.virtual_node_num[s, c]):
                    vertex_info = (s, c, 3, visual_idx)
                    self.vertex_list.append(vertex_info)
                    self.indice_map[vertex_info] = vertex_idx
                    self.slice_indice_map[(s, c)].append(vertex_idx)
                    vertex_idx += 1
        pass

    def to_graph(self):
        edge_index_a = []
        edge_index_b = []
        edge_index_c = []
        x = []
        num_nodes = len(self.vertex_list)

        for vertex_idx in range(num_nodes):
            vertex_info = self.vertex_list[vertex_idx]
            stage_idx, col_idx, type_idx, idx = vertex_info
            type_onehot = np.zeros(4)
            type_onehot[type_idx] = 1
            vertex_attr = np.concatenate(
                [np.array([stage_idx, col_idx, idx]), type_onehot], axis=0
            )
            vertex_attr = torch.tensor(vertex_attr, dtype=torch.float32)
            x.append(vertex_attr)

        def __add_edge_index(src_idx, dst_idx, dst_type_idx):
            if dst_type_idx == 0:
                edge_index_a.append((src_idx, dst_idx))
                edge_index_b.append((src_idx, dst_idx))
                edge_index_c.append((src_idx, dst_idx))
            elif dst_type_idx == 1:
                edge_index_a.append((src_idx, dst_idx))
                edge_index_b.append((src_idx, dst_idx))
            elif dst_type_idx == 3:
                edge_index_a.append((src_idx, dst_idx))
            else:
                raise ValueError("Invalid type index")

        for src_idx in range(num_nodes):
            src_info = self.vertex_list[src_idx]
            src_stage_idx, src_col_idx, src_type_idx, _ = src_info
            if src_type_idx == 2:
                for dst_idx in range(src_idx + 1, num_nodes):
                    dst_info = self.vertex_list[dst_idx]
                    dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
                    if src_col_idx == dst_col_idx and dst_stage_idx == 0:
                        __add_edge_index(src_idx, dst_idx, dst_type_idx)
            else:
                if stage_idx < self.stage_num - 1:
                    for dst_idx in range(src_idx + 1, num_nodes):
                        dst_info = self.vertex_list[dst_idx]
                        dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
                        if (
                            src_col_idx == dst_col_idx
                            and src_stage_idx + 1 == dst_stage_idx
                        ):
                            __add_edge_index(src_idx, dst_idx, dst_type_idx)
                    if col_idx < self.col_num - 1 and src_type_idx != 3:
                        for dst_idx in range(src_idx + 1, num_nodes):
                            dst_info = self.vertex_list[dst_idx]
                            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
                            if (
                                src_stage_idx + 1 == dst_stage_idx
                                and src_col_idx + 1 == dst_col_idx
                            ):
                                __add_edge_index(src_idx, dst_idx, dst_type_idx)
        edge_index_a = torch.tensor(edge_index_a, dtype=torch.long).t().contiguous()
        edge_index_b = torch.tensor(edge_index_b, dtype=torch.long).t().contiguous()
        edge_index_c = torch.tensor(edge_index_c, dtype=torch.long).t().contiguous()
        x = torch.stack(x, dim=0)

        edge_index_a = to_undirected(edge_index_a)
        edge_index_b = to_undirected(edge_index_b)
        edge_index_c = to_undirected(edge_index_c)
        edge_index_a = add_self_loops(edge_index_a)[0]
        edge_index_b = add_self_loops(edge_index_b)[0]
        edge_index_c = add_self_loops(edge_index_c)[0]

        return x, edge_index_a, edge_index_b, edge_index_c

    def get_slice_sum_mask(self, s, c) -> torch.Tensor:
        src_indices = self.slice_indice_map[(s - 1, c)]
        dst_indices = self.slice_indice_map[(s, c)]
        mask = torch.full(
            (3, len(src_indices), len(dst_indices)), True, dtype=torch.bool
        )
        for local_dst_idx, dst_idx in enumerate(dst_indices):
            dst_info = self.vertex_list[dst_idx]
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            if dst_type_idx == 0:
                pass
            elif dst_type_idx == 1:
                mask[2, :, local_dst_idx] = False
            elif dst_type_idx == 3:
                mask[1:, :, local_dst_idx] = False
            else:
                raise ValueError
        return mask

    def get_slice_carry_mask(self, s, c) -> torch.Tensor:
        src_indices = self.slice_indice_map[(s - 1, c - 1)]
        dst_indices = self.slice_indice_map[(s, c)]
        mask = torch.full(
            (3, len(src_indices), len(dst_indices)), True, dtype=torch.bool
        )
        for local_dst_idx, dst_idx in enumerate(dst_indices):
            dst_info = self.vertex_list[dst_idx]
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            if dst_type_idx == 0:
                pass
            elif dst_type_idx == 1:
                mask[2, :, local_dst_idx] = False
            elif dst_type_idx == 3:
                mask[1:, :, local_dst_idx] = False
            else:
                raise ValueError

        for local_src_idx, src_idx in enumerate(src_indices):
            src_info = self.vertex_list[src_idx]
            src_stage_idx, src_col_idx, src_type_idx, _ = src_info
            if src_type_idx == 0 or src_type_idx == 1:
                pass
            else:
                mask[:, local_src_idx, :] = False
        return mask


class CompressorRouting:
    def __init__(
        self,
        bit_width,
        encode_type,
        ct_arch,
        use_ppo_loss,
        ppo_loss_weight,
        use_delay_loss,
        delay_loss_weight,
        lse_gamma_val,
        use_rule_loss,
        rule_loss_weight,
        use_disc_loss,
        disc_loss_weight,
        num_episodes,
        num_samples,
        num_epochs,
        log_dir,
        build_dir,
        save_freq,
        log_freq,
        device,
        optim_name,
        optim_kwargs,
        scheduler_name,
        scheduler_kwargs,
        gcn_kwargs,
        delay_weight,
        area_weight,
        power_weight,
        delay_scale,
        area_scale,
        power_scale,
        clip_range,
        max_grad_norm,
        n_processing,
        reference_point,
        pareto_target,
        pool_size,
        rule_loss_wight_incr,
        disc_loss_weight_incr,
        use_power_proxy=False,
        power_proxy_ckpt=None,
        power_proxy_lib_path="library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib",
        power_proxy_fa_cell="FA1D0BWP12T40P140",
        power_proxy_ha_cell="HA1D0BWP12T40P140",
        power_proxy_output_scale=1e-3,
        fixed_target_delay=None,
        area_budget=None,
        area_violation_weight=2.0,
        delay_violation_weight=2.0,
        power_source=None,
        gomil_path=None,
        synth="openroad",
        # ===== 阶段3 Phase B：近似压缩器类型搜索（全部默认关，关时行为不变）=====
        use_approx_types=False,
        approx_lib_path="Appr_Comp/selected_compressors.json",
        approx_library_path="Appr_Comp/library.json",
        approx_max_col=6,
        # 误差 reward（约束式 A）。med/bias 用 LSB 绝对单位（跨位宽稳定、梯度 O(1)）；
        # NMED=med/maxprod 仅用于上报。budget/weight 都以 LSB 计。
        med_budget=None,
        med_violation_weight=0.1,
        bias_weight=0.0,
        # 误差项归一尺度（点2）：把 med/bias 的 LSB 绝对值除以 error_scale，使 err_term
        # 落到和 PPA(~O(1)) 同量级。默认 1.0 = 不归一（旧行为，向后兼容）。
        error_scale=1.0,
        # 误差作为普通目标项（和 area/power 同评估）：error_as_metric=True 时 get_objective
        # 用 error_weight*med/error_scale 线性计入目标（像 area_weight*area/area_scale），
        # 不再用 med_budget 铰链 max(0,med-budget)；此模式下 med_budget/med_violation_weight 忽略。
        error_as_metric=False,
        error_weight=0.0,
        # 类型头初始化偏向 exact(index0)：>0 时给 type_head bias[0] 设此正偏置 → 初始 P(exact)
        # ≈exp(b)/(exp(b)+N-1)（如 4.0→~0.9）。让策略从"近全 exact"起步、按需加近似 cell，
        # 避免冷启动 ~85% 节点随机近似导致前期 obj 爆高、PPA 梯度被淹。默认 0=旧行为。
        exact_init_bias=0.0,
        # 保底候选：每轮额外评估一个同 routing、全 exact cell 的设计，只参与 best/日志，
        # 不进 PPO loss。用于避免类型采样把 found_best 拖到比纯截断同 routing 更差。
        inject_exact_candidate=False,
        # error_scale 跨-k 归一模式（仅 error_as_metric 用，解决"不同 k 的 MED 差几个数量级、
        # 固定 error_scale 没法用一个 error_weight 通吃"）：
        #   "fixed"  = 用传入的 error_scale 常数（旧行为）；
        #   "pow2k"  = error_scale=2^(trunc_cols-1)（闭式，med/scale 跨 k ~3.7×）；
        #   "sqrt2k" = error_scale=√k·2^(k-1)（闭式，更平，med/scale 跨 k ~1.3×；floor∝std(Δ)）；
        #   "floor"  = error_scale=截断 MED floor self._trunc_med（精确、各 k 归一后 floor 处=1）。
        # pow2k/floor 下 med/error_scale≈O(1) 对所有 k → 单一 error_weight 跨 k 行为一致。
        error_scale_mode="fixed",
        # 可微误差 surrogate（D2 开关）
        use_error_loss=False,
        error_loss_weight=0.0,
        bias_loss_weight=0.0,
        # ④ 尾部/WCE 约束（默认关）：wce_bound = Σ maxe·2^col（误差可加上界，LSB），
        # 控制最坏情况误差、治重尾(RMSE/MAE≫1)。budget/weight 同 med 口径（LSB, /error_scale）。
        # wce_budget=None 或两个 weight=0 时该项不生效（回归字节级一致）。
        wce_budget=None,
        wce_violation_weight=0.0,
        wce_loss_weight=0.0,
        # 误差闸门来源（codex 审过）：
        #   "analytic"  = 解析 proxy（三角不等式估计，实测系统性低估真实 MED 0–30%；默认，向后兼容）
        #   "verilator" = 每个候选用 verilator MC（circular-wrap 真实 MED）测 med/bias 当软罚闸门。
        # 只影响 get_objective 的离散打分（reward 闸门）；可微 error_loss 仍用解析（不可微分 verilator）。
        # WCE 始终用解析上界（MC 尾部不收敛、随 N 单调增长，不可信）。
        error_gate="analytic",
        error_gate_vectors=16_000_000,
        # Phase C ①：低列截断 + 学习校正（默认关）。trunc_cols=k → 最低 k 列的 PP 用常数
        # （校正常数 C=round(E[Δ])，拆成低列槽位的常数 1 位）驱动而非 a&b；压缩树/布线不变，
        # DC 常数传播删低列逻辑＝截断 PPA 收益。误差项 −E[Δ]+C 进 _analytic_error。
        trunc_cols=0,
        trunc_correct="bias",
        # P0(codex)：delay 作为约束而非奖励项。开启后 delay≤delay_target_ns 不奖励也不罚，
        # delay>target 才按 delay_violation_weight 罚 → 优化预算全给 area/power（释放 slack）。
        # 默认关=旧线性奖励 delay_weight·delay。
        delay_as_constraint=False,
        delay_target_ns=None,   # None → 用 fixed_target_delay（DC 时钟周期）当阈值
        # advantage 归一（点1）：A=-(obj-mean)/(std+eps)。默认 False = 旧行为 A=-obj。
        normalize_advantage=False,
        **kwargs,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.ct_arch = ct_arch

        self.lse_gamma_val = lse_gamma_val
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.build_dir = build_dir
        self.device = device
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.n_processing = n_processing

        self.delay_weight = delay_weight
        self.area_weight = area_weight
        self.power_weight = power_weight
        self.delay_scale = delay_scale
        self.area_scale = area_scale
        self.power_scale = power_scale

        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.reference_point = reference_point
        self.use_delay_loss = use_delay_loss
        self.use_rule_loss = use_rule_loss
        self.use_disc_loss = use_disc_loss
        self.delay_loss_weight = delay_loss_weight
        self.rule_loss_weight = rule_loss_weight
        self.disc_loss_weight = disc_loss_weight
        self.pareto_target = pareto_target

        self.use_ppo_loss = use_ppo_loss
        self.ppo_loss_weight = ppo_loss_weight

        self.pool_size = pool_size
        self.rule_loss_weight_incr = rule_loss_wight_incr
        self.disc_loss_weight_incr = disc_loss_weight_incr

        self.gomil_path = gomil_path
        self.synth = synth
        self.kwargs = kwargs
        self.fixed_target_delay = fixed_target_delay
        self.area_budget = area_budget
        self.area_violation_weight = area_violation_weight
        self.delay_violation_weight = delay_violation_weight
        if power_source is None:
            power_source = "proxy" if use_power_proxy else "eda"
        if power_source not in {"proxy", "eda"}:
            raise ValueError(
                f"Invalid power_source={power_source!r}; expected 'proxy' or 'eda'"
            )
        self.power_source = power_source
        self.power_proxy_output_scale = power_proxy_output_scale
        self.power_proxy = None
        if self.power_source == "proxy":
            if power_proxy_ckpt is None:
                raise ValueError("power_source='proxy' requires power_proxy_ckpt")
            self.power_proxy = PowerProxyPredictor(
                ckpt_path=power_proxy_ckpt,
                device=device,
                lib_path=power_proxy_lib_path,
                fa_cell=power_proxy_fa_cell,
                ha_cell=power_proxy_ha_cell,
            )

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.tb_logger = SummaryWriter(self.log_dir)
        else:
            self.tb_logger = None

        self.gnn_a = None
        self.gnn_b = None
        self.gnn_c = None

        self.gcn_kwargs = gcn_kwargs
        self.gcn = MultiChannelResGCN(**gcn_kwargs)
        self.gcn.to(device)

        # ===== Phase B：近似类型搜索状态 =====
        self.use_approx_types = use_approx_types
        self.approx_max_col = approx_max_col
        self.med_budget = med_budget
        self.med_violation_weight = med_violation_weight
        self.error_gate = error_gate
        self.error_gate_vectors = int(error_gate_vectors)
        self.bias_weight = bias_weight
        self.error_scale = error_scale
        self.error_as_metric = bool(error_as_metric)
        self.error_weight = error_weight
        self.inject_exact_candidate = bool(inject_exact_candidate)
        self.error_scale_mode = error_scale_mode
        self.use_error_loss = use_error_loss
        self.error_loss_weight = error_loss_weight
        self.bias_loss_weight = bias_loss_weight
        # ④ 尾部/WCE 约束
        self.wce_budget = wce_budget
        self.wce_violation_weight = wce_violation_weight
        self.wce_loss_weight = wce_loss_weight
        # Phase C ①：截断（_setup_truncation 在 _start_reset 拿到 initial_pp 后填充）
        self.trunc_cols = int(trunc_cols or 0)
        self.trunc_correct = trunc_correct
        self._trunc_bits = {}      # {col: 该列常数 1 的个数}
        self._trunc_const = 0      # 实际注入的校正常数 C
        self._trunc_delta = 0.0    # E[Δ]（截断期望丢失值）
        self._trunc_wce = 0.0      # 截断最坏情况误差 max(C, Δmax−C)
        self._trunc_med = 0.0      # E[|C−Δ|]（截断残差 MED，P=1/4 一阶估计）
        # P0：delay 约束化
        self.delay_as_constraint = bool(delay_as_constraint)
        self.delay_target_ns = delay_target_ns
        self.normalize_advantage = normalize_advantage
        self.type_table_32 = None
        self.type_table_22 = None
        self.type_head_32 = None
        self.type_head_22 = None
        self.approx_module_src_by_name = {}
        self._node_emb = None  # get_Z_mat 设置：逐节点嵌入（类型头 + ppo 复用）
        if self.use_approx_types:
            self._load_approx_types(approx_lib_path, approx_library_path)
            self.type_head_32 = nn.Linear(
                self.gcn.embedding_dim, len(self.type_table_32)
            ).to(device)
            self.type_head_22 = nn.Linear(
                self.gcn.embedding_dim, len(self.type_table_22)
            ).to(device)
            # 类型头初始化偏向 exact(index0)：给 bias[0] 设正偏置，使初始策略≈"近全 exact"、
            # 按需再加近似 cell（而非随机≈均匀→~85% 节点一上来就近似）。默认 0=旧行为。
            if exact_init_bias:
                import math as _m
                with torch.no_grad():
                    self.type_head_32.bias[0] = float(exact_init_bias)
                    self.type_head_22.bias[0] = float(exact_init_bias)
                eb = _m.exp(float(exact_init_bias))
                logging.info(
                    "[approx] type_head exact-init bias=%.2f -> 初始 P(exact)≈%.2f(T32)/%.2f(T22)",
                    exact_init_bias, eb / (eb + len(self.type_table_32) - 1),
                    eb / (eb + len(self.type_table_22) - 1),
                )
            self._bias32 = torch.tensor(
                [e["bias"] for e in self.type_table_32], device=device
            )
            self._wae32 = torch.tensor(
                [e["wae"] for e in self.type_table_32], device=device
            )
            self._bias22 = torch.tensor(
                [e["bias"] for e in self.type_table_22], device=device
            )
            self._wae22 = torch.tensor(
                [e["wae"] for e in self.type_table_22], device=device
            )
            # ④ 每 cell 最坏误差 maxe（exact=0），供 WCE 上界 Σ maxe·2^col。
            # 注意 JSON 里 maxe 是 int → 必须显式 float32，否则 p(Float)@maxes(Long) 报 dtype 错。
            self._maxe32 = torch.tensor(
                [float(e.get("maxe", 0.0)) for e in self.type_table_32],
                device=device, dtype=torch.float32,
            )
            self._maxe22 = torch.tensor(
                [float(e.get("maxe", 0.0)) for e in self.type_table_22],
                device=device, dtype=torch.float32,
            )
            logging.info(
                "[approx] type heads on: T32=%d T22=%d, max_col=%d, "
                "med_budget(LSB)=%s, use_error_loss=%s, wce_budget(LSB)=%s",
                len(self.type_table_32), len(self.type_table_22),
                self.approx_max_col, self.med_budget, self.use_error_loss,
                self.wce_budget,
            )

        opt_params = list(self.gcn.parameters())
        if self.use_approx_types:
            opt_params += list(self.type_head_32.parameters())
            opt_params += list(self.type_head_22.parameters())
        self.optim: optim.Optimizer = getattr(optim, optim_name)(
            opt_params, **optim_kwargs
        )
        self.scheduler: optim.lr_scheduler.LRScheduler = getattr(
            optim.lr_scheduler, scheduler_name
        )(self.optim, **scheduler_kwargs)

        self.comp_graph: CompressorGraph = None
        self.state: Dict[str, np.ndarray] = None
        self.assignment = None

        self.found_best_info = {
            "objective": float("inf"),
            "simulated_result": None,
            "connection": None,
            "assignment": None,
            "ct": None,
            "cell_types": None,  # Phase B：最优设计的每槽 cell 类型，导出时复原近似 cell
        }

        self.total_epoch_num = 0

        self.initial_pp: np.ndarray = None

        if pool_size > 0:
            self.pool = BoundedParetoPool(pool_size)
        else:
            self.pool = None

        self._start_reset()

    # ===================== Phase B：近似类型搜索辅助 =====================
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_path(self, p):
        """相对路径按仓库根解析（pipeline 会 chdir 到 output 目录）。"""
        return p if os.path.isabs(p) else os.path.join(self._REPO_ROOT, p)

    def _load_approx_types(self, sel_path, lib_path):
        """从 selected_compressors.json + library.json 构建类型表（index 0 = exact）。"""
        import itertools

        sel = json.load(open(self._resolve_path(sel_path)))["selected"]
        lib = json.load(open(self._resolve_path(lib_path)))["cells"]
        order32 = [
            "comp32_exact", "comp32_apx_pos_1", "comp32_apx_pos_2", "comp32_apx_pos_3",
            "comp32_apx_neg_1", "comp32_apx_neg_2", "comp32_apx_neg_3",
        ]
        order22 = [
            "comp22_exact", "comp22_apx_pos_1", "comp22_apx_pos_2",
            "comp22_apx_neg_1", "comp22_apx_neg_2",
        ]
        self.type_table_32 = [dict(sel[k]) for k in order32 if k in sel]
        self.type_table_22 = [dict(sel[k]) for k in order22 if k in sel]
        assert self.type_table_32[0]["group"] == "exact", "T32[0] 必须是 exact"
        assert self.type_table_22[0]["group"] == "exact", "T22[0] 必须是 exact"

        # 预生成每个近似 cell 的可综合 SOP module（LUT 取自 library.json）
        APPR = os.path.join(self._REPO_ROOT, "Appr_Comp")
        if APPR not in sys.path:
            sys.path.insert(0, APPR)
        from gen_verilog import emit_module

        pat3 = ["".join(p) for p in itertools.product("01", repeat=3)]
        pat2 = ["".join(p) for p in itertools.product("01", repeat=2)]
        for entry in self.type_table_32 + self.type_table_22:
            name = entry["name"]
            if entry["group"] == "exact":
                continue  # exact 用内置 FA/HA，无需追加 module
            cell = lib[name]
            if entry["type"] == "32":
                src = emit_module(name, ["a", "b", "cin"], pat3,
                                  cell["sum_lut"], cell["carry_lut"],
                                  f"{name} bias={entry['bias']:+.3f}")
            else:
                src = emit_module(name, ["a", "cin"], pat2,
                                  cell["sum_lut"], cell["carry_lut"],
                                  f"{name} bias={entry['bias']:+.3f}")
            self.approx_module_src_by_name[name] = src

    def sample_cell_types(self):
        """对每个压缩器节点采样 cell 类型。返回 (cell_map, type_choices, type_log_prob)。

        cell_map: {node_idx -> module名}（仅非 exact）；type_choices: {node_idx -> (t,k)}。
        col >= approx_max_col 的节点强制 exact。需 self._node_emb（get_Z_mat 已设）。
        """
        cell_map, type_choices = {}, {}
        total_log_prob = 0.0
        if not self.use_approx_types:
            return cell_map, type_choices, total_log_prob
        emb = self._node_emb
        for node_idx, info in enumerate(self.comp_graph.vertex_list):
            _, c, t, _ = info
            if t == 0:
                head, table = self.type_head_32, self.type_table_32
            elif t == 1:
                head, table = self.type_head_22, self.type_table_22
            else:
                continue
            logits = self._masked_type_logits(head(emb[node_idx]), c)
            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()
            total_log_prob += dist.log_prob(sample).item()
            k = sample.item()
            type_choices[node_idx] = (t, k)
            if k != 0:
                cell_map[node_idx] = table[k]["name"]
        return cell_map, type_choices, total_log_prob

    def _approx_modules_src(self, cell_map):
        if not cell_map:
            return ""
        used = sorted(set(cell_map.values()))
        body = "".join(self.approx_module_src_by_name[n] for n in used)
        return "\n// ===== approximate compressor cells =====\n" + body

    def _cell_map_from_types(self, cell_types):
        """从 {node_idx:(t,k)} 复原 {node_idx:module名}（k=0/exact 不收）。
        要求 comp_graph 与采样时同序（同一 assignment 重建即一致）。"""
        cell_map = {}
        for node_idx, tk in (cell_types or {}).items():
            t, k = tk
            if k != 0:
                table = self.type_table_32 if t == 0 else self.type_table_22
                cell_map[int(node_idx)] = table[k]["name"]
        return cell_map

    def _masked_type_logits(self, logits, col):
        """col >= approx_max_col 或 col < trunc_cols 时只留 exact(index 0)，其余置 -1e9。
        截断列被常数驱动、cell 会被 DC 删掉，故不在那放近似 cell。
        用 masked_fill（非 in-place，autograd 安全）。"""
        if col >= self.approx_max_col or col < self.trunc_cols:
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[0] = False
            logits = logits.masked_fill(mask, -1e9)
        return logits

    def _setup_truncation(self):
        """Phase C ①：算截断 [0,k) 的校正常数 C（用低列槽位的常数 1 位表示）+ 误差量。

        E[Δ] = Σ_{c<k} 0.25·pp[c]·2^c   （AND, P=1/4；截断丢失值期望，恒正→负偏置）
        C = round(E[Δ])（trunc_correct='bias'），贪心用 col<k 的槽位表示（列 c 有 pp[c] 个槽、权重 2^c）
        Δmax = Σ_{c<k} pp[c]·2^c，WCE_trunc = max(C, Δmax−C)。"""
        k = self.trunc_cols
        if not (0 <= k <= len(self.initial_pp)):
            raise ValueError(
                f"trunc_cols={k} 越界，应在 [0, {len(self.initial_pp)}]（截断列数）"
            )
        pp = [int(x) for x in self.initial_pp]
        e_delta = sum(0.25 * pp[c] * (1 << c) for c in range(k))
        delta_max = sum(pp[c] * (1 << c) for c in range(k))
        c_target = int(round(e_delta)) if self.trunc_correct == "bias" else 0
        bits, remaining = {}, c_target
        for c in range(k - 1, -1, -1):          # 高列→低列贪心填常数 1
            w = 1 << c
            m = min(pp[c], remaining // w)
            if m > 0:
                bits[c] = m
                remaining -= m * w
        c_actual = c_target - remaining          # 实际可表示的 C（余量通常 0）
        # 截断残差 MED = E[|C−Δ|]，Δ = Σ_{i+j<k} 2^(i+j)·a_i·b_j（被丢的低列加权值）。
        # PP 共享 a/b 位 → 各列高度强相关，独立卷积会低估约 30%（codex review 指出，
        # k=8 独立=142 vs 真实=200）。故用 MC 直接采 a,b 算 Δ，捕捉相关性。均匀输入口径，
        # 仍是一阶估计（⑤ 接 SAIF 真实逐位概率后再精化）；零 bias≠零 MED 必须显式建模。
        rng = np.random.default_rng(0)
        N = 200000
        av = rng.integers(0, 1 << k, size=N, dtype=np.int64)
        bv = rng.integers(0, 1 << k, size=N, dtype=np.int64)
        delta = np.zeros(N, dtype=np.int64)
        for i in range(k):
            ai = (av >> i) & 1
            for j in range(k - i):
                delta += (ai & ((bv >> j) & 1)) << (i + j)
        trunc_med = float(np.abs(c_actual - delta).mean())
        self._trunc_bits = bits
        self._trunc_const = c_actual
        self._trunc_delta = e_delta
        self._trunc_wce = max(c_actual, delta_max - c_actual)
        self._trunc_med = trunc_med
        logging.info(
            "[trunc] cols<%d const-driven; E[Δ]=%.2f C=%d(target %d) Δmax=%d "
            "MED_trunc=%.2f WCE_trunc=%d bits=%s",
            k, e_delta, c_actual, c_target, delta_max, trunc_med, self._trunc_wce, bits,
        )
        # 跨-k 归一 error_scale（仅 error_as_metric）：让 med/error_scale≈O(1) 对所有 k，
        # 从而单一 error_weight 跨 k 行为一致（见 error_scale_mode 注释）。
        if self.error_as_metric and self.error_scale_mode != "fixed" and k > 0:
            if self.error_scale_mode == "pow2k":
                self.error_scale = float(1 << (k - 1))
            elif self.error_scale_mode == "sqrt2k":
                # √k·2^(k-1)：截断 MED floor ∝ std(Δ) ∝ √k·2^(k-1)，比 pow2k 跨 k 更平
                # （med/scale 跨 k 仅 ~1.3× vs pow2k ~3.7×）。
                self.error_scale = (float(k) ** 0.5) * float(1 << (k - 1))
            elif self.error_scale_mode == "floor":
                self.error_scale = max(float(self._trunc_med), 1.0)
            else:
                raise ValueError(f"未知 error_scale_mode={self.error_scale_mode!r}")
            logging.info(
                "[trunc] error_scale_mode=%s -> error_scale=%.2f "
                "(归一后 floor/scale=%.3f)",
                self.error_scale_mode, self.error_scale, trunc_med / self.error_scale,
            )

    def _analytic_error(self, type_choices):
        """从采样类型闭式估计 (med_lsb, abs_bias_lsb, nmed, wce_lsb)。

        med_lsb = Σ wae·2^col   —— MED 的保守上界（输出 LSB 绝对单位，跨位宽稳定）
        abs_bias_lsb = |Σ bias·2^col|  —— 带符号误差的绝对值（抓正负抵消）
        nmed = med_lsb / maxprod  —— 标准 NMED，仅用于上报
        wce_lsb = Σ maxe·2^col  —— ④ WCE 可加上界（最坏情况输出误差，LSB；与传播无关恒成立）
        bias/wae 为 P=1/4 一阶估计；maxe 与概率无关（逐 cell 最坏），故 WCE 上界精确。"""
        bias_total = 0.0
        wae_total = 0.0
        wce_total = 0.0
        for node_idx, (t, k) in type_choices.items():
            if k == 0:
                continue
            table = self.type_table_32 if t == 0 else self.type_table_22
            entry = table[k]
            col = self.comp_graph.vertex_list[node_idx][1]
            w = float(1 << col)
            bias_total += entry["bias"] * w
            wae_total += entry["wae"] * w
            wce_total += entry.get("maxe", 0.0) * w
        # Phase C ①：截断的确定性误差。−E[Δ]+C 为净偏置（bias 项会驱动 cell 抵消残差）；
        # MED_trunc=E[|C−Δ|] 进 MED 上界（三角不等式：MED_total ≤ MED_trunc + Σ wae·2^col，
        # 否则纯截断设计解析 MED=0 会骗过 med_budget）；WCE_trunc 进尾部上界（与 ④ 同口径）。
        if self.trunc_cols > 0:
            bias_total += (-self._trunc_delta + self._trunc_const)
            wae_total += self._trunc_med
            wce_total += self._trunc_wce
        maxprod = float((2 ** self.bit_width - 1) ** 2)
        return wae_total, abs(bias_total), wae_total / maxprod, wce_total

    def get_full_target_delay_result(self):
        build_dir = self.build_dir + "_full_ppa"
        rtl_path = os.path.join(build_dir, "MUL.v")
        if self.synth in ("openroad", "dc"):
            if self.fixed_target_delay is not None:
                full_target_delay = [self.fixed_target_delay]
            else:
                full_target_delay = get_full_target_delay(self.bit_width)
        else:
            raise NotImplementedError
        n_full_target_delay_processing = self.kwargs.get(
            "n_full_target_delay_processing", self.n_processing
        )
        os.makedirs(build_dir, exist_ok=True)

        ct = CompressorTree(self.initial_pp, self.state["ct32"], self.state["ct22"])
        if self.trunc_cols > 0:               # ① full-target-delay 诊断/导出也必须带截断
            ct.trunc_cols = self.trunc_cols
            ct.trunc_bits = self._trunc_bits
        mul = Mul(self.bit_width, self.encode_type, ct)

        # Phase B：用最优设计的近似 cell 评 full-target-delay PPA（否则退化成精确）
        cell_map = self._cell_map_from_types(self.found_best_info.get("cell_types"))
        assignment = self.emit_assignment(
            self.found_best_info["connection"], cell_map=cell_map
        )
        mul.emit_verilog(
            rtl_path,
            assignment=assignment,
            extra_modules_src=self._approx_modules_src(cell_map),
        )
        if self.synth == "dc":
            # 与训练奖励同源：full-target-delay 诊断也走远端 DC 直出
            simulated_result = []
            for td in full_target_delay:
                one = CompressorRouting._dc_simulate_one(
                    self.bit_width, rtl_path, build_dir, td, 0
                )
                if one is None:
                    one = mul.simulate(
                        build_dir, rtl_path, [td], synth="openroad"
                    )[0]
                simulated_result.append(one)
        else:
            simulated_result = mul.simulate(
                build_dir,
                rtl_path,
                full_target_delay,
                n_processing=n_full_target_delay_processing,
            )
        simulated_result = self._apply_power_proxy_to_results(
            simulated_result,
            self.found_best_info["connection"],
        )
        return simulated_result

    def get_full_target_delay_pareto(self, simulated_result, target=["delay", "power"]):
        value_0_list = [item[target[0]] for item in simulated_result]
        value_1_list = [item[target[1]] for item in simulated_result]

        points = np.asarray(list(zip(value_0_list, value_1_list)))
        pareto_indices = paretoset(points, sense=["min", "min"])
        pareto_points = points[pareto_indices]
        return pareto_points

    def save_experiment(self, episode_idx):
        logging.info(f"saving experiment at episode {episode_idx}")
        save_dir = os.path.join(self.log_dir, f"save_iter{episode_idx}")
        os.makedirs(save_dir, exist_ok=True)
        gcn_save_path = os.path.join(save_dir, "gcn.pth")
        torch.save(self.gcn.state_dict(), gcn_save_path)
        # Phase B：类型头不在 gcn 内，单独存，否则 resume/重载会丢失已学的类型策略
        if self.use_approx_types:
            torch.save(
                {
                    "type_head_32": self.type_head_32.state_dict(),
                    "type_head_22": self.type_head_22.state_dict(),
                },
                os.path.join(save_dir, "type_heads.pth"),
            )
        with open(os.path.join(save_dir, "best_info.json"), "w") as f:
            json.dump(
                self.found_best_info, f, indent=4, default=convert_to_serializable
            )

        self.state = self.found_best_info["ct"]
        self.assignment = self.found_best_info["assignment"]
        pp = self.initial_pp
        self.comp_graph = CompressorGraph(pp, self.assignment)

        logging.info(f"testing full target delay at episode {episode_idx}")
        simulated_result = self.get_full_target_delay_result()
        pareto_points = self.get_full_target_delay_pareto(
            simulated_result, self.pareto_target
        )
        pareto_value_0 = [point[0] for point in pareto_points]
        pareto_value_1 = [point[1] for point in pareto_points]
        hv = hypervolume(pareto_points)
        try:
            if self.area_budget is None:
                # Unconstrained (EDA) mode mirrors Arith-DAS: fixed reference point.
                ref = list(self.reference_point)
            else:
                # Ensure reference point strictly dominates all pareto points
                ref = [
                    max(self.reference_point[i], max(p[i] for p in pareto_points) * 1.1)
                    for i in range(len(self.reference_point))
                ]
            hv_value = hv.compute(ref)
            self.tb_logger.add_scalar("hv_value", hv_value, episode_idx)
        except Exception as e:
            logging.error(f"Error computing hypervolume: {e}")
            hv_value = None
        with open(os.path.join(save_dir, "pareto.json"), "w") as f:
            json.dump(
                {
                    "hv_value": hv_value,
                    "pareto_target": self.pareto_target,
                    "pareto_value_0": pareto_value_0,
                    "pareto_value_1": pareto_value_1,
                },
                f,
                indent=4,
            )

        fig = plt.figure()
        pareto_value_0 = np.array(pareto_value_0)
        pareto_value_1 = np.array(pareto_value_1)
        sorted_indices = np.argsort(pareto_value_0)
        pareto_value_0 = pareto_value_0[sorted_indices]
        pareto_value_1 = pareto_value_1[sorted_indices]
        simulated_value_0 = np.asarray(
            [item[self.pareto_target[0]] for item in simulated_result]
        )
        simulated_value_1 = np.asarray(
            [item[self.pareto_target[1]] for item in simulated_result]
        )
        plt.scatter(
            simulated_value_0, simulated_value_1, label="Simulated Result", alpha=0.5
        )
        plt.plot(pareto_value_0, pareto_value_1, "--o", label="Pareto Front")
        plt.xlabel(self.pareto_target[0])
        plt.ylabel(self.pareto_target[1])
        plt.legend()
        self.tb_logger.add_figure("pareto_front", fig, episode_idx)

    def run_experiment(self):
        for episode_idx in range(self.num_episodes):
            self.run_episode(episode_idx)
            if (episode_idx + 1) % self.save_freq == 0:
                self.save_experiment(episode_idx)
        self.save_experiment(self.num_episodes)

    DELAY_CONSTANT = {
        "FA": {
            "Tas": 0.0986,
            "Tac": 0.0491,
            "Tbs": 0.0882,
            "Tbc": 0.0596,
            "Tcs": 0.1019,
            "Tcc": 0.0521,
        },
        "HA": {
            "Tas": 0.0489,
            "Tac": 0.0226,
            "Tbs": 0.0450,
            "Tbc": 0.0213,
        },
    }

    def get_delay_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ):
        max_delay = 0.0
        time_start = time.time()
        slice_delay_dict = {}
        for c in range(self.comp_graph.col_num):
            slice_delay_dict[(-1, c)] = {
                "s": torch.zeros((self.comp_graph.pp[c], 1), device=self.device),
                "c": torch.zeros((self.comp_graph.pp[c], 1), device=self.device),
            }
        out_delay_list = []
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            if c == 0:
                Z_a = Z_mat_slice["sa"]
                Z_b = Z_mat_slice["sb"]
                Z_c = Z_mat_slice["sc"]
                last_slice_delay = slice_delay_dict[(s - 1, c)]["s"]
            else:
                Z_a = torch.cat([Z_mat_slice["sa"], Z_mat_slice["ca"]], dim=0)
                Z_b = torch.cat([Z_mat_slice["sb"], Z_mat_slice["cb"]], dim=0)
                Z_c = torch.cat([Z_mat_slice["sc"], Z_mat_slice["cc"]], dim=0)
                last_slice_delay = torch.cat(
                    [
                        slice_delay_dict[(s - 1, c)]["s"],
                        slice_delay_dict[(s - 1, c - 1)]["c"],
                    ],
                    dim=0,
                )
            Z = torch.cat([Z_a, Z_b, Z_c], dim=1)
            mask = Z > -1e6
            p = torch.softmax(Z, dim=0).masked_fill(~mask, 0.0)
            permutated_delay = p.T @ last_slice_delay

            slice_indices = self.comp_graph.slice_indice_map[(s, c)]
            node_num = len(slice_indices)

            sum_delay = torch.zeros((node_num, 1), device=self.device)
            carry_delay = torch.zeros((node_num, 1), device=self.device)

            a_delay = permutated_delay[:node_num, :]
            b_delay = permutated_delay[node_num : 2 * node_num, :]
            c_delay = permutated_delay[2 * node_num :, :]
            for local_idx, node_idx in enumerate(slice_indices):
                node_info = self.comp_graph.vertex_list[node_idx]
                node_stage_idx, node_col_idx, node_type_idx, _ = node_info
                if node_type_idx == 0:
                    sum_delay[local_idx, :] = lse_gamma(
                        torch.cat(
                            [
                                self.DELAY_CONSTANT["FA"]["Tas"]
                                + a_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["FA"]["Tbs"]
                                + b_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["FA"]["Tcs"]
                                + c_delay[local_idx, :].flatten(),
                            ]
                        ),
                        self.lse_gamma_val,
                    )
                    carry_delay[local_idx, :] = lse_gamma(
                        torch.cat(
                            [
                                self.DELAY_CONSTANT["FA"]["Tac"]
                                + a_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["FA"]["Tbc"]
                                + b_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["FA"]["Tcc"]
                                + c_delay[local_idx, :].flatten(),
                            ]
                        ),
                        self.lse_gamma_val,
                    )
                elif node_type_idx == 1:
                    assert c_delay[local_idx, :].item() == 0.0
                    sum_delay[local_idx, :] = lse_gamma(
                        torch.cat(
                            [
                                self.DELAY_CONSTANT["HA"]["Tas"]
                                + a_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["HA"]["Tbs"]
                                + b_delay[local_idx, :].flatten(),
                            ]
                        ),
                        self.lse_gamma_val,
                    )
                    carry_delay[local_idx, :] = lse_gamma(
                        torch.cat(
                            [
                                self.DELAY_CONSTANT["HA"]["Tac"]
                                + a_delay[local_idx, :].flatten(),
                                self.DELAY_CONSTANT["HA"]["Tbc"]
                                + b_delay[local_idx, :].flatten(),
                            ]
                        ),
                        self.lse_gamma_val,
                    )
                elif node_type_idx == 3:
                    assert c_delay[local_idx, :].item() == 0.0
                    assert b_delay[local_idx, :].item() == 0.0
                    sum_delay[local_idx, :] = a_delay[local_idx, :].flatten()
                    carry_delay[local_idx, :] = a_delay[local_idx, :].flatten()
                else:
                    raise ValueError("Invalid node type index")
            if s == self.comp_graph.stage_num:
                out_delay_list.append(sum_delay.reshape(-1))
            slice_delay_dict[(s, c)] = {
                "s": sum_delay,
                "c": carry_delay,
            }
        max_delay = lse_gamma(torch.cat(out_delay_list), self.lse_gamma_val)

        time_end = time.time()
        return max_delay

    def get_rule_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        l = 0.0
        time_start = time.time()
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            if c == 0:
                Z_a = Z_mat_slice["sa"]
                Z_b = Z_mat_slice["sb"]
                Z_c = Z_mat_slice["sc"]
            else:
                Z_a = torch.cat([Z_mat_slice["sa"], Z_mat_slice["ca"]], dim=0)
                Z_b = torch.cat([Z_mat_slice["sb"], Z_mat_slice["cb"]], dim=0)
                Z_c = torch.cat([Z_mat_slice["sc"], Z_mat_slice["cc"]], dim=0)
            Z = torch.cat([Z_a, Z_b, Z_c], dim=1)
            mask = Z > -1e6
            p = torch.softmax(Z, dim=0).masked_fill(~mask, 0.0)
            row_sum = torch.sum(p, dim=1)
            row_sum_target = (torch.sum(mask.float(), dim=1) > 0).float()
            l += torch.sum(torch.pow(row_sum - row_sum_target, 2))

        time_end = time.time()
        return l

    def get_discrete_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        l = 0.0
        time_start = time.time()
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            if c == 0:
                Z_a = Z_mat_slice["sa"]
                Z_b = Z_mat_slice["sb"]
                Z_c = Z_mat_slice["sc"]
            else:
                Z_a = torch.cat([Z_mat_slice["sa"], Z_mat_slice["ca"]], dim=0)
                Z_b = torch.cat([Z_mat_slice["sb"], Z_mat_slice["cb"]], dim=0)
                Z_c = torch.cat([Z_mat_slice["sc"], Z_mat_slice["cc"]], dim=0)
            Z = torch.cat([Z_a, Z_b, Z_c], dim=1)
            mask = Z > -1e6
            p = torch.softmax(Z, dim=0).masked_fill(~mask, 0.0)
            l += torch.sum(torch.pow((p * (1 - p)), 2))
        time_end = time.time()
        return l

    def get_error_loss(self) -> torch.Tensor:
        """D2（可微误差 surrogate，由 use_error_loss 开关）：每个压缩器节点用 softmax
        类型分布算期望 bias/wae，按列权重 2^col 求和（LSB 绝对单位），返回
        bias_loss_weight·|Σ bias·2^col| + error_loss_weight·relu(Σ wae·2^col − med_budget)。
        用 LSB 单位（非 /maxprod）保证梯度 O(1)、不随位宽消失。需 self._node_emb（含梯度）。"""
        emb = self._node_emb
        bias_total = torch.zeros((), device=self.device)
        med_total = torch.zeros((), device=self.device)
        wce_total = torch.zeros((), device=self.device)  # ④ 期望 WCE 上界
        for node_idx, info in enumerate(self.comp_graph.vertex_list):
            _, c, t, _ = info
            if t == 0:
                head, biases, waes, maxes = (
                    self.type_head_32, self._bias32, self._wae32, self._maxe32
                )
            elif t == 1:
                head, biases, waes, maxes = (
                    self.type_head_22, self._bias22, self._wae22, self._maxe22
                )
            else:
                continue
            logits = self._masked_type_logits(head(emb[node_idx]), c)
            p = torch.softmax(logits, dim=0)
            w = float(1 << c)
            bias_total = bias_total + (p @ biases) * w
            med_total = med_total + (p @ waes) * w
            wce_total = wce_total + (p @ maxes) * w
        # Phase C ①：与 _analytic_error 同口径加入截断的确定性误差（对参数为常数偏移，但
        # 改变 |bias| 的零点→驱动 cell bias 趋向 +Δ−C 抵消残差，而非自身趋零；MED/WCE 偏移
        # 影响 relu 的越界判定）。漏掉会让可微 surrogate 与 reward 口径不一致、梯度方向相反。
        if self.trunc_cols > 0:
            bias_total = bias_total + (-self._trunc_delta + self._trunc_const)
            med_total = med_total + self._trunc_med
            wce_total = wce_total + self._trunc_wce
        # 点2：与 get_objective 同尺度，除 error_scale 落到 O(1)。
        l = self.bias_loss_weight * bias_total.abs() / self.error_scale
        budget = 0.0 if self.med_budget is None else self.med_budget
        l = l + self.error_loss_weight * torch.relu(med_total - budget) / self.error_scale
        # ④ 尾部/WCE 可微 surrogate（默认关）：塑形类型分布远离大 maxe cell。
        if self.wce_loss_weight and self.wce_budget is not None:
            l = l + self.wce_loss_weight * torch.relu(
                wce_total - self.wce_budget
            ) / self.error_scale
        return l

    def get_cache(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ):
        mask_cache: Dict[Tuple, torch.Tensor] = {}
        Z_cache: Dict[Tuple, torch.Tensor] = {}
        stage_num, col_num = self.comp_graph.stage_num, self.comp_graph.col_num
        for s in range(stage_num + 1):
            for c in range(col_num):
                sum_mask = self.comp_graph.get_slice_sum_mask(s, c).to(self.device)
                if c == 0:
                    M_a = sum_mask[0, :, :]
                    M_b = sum_mask[1, :, :]
                    M_c = sum_mask[2, :, :]
                else:
                    carry_mask = self.comp_graph.get_slice_carry_mask(s, c).to(
                        self.device
                    )
                    M_a = torch.cat((sum_mask[0, :, :], carry_mask[0, :, :]), dim=0)
                    M_b = torch.cat((sum_mask[1, :, :], carry_mask[1, :, :]), dim=0)
                    M_c = torch.cat((sum_mask[2, :, :], carry_mask[2, :, :]), dim=0)
                M = torch.cat((M_a, M_b, M_c), dim=1)
                mask_cache[(s, c)] = M
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            if c == 0:
                Z_a = Z_mat_slice["sa"]
                Z_b = Z_mat_slice["sb"]
                Z_c = Z_mat_slice["sc"]
            else:
                Z_a = torch.cat([Z_mat_slice["sa"], Z_mat_slice["ca"]], dim=0)
                Z_b = torch.cat([Z_mat_slice["sb"], Z_mat_slice["cb"]], dim=0)
                Z_c = torch.cat([Z_mat_slice["sc"], Z_mat_slice["cc"]], dim=0)
            Z = torch.cat([Z_a, Z_b, Z_c], dim=1)
            Z_cache[(s, c)] = Z
        return mask_cache, Z_cache

    def sample_from_logits(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ):
        samples_connection = []
        overall_log_prob = 0.0
        mask_cache, Z_cache = self.get_cache(Z_mat_dict)

        for (s, c), Z_mat_slice in Z_mat_dict.items():
            Z = Z_cache[(s, c)]
            M = mask_cache[(s, c)]
            sum_src_indices = self.comp_graph.slice_indice_map[(s - 1, c)]
            dst_indices = self.comp_graph.slice_indice_map[(s, c)]
            for local_src_idx, src_idx in enumerate(sum_src_indices):
                logits = Z[local_src_idx, :].masked_fill(~M[local_src_idx, :], -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                sample = dist.sample()
                log_prob = dist.log_prob(sample)
                overall_log_prob += log_prob.item()

                local_dst_idx = sample.item() % len(dst_indices)
                dst_connec_type = sample.item() // len(dst_indices)
                dst_idx = dst_indices[local_dst_idx]

                dst_info = self.comp_graph.vertex_list[dst_idx]
                src_info = self.comp_graph.vertex_list[src_idx]

                assert dst_info[0] == src_info[0] + 1
                assert dst_info[1] == src_info[1]

                M[:, sample.item()] = False

                samples_connection.append(
                    (
                        src_idx,
                        dst_idx,
                        dst_connec_type,
                        {
                            "log_prob": log_prob.item(),
                            "local_src_idx": local_src_idx,
                            "local_dst_idx": local_dst_idx,
                            "sample": sample.item(),
                            "slice": (s, c),
                        },
                    )
                )
            if c > 0:
                carry_src_indices = self.comp_graph.slice_indice_map[(s - 1, c - 1)]
                for local_src_idx, src_idx in enumerate(carry_src_indices):
                    src_info = self.comp_graph.vertex_list[src_idx]
                    src_stage_idx, src_col_idx, src_type_idx, _ = src_info
                    if src_type_idx == 2 or src_type_idx == 3:
                        continue
                    logits = Z[local_src_idx + len(sum_src_indices), :].masked_fill(
                        ~M[local_src_idx + len(sum_src_indices), :], -1e9
                    )
                    dist = torch.distributions.Categorical(logits=logits)
                    sample = dist.sample()
                    log_prob = dist.log_prob(sample)
                    overall_log_prob += log_prob.item()
                    local_dst_idx = sample.item() % len(dst_indices)
                    dst_connec_type = sample.item() // len(dst_indices)
                    dst_idx = dst_indices[local_dst_idx]
                    dst_info = self.comp_graph.vertex_list[dst_idx]
                    assert dst_info[0] == src_info[0] + 1
                    assert dst_info[1] == src_info[1] + 1
                    M[:, sample.item()] = False
                    samples_connection.append(
                        (
                            src_idx,
                            dst_idx,
                            dst_connec_type,
                            {
                                "log_prob": log_prob.item(),
                                "local_src_idx": local_src_idx,
                                "local_dst_idx": local_dst_idx,
                                "sample": sample.item(),
                                "slice": (s, c),
                            },
                        )
                    )

        return samples_connection, overall_log_prob

    @staticmethod
    def _add_node(node_id, node_type, node_wires):
        if node_id not in node_wires:
            if node_type == 0:
                node_wires[node_id] = {
                    "from": {"a": None, "b": None, "c": None},
                    "to": {"sum": None, "carry": None},
                }
            elif node_type == 1:
                node_wires[node_id] = {
                    "from": {"a": None, "b": None},
                    "to": {"sum": None, "carry": None},
                }
            elif node_type == 2:
                node_wires[node_id] = {
                    "from": None,
                    "to": {"sum": None},
                }
            elif node_type == 3:
                node_wires[node_id] = {
                    "from": {"a": None},
                    "to": {"sum": None},
                }
            else:
                raise ValueError("Invalid node type")
        return node_wires

    @staticmethod
    def _declare_wire(wire_name, wire_set: Set, comment=""):
        if wire_name is None:
            return "", wire_set
        v_src = ""
        if wire_name not in wire_set:
            wire_set.add(wire_name)
            v_src += f"    // {comment}\n"
            v_src += f"    wire {wire_name};\n"
        return v_src, wire_set

    def _declare_pp(self, node_idx, wire_set: Set, node_wires: Dict):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 2
        v_src = ""
        instance_name = f"pp_{col_idx}[{idx}]"
        sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
        v, wire_set = self._declare_wire(sum_wire, wire_set)
        v_src += v

        v_src += f"    // pp node {(stage_idx, col_idx, type_idx, idx)}\n"
        v_src += f"    assign {sum_wire} = {instance_name};\n"
        return v_src, wire_set

    def _declare_visual(self, node_idx, wire_set: Set, node_wires: Dict):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 3
        v_src = ""
        instance_name = f"visual_{node_idx}"

        a_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}"
        if stage_idx < self.comp_graph.stage_num:
            sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
        else:
            sum_wire = None

        for wire in [a_wire, sum_wire]:
            v, wire_set = self._declare_wire(wire, wire_set)
            v_src += v
        v, wire_set = self._declare_wire(
            instance_name,
            wire_set,
            f"visual node {(stage_idx, col_idx, type_idx, idx)}",
        )
        v_src += v

        v_src += f"    assign {instance_name} = {a_wire};\n"
        if sum_wire is not None:
            v_src += f"    assign {sum_wire} = {instance_name};\n"
        return v_src, wire_set

    def _declare_ct32(self, node_idx, wire_set: Set, node_wires: Dict, cell_map=None):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 0
        v_src = ""
        instance_name = f"ct32_{node_idx}"
        cell = (cell_map or {}).get(node_idx) or "FA"

        a_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}"
        b_wire = f"from_{node_wires[node_idx]['from']['b']}_to_{node_idx}"
        c_wire = f"from_{node_wires[node_idx]['from']['c']}_to_{node_idx}"

        sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
        if node_wires[node_idx]["to"]["carry"] is not None:
            carry_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['carry']}"
        else:
            assert col_idx == self.comp_graph.col_num - 1
            carry_wire = None

        for wire in [a_wire, b_wire, c_wire, sum_wire, carry_wire]:
            v, wire_set = self._declare_wire(wire, wire_set)
            v_src += v
        v_src += f"// ct32 node {(stage_idx, col_idx, type_idx, idx)}\n"
        if carry_wire is not None:
            v_src += f"    {cell} {instance_name} (.a({a_wire}), .b({b_wire}), .cin({c_wire}), .sum({sum_wire}), .cout({carry_wire}));\n"
        else:
            # 末列无 carry：近似区不会到此（approx_max_col 远小于列数），仍用精确 FA
            v_src += f"    FA_no_carry {instance_name} (.a({a_wire}), .b({b_wire}), .cin({c_wire}), .sum({sum_wire}));\n"

        return v_src, wire_set

    def _declare_ct22(self, node_idx, wire_set: Set, node_wires: Dict, cell_map=None):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 1
        v_src = ""
        instance_name = f"ct22_{node_idx}"
        cell = (cell_map or {}).get(node_idx) or "HA"

        a_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}"
        b_wire = f"from_{node_wires[node_idx]['from']['b']}_to_{node_idx}"
        sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
        if node_wires[node_idx]["to"]["carry"] is not None:
            carry_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['carry']}"
        else:
            assert col_idx == self.comp_graph.col_num - 1
            carry_wire = None
        for wire in [a_wire, b_wire, sum_wire, carry_wire]:
            v, wire_set = self._declare_wire(wire, wire_set)
            v_src += v
        v_src += f"// ct22 node {(stage_idx, col_idx, type_idx, idx)}\n"
        if carry_wire is not None:
            v_src += f"    {cell} {instance_name} (.a({a_wire}), .cin({b_wire}), .sum({sum_wire}), .cout({carry_wire}));\n"
        else:
            # 末列无 carry：近似区不会到此，仍用精确 HA
            v_src += f"    HA_no_carry {instance_name} (.a({a_wire}), .cin({b_wire}), .sum({sum_wire}));\n"
        return v_src, wire_set

    def emit_assignment(self, samples_connection, cell_map=None):
        node_wires = {}
        INPUT_PORTS = ["a", "b", "c"]

        for src_idx, dst_idx, dst_conc_type, _ in samples_connection:
            src_info = self.comp_graph.vertex_list[src_idx]
            dst_info = self.comp_graph.vertex_list[dst_idx]
            src_stage_idx, src_col_idx, src_type_idx, _ = src_info
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            node_wires = self._add_node(src_idx, src_type_idx, node_wires)
            node_wires = self._add_node(dst_idx, dst_type_idx, node_wires)

            assert src_stage_idx + 1 == dst_stage_idx
            if src_col_idx == dst_col_idx:
                input_port_name = INPUT_PORTS[dst_conc_type]
                assert input_port_name in node_wires[dst_idx]["from"]
                node_wires[dst_idx]["from"][input_port_name] = src_idx
                assert "sum" in node_wires[src_idx]["to"]
                node_wires[src_idx]["to"]["sum"] = dst_idx
            elif src_col_idx + 1 == dst_col_idx:
                input_port_name = INPUT_PORTS[dst_conc_type]
                assert input_port_name in node_wires[dst_idx]["from"]
                node_wires[dst_idx]["from"][input_port_name] = src_idx
                assert "carry" in node_wires[src_idx]["to"]
                node_wires[src_idx]["to"]["carry"] = dst_idx
            else:
                raise ValueError(
                    f"Invalid edge: {src_info} -> {dst_info}, {src_col_idx} -> {dst_col_idx}"
                )
        v_src = ""
        wire_set = set()

        for node_idx in node_wires.keys():
            node_info = self.comp_graph.vertex_list[node_idx]
            stage_idx, col_idx, type_idx, idx = node_info
            if type_idx == 2:
                v, wire_set = self._declare_pp(node_idx, wire_set, node_wires)
            elif type_idx == 3:
                v, wire_set = self._declare_visual(node_idx, wire_set, node_wires)
            elif type_idx == 0:
                v, wire_set = self._declare_ct32(node_idx, wire_set, node_wires, cell_map)
            elif type_idx == 1:
                v, wire_set = self._declare_ct22(node_idx, wire_set, node_wires, cell_map)
            else:
                raise ValueError("Invalid node type")
            v_src += v

        routed_wire_list = [[] for _ in range(self.comp_graph.col_num)]
        for vertex_idx in range(len(self.comp_graph.vertex_list)):
            stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[vertex_idx]
            if type_idx == 3 and stage_idx == self.comp_graph.stage_num:
                routed_wire_list[col_idx].append(f"visual_{vertex_idx}")

        assignment = {
            "router_src": v_src,
            "routed_wire_list": routed_wire_list,
        }
        return assignment

    def get_Z_mat(self):
        time_start = time.time()
        x, edge_index_a, edge_index_b, edge_index_c = self.comp_graph.to_graph()
        x = x.to(self.device)
        edge_index_a = edge_index_a.to(self.device)
        edge_index_b = edge_index_b.to(self.device)
        edge_index_c = edge_index_c.to(self.device)
        time_end = time.time()
        time_start = time.time()
        if self.use_approx_types:
            out_a, out_b, out_c, out_sum, out_carry, self._node_emb = self.gcn.forward(
                x, edge_index_a, edge_index_b, edge_index_c, return_embedding=True
            )
        else:
            out_a, out_b, out_c, out_sum, out_carry = self.gcn.forward(
                x, edge_index_a, edge_index_b, edge_index_c
            )
        time_end = time.time()
        stage_num, col_num = self.comp_graph.stage_num, self.comp_graph.col_num
        Z_mat_dict = {}

        time_start = time.time()
        for s in range(stage_num + 1):
            for c in range(col_num):
                Z_mat_dict[(s, c)] = {}
                sum_src_indices = torch.tensor(
                    self.comp_graph.slice_indice_map[(s - 1, c)], device=self.device
                )
                dst_indices = torch.tensor(
                    self.comp_graph.slice_indice_map[(s, c)], device=self.device
                )
                sum_mask = self.comp_graph.get_slice_sum_mask(s, c).to(self.device)
                Z_sa = out_sum[sum_src_indices, :] @ out_a[dst_indices, :].T
                Z_sb = out_sum[sum_src_indices, :] @ out_b[dst_indices, :].T
                Z_sc = out_sum[sum_src_indices, :] @ out_c[dst_indices, :].T
                Z_sa = Z_sa.masked_fill(~sum_mask[0, :, :], -1e9)
                Z_sb = Z_sb.masked_fill(~sum_mask[1, :, :], -1e9)
                Z_sc = Z_sc.masked_fill(~sum_mask[2, :, :], -1e9)
                Z_mat_dict[(s, c)]["sa"] = Z_sa
                Z_mat_dict[(s, c)]["sb"] = Z_sb
                Z_mat_dict[(s, c)]["sc"] = Z_sc
                if c > 0:
                    carry_src_indices = torch.tensor(
                        self.comp_graph.slice_indice_map[(s - 1, c - 1)],
                        device=self.device,
                    )
                    carry_mask = self.comp_graph.get_slice_carry_mask(s, c).to(
                        self.device
                    )
                    Z_ca = out_carry[carry_src_indices, :] @ out_a[dst_indices, :].T
                    Z_cb = out_carry[carry_src_indices, :] @ out_b[dst_indices, :].T
                    Z_cc = out_carry[carry_src_indices, :] @ out_c[dst_indices, :].T
                    Z_ca = Z_ca.masked_fill(~carry_mask[0, :, :], -1e9)
                    Z_cb = Z_cb.masked_fill(~carry_mask[1, :, :], -1e9)
                    Z_cc = Z_cc.masked_fill(~carry_mask[2, :, :], -1e9)
                    Z_mat_dict[(s, c)]["ca"] = Z_ca
                    Z_mat_dict[(s, c)]["cb"] = Z_cb
                    Z_mat_dict[(s, c)]["cc"] = Z_cc
        time_end = time.time()
        return Z_mat_dict

    @staticmethod
    def parallel_simulate_worker(
        bit_width,
        encode_type,
        ct,
        rtl_path,
        build_path,
        target_delay,
        id,
        target_delay_id,
        synth,
        error_gate="analytic",
        error_gate_vectors=16_000_000,
    ):
        if synth == "dc":
            # 远端 DC 直出 PPA（功耗取 DC report_power，不走 VCS/XA）
            one = CompressorRouting._dc_simulate_one(
                bit_width, rtl_path, build_path, target_delay, id
            )
            if one is not None:
                simulated_result = [one]
            else:
                # P0(codex)：远端 DC 多次重试仍失败 → **不**回退本地 ABC（量纲差 ~20×，
                # 混进同一 PPO batch 会污染梯度/best；正是断网那次的故障）。标记失败，上层踢出本批。
                logging.warning(
                    f"[dc] worker {id} remote DC failed → 丢弃该样本(不混 ABC): {rtl_path}"
                )
                return {
                    "result": None,
                    "failed": True,
                    "id": id,
                    "target_delay_id": target_delay_id,
                    "target_delay": target_delay,
                }
        else:
            mul = Mul(bit_width, encode_type, ct)
            simulated_result = mul.simulate(
                build_path,
                rtl_path,
                [target_delay],
                synth=synth,
            )
        # 误差闸门：DC/综合成功后，并行在本 worker 测 verilator 真实 MED（失败=None，不丢样本）。
        measured_error = None
        if error_gate == "verilator":
            measured_error = CompressorRouting._measure_error_verilator(
                rtl_path, build_path, error_gate_vectors
            )
        return {
            "result": simulated_result,
            "measured_error": measured_error,
            "id": id,
            "target_delay_id": target_delay_id,
            "target_delay": target_delay,
        }

    @staticmethod
    def _dc_simulate_one(bit_width, rtl_path, build_path, target_delay, worker_id):
        """远端 DC 直出 PPA，返回与本地 simulate_worker 同构的 dict（power 转为 W），
        失败返回 None。使用专用 base 副本 sandbox_base_dcpwr（默认 POWER_MODE=dc，
        跳过 v2lvs/SPICE/VCS/XA，area/delay/power 全部取自 DC 综合）。"""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import run_power_sweep as rps

        # 指向 DC 直出专用 base 副本；原 sandbox_base 不受影响。
        rps.EDA_BASE_DIR = os.environ.get(
            "EDA_BASE_DIR_DC", "/home/lchangxian/sandbox/sandbox_base_dcpwr"
        )
        # evaluate_single_routing 在 cwd 下写 build/ 临时文件；切到可写目录避开 build 符号链接坑
        os.makedirs(build_path, exist_ok=True)
        try:
            os.chdir(build_path)
        except OSError:
            pass
        with open(rtl_path) as f:
            rtl_src = f.read()
        r = rps.evaluate_single_routing(
            worker_id, rtl_src, bit_width=bit_width, target_delay=target_delay
        )
        if (
            not r
            or not r.get("success")
            or r.get("area") is None
            or r.get("power_mw") is None
        ):
            return None
        delay = r.get("delay")
        if delay is None:
            delay = target_delay
        # evaluate_single_routing 的 delay 约定为负（关键路径到达时间），取绝对值得正向延时
        return {
            "delay": abs(float(delay)),
            "area": float(r["area"]),
            "power": float(r["power_mw"]) / 1000.0,  # mW → W，对齐本地 simulate_worker
            "target_delay": target_delay,
            "worker_id": worker_id,
        }

    @staticmethod
    def _measure_error_verilator(rtl_path, build_path, n_vectors):
        """误差闸门：verilator MC 实测 circular-wrap 真实误差（codex 审过的接入）。
        返回 dict(med, bias, wce_mc, source="verilator") 或 None（编译/运行/解析失败）。
        - 每次用全新 obj 目录（绝对路径；_dc_simulate_one 改过 cwd 不恢复，故全部绝对化）。
        - verilator --build -j1（8 worker 同跑时避免 make 多核过订阅）。
        - 失败重试 1 次；仍失败返回 None → 上层回退解析闸门（不丢该样本，别浪费 DC）。
        WCE 只上报不当闸门（MC 尾部不收敛）。"""
        import shutil
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        harness = os.path.join(repo_root, "verilate", "mul_err_wrap.cpp")
        rtl_abs = os.path.abspath(rtl_path)
        for attempt in range(2):
            verr = os.path.abspath(os.path.join(build_path, f"verr_{attempt}"))
            try:
                shutil.rmtree(verr, ignore_errors=True)
                os.makedirs(verr, exist_ok=True)
                obj = os.path.join(verr, "obj_dir")
                exe = os.path.join(obj, "mul_err")
                bcmd = ["verilator", "--cc", "--exe", "--build", "-j", "1", "-O3",
                        "-Wno-fatal", "--top-module", "MUL", "--Mdir", obj,
                        rtl_abs, harness, "-o", "mul_err"]
                b = subprocess.run(bcmd, cwd=verr, capture_output=True, text=True, timeout=180)
                if b.returncode != 0 or not os.path.exists(exe):
                    raise RuntimeError(f"verilator build rc={b.returncode}")
                r = subprocess.run([exe, str(int(n_vectors))], cwd=verr,
                                   capture_output=True, text=True, timeout=120)
                if r.returncode != 0:
                    raise RuntimeError(f"verilator run rc={r.returncode}")
                med = bias = wce = None
                for line in r.stdout.strip().splitlines():
                    p = line.split(",")
                    if p[0] == "masked":
                        med, bias, wce = float(p[1]), float(p[2]), float(p[5])
                        break
                if med is None:
                    raise RuntimeError("no masked line")
                shutil.rmtree(verr, ignore_errors=True)
                return {"med": med, "bias": bias, "wce_mc": wce, "source": "verilator"}
            except Exception as e:  # noqa: BLE001
                logging.warning("[errgate] verilator measure attempt %d failed (%s): %s",
                                attempt, os.path.basename(rtl_path), e)
                shutil.rmtree(verr, ignore_errors=True)
        return None

    def _apply_power_proxy_to_results(self, simulated_result, samples_connection):
        for item in simulated_result:
            item["eda_power"] = item.get("power")

        if self.power_source == "eda":
            for item in simulated_result:
                item["proxy_power_mw"] = None
                item["power_source"] = "eda"
            return simulated_result

        proxy_power_mw = self.power_proxy.predict_mw(self.comp_graph, samples_connection)
        proxy_power = proxy_power_mw * self.power_proxy_output_scale
        for item in simulated_result:
            item["proxy_power_mw"] = proxy_power_mw
            item["power"] = proxy_power
            item["power_source"] = "proxy"
        return simulated_result

    def get_samples(self):
        with torch.no_grad():
            sample_info = []
            Z_mat_dict = self.get_Z_mat()
            for sample_idx in range(self.num_samples):
                samples_connection, overall_log_prob = self.sample_from_logits(
                    Z_mat_dict
                )
                # Phase B：采样每个压缩器槽的 cell 类型（关时返回空，行为不变）
                cell_map, type_choices, type_log_prob = self.sample_cell_types()
                overall_log_prob += type_log_prob
                assignment = self.emit_assignment(samples_connection, cell_map=cell_map)

                ct = CompressorTree(
                    self.initial_pp, self.state["ct32"], self.state["ct22"]
                )
                if self.trunc_cols > 0:           # ① 把截断信息挂到 ct，emit_pp_encoder 会读
                    ct.trunc_cols = self.trunc_cols
                    ct.trunc_bits = self._trunc_bits
                mul = Mul(self.bit_width, self.encode_type, ct)
                rtl_path = os.path.join(self.build_dir, f"MUL-{sample_idx}.v")
                mul.emit_verilog(
                    rtl_path,
                    assignment=assignment,
                    extra_modules_src=self._approx_modules_src(cell_map),
                )
                sample_info.append(
                    {
                        "rtl_path": rtl_path,
                        "connection": samples_connection,
                        "overall_log_prob": overall_log_prob,
                        "cell_types": type_choices,
                    }
                )
                if (
                    self.inject_exact_candidate
                    and self.use_approx_types
                    and sample_idx == 0
                ):
                    exact_rtl_path = os.path.join(
                        self.build_dir, f"MUL-{self.num_samples}-exact.v"
                    )
                    mul.emit_verilog(
                        exact_rtl_path,
                        assignment=self.emit_assignment(samples_connection, cell_map={}),
                        extra_modules_src="",
                    )
                    sample_info.append(
                        {
                            "rtl_path": exact_rtl_path,
                            "connection": samples_connection,
                            "overall_log_prob": overall_log_prob,
                            "cell_types": {},
                            "baseline_only": True,
                            "candidate_kind": "all_exact",
                        }
                    )

            if self.synth in ("openroad", "dc"):
                if self.fixed_target_delay is not None:
                    target_delay_list = [self.fixed_target_delay]
                else:
                    target_delay_list = get_target_delay(self.bit_width)
            else:
                raise NotImplementedError
            params_list = [
                (
                    self.bit_width,
                    self.encode_type,
                    copy.deepcopy(ct),
                    sample["rtl_path"],
                    os.path.join(self.build_dir, f"worker_{i}_{target_delay_id}"),
                    target_delay,
                    i,
                    target_delay_id,
                    self.synth,
                    self.error_gate,
                    self.error_gate_vectors,
                )
                for i, sample in enumerate(sample_info)
                for target_delay_id, target_delay in enumerate(target_delay_list)
            ]
            logging.info(f"processings: {self.n_processing}")
            if self.n_processing == 1:
                results = [
                    self.parallel_simulate_worker(*param) for param in params_list
                ]
            else:
                with multiprocessing.Pool(self.n_processing) as pool:
                    results = pool.starmap(self.parallel_simulate_worker, params_list)
            processed_results = {}
            measured_errors = {}      # rid -> verilator 实测误差 dict（误差闸门，与 PPA 分开聚合）
            failed_ids = set()
            for result in results:
                rid = result["id"]
                if (
                    result.get("failed")
                    or not result.get("result")
                    or result["result"][0] is None
                ):
                    failed_ids.add(rid)
                    continue
                processed_results.setdefault(rid, []).append(result["result"][0])
                # verilator 失败 ≠ DC 失败：只在成功时记录，None 时上层回退解析（不丢样本）
                me = result.get("measured_error")
                if me is not None:
                    measured_errors.setdefault(rid, me)

            # P0(codex)：远端 DC 失败的样本直接踢出本批（不回退 ABC），避免量纲污染 PPO/best。
            kept_sample_info = []
            for i in range(len(sample_info)):
                if i not in processed_results:
                    continue
                result_list = self._apply_power_proxy_to_results(
                    processed_results[i],
                    sample_info[i]["connection"],
                )
                sample_info[i]["result"] = result_list
                sample_info[i]["measured_error"] = measured_errors.get(i)
                sample_info[i]["objective"] = self.get_objective(
                    result_list,
                    cell_types=sample_info[i].get("cell_types"),
                    measured_error=sample_info[i]["measured_error"],
                )
                kept_sample_info.append(sample_info[i])
            if failed_ids:
                logging.warning(
                    "[dc] %d/%d 样本远端 DC 失败已丢弃(不混 ABC); 本批保留 %d",
                    len(failed_ids), len(sample_info), len(kept_sample_info),
                )
            # 误差闸门健康度：verilator 失败回退解析会给该样本虚假优势（解析低估），
            # codex 建议监控回退率；过高(>50%)则本批 verilator 闸门失真，告警。
            if self.error_gate == "verilator" and kept_sample_info:
                n_fb = sum(1 for s in kept_sample_info if s.get("measured_error") is None)
                if n_fb:
                    lvl = logging.ERROR if n_fb * 2 > len(kept_sample_info) else logging.INFO
                    logging.log(lvl, "[errgate] verilator 回退解析 %d/%d 样本",
                                n_fb, len(kept_sample_info))
            if self.inject_exact_candidate:
                exact_kept = sum(
                    1 for s in kept_sample_info
                    if s.get("candidate_kind") == "all_exact"
                )
                logging.info("[exact] all-exact baseline candidates kept: %d", exact_kept)
        return kept_sample_info

    def _summarize_result(self, simulated_result):
        delay = float(np.mean([item["delay"] for item in simulated_result]))
        area = float(np.mean([item["area"] for item in simulated_result]))
        power = float(np.mean([item["power"] for item in simulated_result]))
        eda_power_values = [item.get("eda_power") for item in simulated_result]
        eda_power = None
        if all(value is not None for value in eda_power_values):
            eda_power = float(np.mean(eda_power_values))
        proxy_values = [item.get("proxy_power_mw") for item in simulated_result]
        proxy_power_mw = None
        if all(value is not None for value in proxy_values):
            proxy_power_mw = float(np.mean(proxy_values))

        area_violation = 0.0
        area_feasible = True
        if self.area_budget is not None:
            area_violation = max(0.0, area - float(self.area_budget))
            area_feasible = area_violation <= 0.0

        delay_violation = 0.0
        delay_feasible = True
        if self.fixed_target_delay is not None:
            delay_violation = max(0.0, delay - float(self.fixed_target_delay))
            delay_feasible = delay_violation <= 0.0

        return {
            "delay": delay,
            "area": area,
            "power": power,
            "eda_power": eda_power,
            "proxy_power_mw": proxy_power_mw,
            "area_budget": self.area_budget,
            "fixed_target_delay": self.fixed_target_delay,
            "area_violation": area_violation,
            "delay_violation": delay_violation,
            "area_feasible": area_feasible,
            "delay_feasible": delay_feasible,
            "power_source": self.power_source,
        }

    def get_objective(self, simulated_result, cell_types=None, measured_error=None):
        summary = self._summarize_result(simulated_result)

        # Phase B/C：近似+截断误差成本（约束式 A，LSB 单位）。两者全关时为 0，不影响。
        # codex review(medium)：截断误差不应被 use_approx_types/cell_types 门控——纯截断
        # （trunc_cols>0 但无类型搜索）也要计入，否则误差预算骗不过。
        err_term = 0.0
        if (self.use_approx_types and cell_types) or self.trunc_cols > 0:
            med, abs_bias, _nmed, wce = self._analytic_error(cell_types or {})
            # 误差闸门（codex 审过）：verilator 模式且实测可用 → med/bias 用 circular-wrap 真实值
            # （解析 proxy 系统性低估 0–30%）；wce 始终用解析上界（MC 尾部不收敛、不可信）。
            # verilator 失败(measured_error=None) → 回退解析 med/bias（不浪费已花的 DC）。
            if self.error_gate == "verilator" and measured_error is not None:
                med = measured_error["med"]
                abs_bias = abs(measured_error["bias"])
            # 点2：除 error_scale 把 LSB 绝对值压到和 PPA 同量级。
            if self.error_as_metric:
                # 误差作为普通目标项，评估方式同 area/power：error_weight*med/error_scale
                # （线性计入，无 budget 铰链；med_budget/med_violation_weight 此模式忽略）。
                err_term += self.error_weight * med / self.error_scale
            elif self.med_budget is not None:
                err_term += (
                    self.med_violation_weight
                    * max(0.0, med - self.med_budget)
                    / self.error_scale
                )
            err_term += self.bias_weight * abs_bias / self.error_scale
            # ④ 尾部/WCE 约束（默认关）：超出 wce_budget 才罚，压重尾/最坏情况误差。
            if self.wce_budget is not None:
                err_term += (
                    self.wce_violation_weight
                    * max(0.0, wce - self.wce_budget)
                    / self.error_scale
                )

        if self.area_budget is not None:
            objective = summary["power"] / self.power_scale
            objective += (
                self.area_violation_weight
                * summary["area_violation"]
                / self.area_scale
            )
            if self.fixed_target_delay is not None:
                objective += (
                    self.delay_violation_weight
                    * summary["delay_violation"]
                    / self.delay_scale
                )
            return objective + err_term

        if self.delay_as_constraint:
            # P0(codex)：delay≤target 不奖励也不罚（不再把预算花在压 delay）；超 target 才罚。
            dtarget = (
                self.delay_target_ns
                if self.delay_target_ns is not None
                else self.fixed_target_delay
            )
            delay_cost = 0.0
            if dtarget is not None:
                delay_cost = (
                    self.delay_violation_weight
                    * max(0.0, summary["delay"] - float(dtarget))
                    / self.delay_scale
                )
            objective = (
                delay_cost
                + self.area_weight * summary["area"] / self.area_scale
                + self.power_weight * summary["power"] / self.power_scale
            )
        else:
            objective = (
                self.delay_weight * summary["delay"] / self.delay_scale
                + self.area_weight * summary["area"] / self.area_scale
                + self.power_weight * summary["power"] / self.power_scale
            )
        return objective + err_term

    def _candidate_rank(self, sample_info):
        result = sample_info.get("result", sample_info.get("simulated_result"))
        summary = self._summarize_result(result)
        # Unconstrained (EDA) mode mirrors Arith-DAS exactly: rank purely by the
        # scalar objective so the exported "best" design matches the baseline.
        # The feasibility/power ranking below only applies to area-budget runs.
        if self.area_budget is None:
            return (0, 0.0, sample_info["objective"])
        feasible = summary["area_feasible"] and summary["delay_feasible"]
        violation = summary["area_violation"] + summary["delay_violation"]
        return (0 if feasible else 1, violation, summary["power"])

    def _best_info_metadata(self):
        if self.found_best_info["simulated_result"] is None:
            return {}
        summary = self._summarize_result(self.found_best_info["simulated_result"])
        summary["objective"] = self.found_best_info["objective"]
        return summary

    def export_best_candidate(self, export_dir):
        if self.found_best_info["connection"] is None:
            raise ValueError("No best candidate has been found; run training first")

        os.makedirs(export_dir, exist_ok=True)
        old_state = self.state
        old_assignment = self.assignment
        old_comp_graph = self.comp_graph
        try:
            self.state = copy.deepcopy(self.found_best_info["ct"])
            self.assignment = copy.deepcopy(self.found_best_info["assignment"])
            self.comp_graph = CompressorGraph(self.initial_pp, self.assignment)
            # Phase B：从最优设计的 cell_types 复原近似 cell（comp_graph 同序，node_idx 一致）
            cell_types = self.found_best_info.get("cell_types") or {}
            cell_map = self._cell_map_from_types(cell_types)
            routing_assignment = self.emit_assignment(
                self.found_best_info["connection"], cell_map=cell_map
            )
            ct = CompressorTree(self.initial_pp, self.state["ct32"], self.state["ct22"])
            if self.trunc_cols > 0:               # ① 导出也带截断
                ct.trunc_cols = self.trunc_cols
                ct.trunc_bits = self._trunc_bits
            mul = Mul(self.bit_width, self.encode_type, ct)
            rtl_path = os.path.join(export_dir, "MUL.v")
            mul.emit_verilog(
                rtl_path,
                assignment=routing_assignment,
                extra_modules_src=self._approx_modules_src(cell_map),
            )

            best_info = {
                **self._best_info_metadata(),
                "connection": self.found_best_info["connection"],
                "ct": self.found_best_info["ct"],
                "assignment": self.found_best_info["assignment"],
                "cell_types": cell_types,
                "approx_cells": {str(k): v for k, v in cell_map.items()},
                "routing_assignment": routing_assignment,
                "rtl_path": rtl_path,
                "simulated_result": self.found_best_info["simulated_result"],
                # 误差闸门审计：该最优点用的是 verilator 实测还是解析回退
                "error_source": self.found_best_info.get("error_source"),
                "measured_error": self.found_best_info.get("measured_error"),
            }
            with open(os.path.join(export_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=4, default=convert_to_serializable)
            return rtl_path
        finally:
            self.state = old_state
            self.assignment = old_assignment
            self.comp_graph = old_comp_graph

    def get_ppo_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
        sample_info_list: List[Dict],
    ):
        l = torch.tensor([0.0], device=self.device)
        # 点1：advantage 归一。把 A 从原始 -obj 改成 -(obj-mean)/(std+eps)，
        # 减均值给出"比本批平均好/差多少"的相对信号，除标准差把量纲压到 O(1)，
        # 使学习信号不再被 obj 绝对大小（误差尺度）主导。normalize_advantage=False 时为旧行为。
        if self.normalize_advantage:
            _objs = np.array(
                [si["objective"] for si in sample_info_list], dtype=np.float64
            )
            _adv_mean = float(_objs.mean())
            _adv_std = float(_objs.std()) + 1e-8
        for sample_info in sample_info_list:
            old_log_prob = sample_info["overall_log_prob"]
            new_log_prob = 0.0
            sample_id = 0
            mask_cache, Z_cache = self.get_cache(Z_mat_dict)
            for (s, c), Z_mat_slice in Z_mat_dict.items():
                Z = Z_cache[(s, c)]
                M = mask_cache[(s, c)]
                sum_src_indices = torch.tensor(
                    self.comp_graph.slice_indice_map[(s - 1, c)], device=self.device
                )
                for local_src_idx, src_idx in enumerate(sum_src_indices):
                    sample = sample_info["connection"][sample_id][3]
                    sample_id += 1
                    logits = Z[local_src_idx, :].masked_fill(~M[local_src_idx, :], -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(
                        torch.tensor([sample["sample"]], device=self.device)
                    )
                    new_log_prob += log_prob
                    M[:, sample["sample"]] = False

                if c > 0:
                    carry_src_indices = torch.tensor(
                        self.comp_graph.slice_indice_map[(s - 1, c - 1)],
                        device=self.device,
                    )
                    for local_src_idx, src_idx in enumerate(carry_src_indices):
                        src_info = self.comp_graph.vertex_list[src_idx]
                        src_stage_idx, src_col_idx, src_type_idx, _ = src_info
                        if src_type_idx == 2 or src_type_idx == 3:
                            continue

                        sample = sample_info["connection"][sample_id][3]
                        sample_id += 1
                        logits = Z[local_src_idx + len(sum_src_indices), :].masked_fill(
                            ~M[local_src_idx + len(sum_src_indices), :], -1e9
                        )
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(
                            torch.tensor([sample["sample"]], device=self.device)
                        )
                        new_log_prob += log_prob
                        M[:, sample["sample"]] = False

            # Phase B：重算类型头 log_prob，与采样口径一致（同 approx_max_col 掩码）
            if self.use_approx_types and sample_info.get("cell_types"):
                emb = self._node_emb
                for node_idx, (t, k) in sample_info["cell_types"].items():
                    head = self.type_head_32 if t == 0 else self.type_head_22
                    c = self.comp_graph.vertex_list[node_idx][1]
                    logits = self._masked_type_logits(head(emb[node_idx]), c)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_prob = new_log_prob + dist.log_prob(
                        torch.tensor([k], device=self.device)
                    )

            if self.normalize_advantage:
                A = -(sample_info["objective"] - _adv_mean) / _adv_std
            else:
                A = -sample_info["objective"]
            ratio = torch.exp(new_log_prob - old_log_prob)
            loss_1 = A * ratio
            loss_2 = A * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss_1, loss_2)
            l += -loss
        l /= len(sample_info_list)
        return l

    def update_found_best_info(self, sample_info_list):
        for sample_info in sample_info_list:
            is_better = self.found_best_info["connection"] is None
            if not is_better:
                is_better = self._candidate_rank(sample_info) < self._candidate_rank(
                    self.found_best_info
                )
            if is_better:
                self.found_best_info["objective"] = sample_info["objective"]
                self.found_best_info["connection"] = sample_info["connection"]
                self.found_best_info["ct"] = copy.deepcopy(self.state)
                self.found_best_info["assignment"] = copy.deepcopy(self.assignment)
                self.found_best_info["simulated_result"] = sample_info["result"]
                self.found_best_info["cell_types"] = copy.deepcopy(
                    sample_info.get("cell_types")
                )
                # 可审计：记录该最优点的误差闸门来源（verilator 实测 / analytic 回退）
                me = sample_info.get("measured_error")
                self.found_best_info["measured_error"] = copy.deepcopy(me)
                self.found_best_info["error_source"] = (
                    me["source"] if me else
                    ("analytic_fallback" if self.error_gate == "verilator" else "analytic")
                )
                self.found_best_info.update(self._best_info_metadata())

    def _effective_med(self, d):
        """与 get_objective 同口径的 MED（供日志/上报）：verilator 实测优先，失败回退
        解析；无误差源（无近似 cell 且无截断）= 0。d 可为单个 sample_info 或 found_best_info。"""
        me = d.get("measured_error")
        if (self.error_gate == "verilator"
                and me is not None and me.get("med") is not None):
            return float(me["med"])
        if (self.use_approx_types and d.get("cell_types")) or self.trunc_cols > 0:
            med, _b, _n, _w = self._analytic_error(d.get("cell_types") or {})
            return float(med)
        return 0.0

    def log_episode(self, episode_idx, info):
        self.tb_logger.add_scalar("objective", info["objective"], episode_idx)
        self.tb_logger.add_scalar(
            "weight/disc_loss_weight", self.disc_loss_weight, episode_idx
        )
        self.tb_logger.add_scalar(
            "weight/rule_loss_weight", self.rule_loss_weight, episode_idx
        )
        for epoch_loss_info in info["epoch_loss"]:
            for loss_key in epoch_loss_info.keys():
                self.tb_logger.add_scalar(
                    f"epoch_loss/{loss_key}",
                    epoch_loss_info[loss_key],
                    self.total_epoch_num,
                )
            self.total_epoch_num += 1

        for loss_key in info["epoch_loss"][0].keys():
            loss_value = 0.0
            for epoch_loss_info in info["epoch_loss"]:
                loss_value += epoch_loss_info[loss_key]
            loss_value /= len(info["epoch_loss"])
            self.tb_logger.add_scalar(
                f"episode_loss/{loss_key}", loss_value, episode_idx
            )

        for ppa_key in ["area", "delay", "power"]:
            ppa_value = 0.0
            for simulated_result in info["simulated_result"]:
                ppa_value += simulated_result[ppa_key]
            ppa_value /= len(info["simulated_result"])
            self.tb_logger.add_scalar(f"ppa/{ppa_key}", ppa_value, episode_idx)
        self.tb_logger.add_scalar("ppa/med", info.get("med", 0.0), episode_idx)
        self.tb_logger.add_scalar("lr", self.scheduler.get_last_lr()[0], episode_idx)

        self.tb_logger.add_scalar(
            "found_best/objective",
            self.found_best_info["objective"],
            episode_idx,
        )
        for ppa_key in ["area", "delay", "power"]:
            ppa_value = 0.0
            for simulated_result in self.found_best_info["simulated_result"]:
                ppa_value += simulated_result[ppa_key]
            ppa_value /= len(self.found_best_info["simulated_result"])
            self.tb_logger.add_scalar(f"found_best/{ppa_key}", ppa_value, episode_idx)
        best_med = self._effective_med(self.found_best_info)
        self.tb_logger.add_scalar("found_best/med", best_med, episode_idx)
        self.tb_logger.add_scalar("lr", self.scheduler.get_last_lr()[0], episode_idx)

        # ── real-time console progress ──────────────────────────────────────
        cur  = self._summarize_result(info["simulated_result"])
        best = self._best_info_metadata()
        vio_str = (
            f"  area_vio={cur['area_violation']:.1f}({'OK' if cur['area_feasible'] else 'X'})"
            if self.area_budget is not None else ""
        )
        proxy_str = (
            f"  eda_pwr={cur.get('eda_power', 0) * 1000:.4f}mW"
            if self.power_source == "proxy" and cur.get("eda_power") is not None
            else ""
        )
        logging.info(
            "[ep %4d/%d]  obj=%.6f  area=%.1f  delay=%.4fns"
            "  pwr=%.4fmW[%s]  med=%.1f%s%s"
            "  || best: obj=%.6f  area=%.1f  pwr=%.4fmW  med=%.1f",
            episode_idx, self.num_episodes, info["objective"],
            cur["area"], cur["delay"], cur["power"] * 1000,
            self.power_source, info.get("med", float("nan")), vio_str, proxy_str,
            best["objective"], best["area"], best["power"] * 1000, best_med,
        )

    def run_episode(self, episode_idx):
        logging.info(f"Episode {episode_idx} start")

        logging.info(f"sampling")
        self.reset()
        sample_info_list = self.get_samples()
        if not sample_info_list:
            # P0(codex)：整批远端 DC 都失败 → 跳过本 episode（不更新策略/best），保持 LR schedule 对齐。
            logging.warning(f"Episode {episode_idx}: 全批远端 DC 失败, 跳过本轮更新")
            self.scheduler.step()
            return
        self.update_found_best_info(sample_info_list)
        ppo_sample_info_list = [
            item for item in sample_info_list if not item.get("baseline_only")
        ]

        min_idx = np.argmin([item["objective"] for item in sample_info_list])
        info = {}
        info["epoch_loss"] = []
        info["objective"] = sample_info_list[min_idx]["objective"]
        info["simulated_result"] = sample_info_list[min_idx]["result"]
        info["med"] = self._effective_med(sample_info_list[min_idx])

        self.update_pool(sample_info_list[min_idx]["objective"], self.state)

        logging.info(f"updating")
        for epoch_idx in range(self.num_epochs):
            Z_mat_dict = self.get_Z_mat()

            loss_info = {}
            l = torch.tensor([0.0], device=self.device)
            if self.use_ppo_loss and ppo_sample_info_list:
                l_ppo = self.get_ppo_loss(Z_mat_dict, ppo_sample_info_list)
                l += self.ppo_loss_weight * l_ppo
                loss_info["l_ppo"] = l_ppo.item()
            elif self.use_ppo_loss:
                loss_info["l_ppo"] = 0.0
            if self.use_disc_loss:
                l_discrete = self.get_discrete_loss(Z_mat_dict)
                l += self.disc_loss_weight * l_discrete
                loss_info["l_discrete"] = l_discrete.item()
                self.disc_loss_weight += self.disc_loss_weight_incr
            if self.use_rule_loss:
                l_rule = self.get_rule_loss(Z_mat_dict)
                l += self.rule_loss_weight * l_rule
                loss_info["l_rule"] = l_rule.item()
                self.rule_loss_weight += self.rule_loss_weight_incr
            if self.use_delay_loss:
                l_delay = self.get_delay_loss(Z_mat_dict)
                l += self.delay_loss_weight * l_delay
                loss_info["l_delay"] = l_delay.item()
            if self.use_error_loss and self.use_approx_types and not self.error_as_metric:
                # D2 可微误差 surrogate（权重已在 get_error_loss 内部乘好）。
                # error_as_metric 模式下误差已作普通目标项进 reward，可微 surrogate 关闭。
                l_error = self.get_error_loss()
                l += l_error
                loss_info["l_error"] = l_error.item()

            self.optim.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), self.max_grad_norm)
            self.optim.step()

            loss_info["l"] = l.item()
            info["epoch_loss"].append(loss_info)
        if episode_idx % self.log_freq == 0:
            self.log_episode(episode_idx, info)
        self.scheduler.step()

    def _start_reset(self):
        self.initial_pp = get_initial_partial_product(
            self.bit_width, self.encode_type
        ).astype(int)
        if self.trunc_cols > 0 and not self._trunc_bits:
            self._setup_truncation()
        if self.ct_arch == "wallace":
            ct = CompressorTree.wallace(self.initial_pp)
        elif self.ct_arch == "dadda":
            ct = CompressorTree.dadda(self.initial_pp)
        else:
            raise ValueError("Invalid compressor tree architecture")
        init_objective = self.get_objective(
            [
                {
                    "delay": self.delay_scale,
                    "area": self.area_scale,
                    "power": self.power_scale,
                }
            ]
        )
        init_state = {
            "ct32": ct.ct32.astype(int),
            "ct22": ct.ct22.astype(int),
        }
        self.pool.add(init_objective, init_state)

        if self.gomil_path is not None:
            logging.info(f"Loading gomil from {self.gomil_path}")
            try:
                with open(self.gomil_path, "r") as f:
                    gomil_data = json.load(f)
                    gomil_state = {
                        "ct32": np.asarray(gomil_data["ct"]["ct32"], dtype=int),
                        "ct22": np.asarray(gomil_data["ct"]["ct22"], dtype=int),
                    }
                    gomil_objective = self.get_objective(
                        gomil_data["simulated_result_list"],
                        cell_types=gomil_data.get("cell_types"),
                    )
                    self.pool.add(gomil_objective, gomil_state)

            except Exception as e:
                logging.error(f"Failed to load gomil: {e}")

    def reset(self):
        pool_list = self.pool.get_pool()
        logging.info(f"pool size: {len(pool_list)}")
        if len(pool_list) == 0:
            raise ValueError("Pool is empty, cannot reset environment.")
        sampled_item = random.choice(pool_list)
        random_objective, random_state = sampled_item

        self.state = copy.deepcopy(random_state)
        action_mask = self.get_action_mask()
        action = random.choice(np.where(action_mask == 1)[0])
        self.transition(action)

        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct = CompressorTree(pp, self.state["ct32"], self.state["ct22"])
        self.assignment = ct.compressor_assignment_fused()
        self.comp_graph = CompressorGraph(pp, self.assignment)

    def legalize_ct_architecture(self, ct32: np.ndarray, ct22: np.ndarray):
        initial_pp = copy.deepcopy(self.initial_pp)
        assert len(ct32) == len(initial_pp) and len(ct22) == len(initial_pp)
        ct32 = copy.deepcopy(ct32).astype(int)
        ct22 = copy.deepcopy(ct22).astype(int)
        for column_index in range(0, len(initial_pp)):
            ct32[column_index] = max(0, ct32[column_index])
            ct22[column_index] = max(0, ct22[column_index])
            if column_index == 0:
                remain_pp = (
                    initial_pp[column_index]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            else:
                remain_pp = (
                    initial_pp[column_index]
                    + ct32[column_index - 1]
                    + ct22[column_index - 1]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            if remain_pp < 1:
                if ct22[column_index] + remain_pp >= 1:
                    ct22[column_index] += remain_pp - 1
                else:
                    remain_pp += ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] -= (2 - remain_pp) // 2
                    else:
                        ct32[column_index] -= (1 - remain_pp) // 2
            elif remain_pp > 2:
                if remain_pp - ct22[column_index] <= 2:
                    ct22[column_index] -= remain_pp - 2
                    ct32[column_index] += remain_pp - 2
                else:
                    ct32[column_index] += ct22[column_index]
                    remain_pp -= ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] += (remain_pp - 2) / 2
                    else:
                        ct32[column_index] += (remain_pp - 1) / 2

        remain_pp = copy.deepcopy(initial_pp)
        remain_pp[0] = initial_pp[0] - 2 * ct32[0] - ct22[0]
        for column_index in range(1, len(initial_pp)):
            remain_pp[column_index] = (
                initial_pp[column_index]
                + ct32[column_index - 1]
                + ct22[column_index - 1]
                - 2 * ct32[column_index]
                - ct22[column_index]
            )
        remain_pp = np.asarray(remain_pp)
        return ct32, ct22

    def transition(self, action: int) -> np.ndarray:
        action_column = action // 4
        action_type = action % 4
        ct_32 = copy.deepcopy(self.state["ct32"])
        ct_22 = copy.deepcopy(self.state["ct22"])

        if action_type == 0:
            ct_22[action_column] += 1
        elif action_type == 1:
            ct_22[action_column] -= 1
        elif action_type == 2:
            ct_22[action_column] += 1
            ct_32[action_column] -= 1
        elif action_type == 3:
            ct_22[action_column] -= 1
            ct_32[action_column] += 1
        else:
            raise NotImplementedError

        legalized_ct32, legalized_ct22 = self.legalize_ct_architecture(ct_32, ct_22)
        self.state["ct32"] = legalized_ct32
        self.state["ct22"] = legalized_ct22

    def get_action_mask(self):
        action_type_num = 4
        ct_32 = self.state["ct32"]
        ct_22 = self.state["ct22"]

        initial_pp = self.initial_pp
        mask = np.zeros([action_type_num * len(initial_pp)])
        remain_pp = copy.deepcopy(initial_pp)
        for column_index in range(len(remain_pp)):
            if column_index > 0:
                remain_pp[column_index] += (
                    ct_32[column_index - 1] + ct_22[column_index - 1]
                )
            remain_pp[column_index] += -2 * ct_32[column_index] - ct_22[column_index]

        legal_act = []
        for column_index in range(2, len(initial_pp)):
            if remain_pp[column_index] == 2:
                legal_act.append((column_index, 0))
                if ct_22[column_index] >= 1:
                    legal_act.append((column_index, 3))
            if remain_pp[column_index] == 1:
                if ct_32[column_index] >= 1:
                    legal_act.append((column_index, 2))
                if ct_22[column_index] >= 1:
                    legal_act.append((column_index, 1))

        for act_col, action in legal_act:
            pp = copy.deepcopy(remain_pp)
            ct_32 = copy.deepcopy(self.state["ct32"])
            ct_22 = copy.deepcopy(self.state["ct22"])

            if action == 0:
                ct_22[act_col] = ct_22[act_col] + 1
                pp[act_col] = pp[act_col] - 1
                if act_col + 1 < len(pp):
                    pp[act_col + 1] = pp[act_col + 1] + 1
            elif action == 1:
                ct_22[act_col] = ct_22[act_col] - 1
                pp[act_col] = pp[act_col] + 1
                if act_col + 1 < len(pp):
                    pp[act_col + 1] = pp[act_col + 1] - 1
            elif action == 2:
                ct_22[act_col] = ct_22[act_col] + 1
                ct_32[act_col] = ct_32[act_col] - 1
                pp[act_col] = pp[act_col] + 1
            elif action == 3:
                ct_22[act_col] = ct_22[act_col] - 1
                ct_32[act_col] = ct_32[act_col] + 1
                pp[act_col] = pp[act_col] - 1

            for i in range(act_col + 1, len(pp) + 1):
                if i == len(pp):
                    mask[act_col * action_type_num + action] = 1
                    break
                elif pp[i] == 1 or pp[i] == 2:
                    mask[act_col * action_type_num + action] = 1
                    break
                elif pp[i] == 3:
                    ct_32[i] = ct_32[i] + 1
                    if i + 1 < len(pp):
                        pp[i + 1] = pp[i + 1] + 1
                    pp[i] = 1
                elif pp[i] == 0:
                    if ct_22[i] >= 1:
                        ct_22[i] = ct_22[i] - 1
                        if i + 1 < len(pp):
                            pp[i + 1] = pp[i + 1] - 1
                        pp[i] = 1
                    else:
                        ct_32[i] = ct_32[i] - 1
                        if i + 1 < len(pp):
                            pp[i + 1] = pp[i + 1] - 1
                        pp[i] = 2
        mask = mask != 0
        return mask

    def update_pool(
        self,
        objective: float,
        state: Dict[str, np.ndarray],
    ):
        self.pool.add(objective, state)

    def get_pool_objectives(self):
        pool_list = self.pool.get_pool()
        objectives = [item[0] for item in pool_list]
        return objectives
