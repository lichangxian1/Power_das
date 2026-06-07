import os
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

        self.fc_a = nn.Linear(in_dim, output_dim)
        self.fc_b = nn.Linear(in_dim, output_dim)
        self.fc_c = nn.Linear(in_dim, output_dim)

        self.fc_sum = nn.Linear(in_dim, output_dim)
        self.fc_carry = nn.Linear(in_dim, output_dim)

    def forward(
        self, x, edge_index_a, edge_index_b, edge_index_c
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, edge_index_a, edge_index_b, edge_index_c)
        out_a = self.fc_a(x)
        out_b = self.fc_b(x)
        out_c = self.fc_c(x)

        out_sum = self.fc_sum(x)
        out_carry = self.fc_carry(x)
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

        self.optim: optim.Optimizer = getattr(optim, optim_name)(
            self.gcn.parameters(), **optim_kwargs
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
        }

        self.total_epoch_num = 0

        self.initial_pp: np.ndarray = None

        if pool_size > 0:
            self.pool = BoundedParetoPool(pool_size)
        else:
            self.pool = None

        self._start_reset()

    def get_full_target_delay_result(self):
        build_dir = self.build_dir + "_full_ppa"
        rtl_path = os.path.join(build_dir, "MUL.v")
        if self.synth == "openroad":
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
        mul = Mul(self.bit_width, self.encode_type, ct)

        assignment = self.emit_assignment(self.found_best_info["connection"])
        mul.emit_verilog(rtl_path, assignment=assignment)
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

    def _declare_ct32(self, node_idx, wire_set: Set, node_wires: Dict):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 0
        v_src = ""
        instance_name = f"ct32_{node_idx}"

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
            v_src += f"    FA {instance_name} (.a({a_wire}), .b({b_wire}), .cin({c_wire}), .sum({sum_wire}), .cout({carry_wire}));\n"
        else:
            v_src += f"    FA_no_carry {instance_name} (.a({a_wire}), .b({b_wire}), .cin({c_wire}), .sum({sum_wire}));\n"

        return v_src, wire_set

    def _declare_ct22(self, node_idx, wire_set: Set, node_wires: Dict):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 1
        v_src = ""
        instance_name = f"ct22_{node_idx}"

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
            v_src += f"    HA {instance_name} (.a({a_wire}), .cin({b_wire}), .sum({sum_wire}), .cout({carry_wire}));\n"
        else:
            v_src += f"    HA_no_carry {instance_name} (.a({a_wire}), .cin({b_wire}), .sum({sum_wire}));\n"
        return v_src, wire_set

    def emit_assignment(self, samples_connection):
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
                v, wire_set = self._declare_ct32(node_idx, wire_set, node_wires)
            elif type_idx == 1:
                v, wire_set = self._declare_ct22(node_idx, wire_set, node_wires)
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
    ):
        mul = Mul(bit_width, encode_type, ct)
        simulated_result = mul.simulate(
            build_path,
            rtl_path,
            [target_delay],
            synth=synth,
        )
        return {
            "result": simulated_result,
            "id": id,
            "target_delay_id": target_delay_id,
            "target_delay": target_delay,
        }

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
                assignment = self.emit_assignment(samples_connection)

                ct = CompressorTree(
                    self.initial_pp, self.state["ct32"], self.state["ct22"]
                )
                mul = Mul(self.bit_width, self.encode_type, ct)
                rtl_path = os.path.join(self.build_dir, f"MUL-{sample_idx}.v")
                mul.emit_verilog(rtl_path, assignment=assignment)
                sample_info.append(
                    {
                        "rtl_path": rtl_path,
                        "connection": samples_connection,
                        "overall_log_prob": overall_log_prob,
                    }
                )

            if self.synth == "openroad":
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
            for result in results:
                id = result["id"]
                if id not in processed_results:
                    processed_results[id] = []
                processed_results[id].append(result["result"][0])

            for i, result_list in processed_results.items():
                result_list = self._apply_power_proxy_to_results(
                    result_list,
                    sample_info[i]["connection"],
                )
                sample_info[i]["result"] = result_list
                sample_info[i]["objective"] = self.get_objective(result_list)
        return sample_info

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

    def get_objective(self, simulated_result):
        summary = self._summarize_result(simulated_result)
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
            return objective

        objective = (
            self.delay_weight * summary["delay"] / self.delay_scale
            + self.area_weight * summary["area"] / self.area_scale
            + self.power_weight * summary["power"] / self.power_scale
        )
        return objective

    def _candidate_rank(self, sample_info):
        result = sample_info.get("result", sample_info.get("simulated_result"))
        summary = self._summarize_result(result)
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
            routing_assignment = self.emit_assignment(self.found_best_info["connection"])
            ct = CompressorTree(self.initial_pp, self.state["ct32"], self.state["ct22"])
            mul = Mul(self.bit_width, self.encode_type, ct)
            rtl_path = os.path.join(export_dir, "MUL.v")
            mul.emit_verilog(rtl_path, assignment=routing_assignment)

            best_info = {
                **self._best_info_metadata(),
                "connection": self.found_best_info["connection"],
                "ct": self.found_best_info["ct"],
                "assignment": self.found_best_info["assignment"],
                "routing_assignment": routing_assignment,
                "rtl_path": rtl_path,
                "simulated_result": self.found_best_info["simulated_result"],
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
                self.found_best_info.update(self._best_info_metadata())

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
        self.tb_logger.add_scalar("lr", self.scheduler.get_last_lr()[0], episode_idx)

    def run_episode(self, episode_idx):
        logging.info(f"Episode {episode_idx} start")

        logging.info(f"sampling")
        self.reset()
        sample_info_list = self.get_samples()
        self.update_found_best_info(sample_info_list)

        min_idx = np.argmin([item["objective"] for item in sample_info_list])
        info = {}
        info["epoch_loss"] = []
        info["objective"] = sample_info_list[min_idx]["objective"]
        info["simulated_result"] = sample_info_list[min_idx]["result"]

        self.update_pool(sample_info_list[min_idx]["objective"], self.state)

        logging.info(f"updating")
        for epoch_idx in range(self.num_epochs):
            Z_mat_dict = self.get_Z_mat()

            loss_info = {}
            l = torch.tensor([0.0], device=self.device)
            if self.use_ppo_loss:
                l_ppo = self.get_ppo_loss(Z_mat_dict, sample_info_list)
                l += self.ppo_loss_weight * l_ppo
                loss_info["l_ppo"] = l_ppo.item()
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
                        gomil_data["simulated_result_list"]
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
