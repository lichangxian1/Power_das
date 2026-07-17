"""CompressorGraph：把压缩树连接矩阵翻译成 GCN 输入图
（节点特征 + sum/carry/type 三类边）。"""
from typing import Dict, List, Tuple
import logging

import torch
from torch_geometric.utils import to_undirected, add_self_loops

import numpy as np


class CompressorGraph:
    def __init__(
        self,
        pp: np.ndarray,
        assignment: List[List[Tuple]],
        num_node_types: int = 4,
    ):
        self.assignment = assignment
        self.pp = pp
        self.num_node_types = int(num_node_types)
        self.port_num = 4 if self.num_node_types >= 5 else 3

        self.stage_num = len(assignment)
        self.col_num = len(assignment[0])
        self.vertex_list = []
        self.indice_map = {}

        remain_pp = np.zeros_like(pp, dtype=int)
        ct32 = np.zeros_like(pp, dtype=int)
        ct22 = np.zeros_like(pp, dtype=int)
        ct42 = np.zeros_like(pp, dtype=int)
        dec_ct32 = np.zeros((self.stage_num, self.col_num), dtype=int)
        dec_ct22 = np.zeros((self.stage_num, self.col_num), dtype=int)
        dec_ct42 = np.zeros((self.stage_num, self.col_num), dtype=int)

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
                    elif type_idx == 4:
                        ct42[c] += 1
                        dec_ct42[s, c] += 1
                    else:
                        raise ValueError
        carry_num = 0
        for c in range(self.col_num):
            remain_pp[c] = pp[c] + carry_num - 2 * ct32[c] - ct22[c] - 3 * ct42[c]
            carry_num = ct32[c] + ct22[c] + 2 * ct42[c]
        logging.info(f"remain_pp\n: {remain_pp}")

        self.remain_pp = remain_pp
        self.dec_ct32 = dec_ct32
        self.dec_ct22 = dec_ct22
        self.dec_ct42 = dec_ct42
        self.ct32 = ct32
        self.ct22 = ct22
        self.ct42 = ct42
        self.slice_size = np.zeros((self.stage_num + 1, self.col_num), dtype=int)
        self.slice_size[0, :] = pp
        for s in range(1, self.stage_num + 1):
            self.slice_size[s, 0] = (
                self.slice_size[s - 1, 0]
                - dec_ct32[s - 1, 0] * 2
                - dec_ct22[s - 1, 0]
                - dec_ct42[s - 1, 0] * 3
            )
            for c in range(1, self.col_num):
                self.slice_size[s, c] = (
                    self.slice_size[s - 1, c]
                    - dec_ct32[s - 1, c] * 2
                    - dec_ct22[s - 1, c]
                    - dec_ct42[s - 1, c] * 3
                    + dec_ct32[s - 1, c - 1]
                    + dec_ct22[s - 1, c - 1]
                    + dec_ct42[s - 1, c - 1] * 2
                )
        self.port_size = np.zeros((self.stage_num + 1, self.col_num), dtype=int)
        for s in range(self.stage_num):
            for c in range(self.col_num):
                self.port_size[s, c] = (
                    3 * dec_ct32[s, c] + 2 * dec_ct22[s, c] + 4 * dec_ct42[s, c]
                )
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
            type_onehot = np.zeros(self.num_node_types)
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
            elif dst_type_idx == 4:
                edge_index_a.append((src_idx, dst_idx))
                edge_index_b.append((src_idx, dst_idx))
                edge_index_c.append((src_idx, dst_idx))
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
                if src_stage_idx < self.stage_num - 1:
                    for dst_idx in range(src_idx + 1, num_nodes):
                        dst_info = self.vertex_list[dst_idx]
                        dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
                        if (
                            src_col_idx == dst_col_idx
                            and src_stage_idx + 1 == dst_stage_idx
                        ):
                            __add_edge_index(src_idx, dst_idx, dst_type_idx)
                    if src_col_idx < self.col_num - 1 and src_type_idx != 3:
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
            (self.port_num, len(src_indices), len(dst_indices)), True, dtype=torch.bool
        )
        for local_dst_idx, dst_idx in enumerate(dst_indices):
            dst_info = self.vertex_list[dst_idx]
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            if dst_type_idx == 0:
                if self.port_num > 3:
                    mask[3:, :, local_dst_idx] = False
            elif dst_type_idx == 4:
                pass
            elif dst_type_idx == 1:
                mask[2, :, local_dst_idx] = False
                if self.port_num > 3:
                    mask[3:, :, local_dst_idx] = False
            elif dst_type_idx == 3:
                mask[1:, :, local_dst_idx] = False
            else:
                raise ValueError
        return mask

    def get_slice_carry_sources(self, s, c):
        """Return routable carry-output events from column c-1 into (s, c).

        FA/HA emit one carry. CT42 emits two same-weight carry outputs, so it appears
        twice with distinct output names.
        """
        src_indices = self.slice_indice_map[(s - 1, c - 1)]
        out = []
        for src_idx in src_indices:
            _ss, _cc, src_type_idx, _ii = self.vertex_list[src_idx]
            if src_type_idx in (0, 1):
                out.append((src_idx, "carry"))
            elif src_type_idx == 4:
                out.append((src_idx, "carry"))
                out.append((src_idx, "cout"))
        return out

    def get_slice_carry_mask(self, s, c) -> torch.Tensor:
        carry_sources = self.get_slice_carry_sources(s, c)
        dst_indices = self.slice_indice_map[(s, c)]
        mask = torch.full(
            (self.port_num, len(carry_sources), len(dst_indices)), True, dtype=torch.bool
        )
        for local_dst_idx, dst_idx in enumerate(dst_indices):
            dst_info = self.vertex_list[dst_idx]
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            if dst_type_idx == 0:
                if self.port_num > 3:
                    mask[3:, :, local_dst_idx] = False
            elif dst_type_idx == 4:
                pass
            elif dst_type_idx == 1:
                mask[2, :, local_dst_idx] = False
                if self.port_num > 3:
                    mask[3:, :, local_dst_idx] = False
            elif dst_type_idx == 3:
                mask[1:, :, local_dst_idx] = False
            else:
                raise ValueError
        return mask
