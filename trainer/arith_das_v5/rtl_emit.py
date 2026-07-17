"""RTL 发射：把采样出的连接矩阵 + cell 映射逐节点翻译成 Verilog 乘法器。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
from typing import Dict, Set
import time

import torch


class RtlEmitMixin:
    """把连接矩阵 + cell 映射发射成 Verilog 乘法器 RTL。"""

    @staticmethod
    def _add_node(node_id, node_type, node_wires):
        if node_id not in node_wires:
            if node_type == 0:
                node_wires[node_id] = {
                    "from": {"a": None, "b": None, "c": None},
                    "to": {"sum": None, "carry": None},
                }
            elif node_type == 4:
                node_wires[node_id] = {
                    "from": {"a": None, "b": None, "c": None, "d": None},
                    "to": {"sum": None, "carry": None, "cout": None},
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

    @staticmethod
    def _edge_ref(src_idx, src_output):
        return int(src_idx), str(src_output)

    @staticmethod
    def _wire_from_ref(src_ref, dst_idx):
        if src_ref is None or dst_idx is None:
            return None
        src_idx, src_output = src_ref
        return f"from_{src_idx}_{src_output}_to_{dst_idx}"

    def _wire_from_output(self, src_idx, src_output, dst_idx):
        return self._wire_from_ref(self._edge_ref(src_idx, src_output), dst_idx)

    def _input_wire(self, node_wires: Dict, node_idx, input_port):
        src_ref = node_wires[node_idx]["from"][input_port]
        if src_ref is None:
            raise ValueError(f"unrouted input {input_port} for node {node_idx}")
        return self._wire_from_ref(src_ref, node_idx)

    def _output_wire(self, node_wires: Dict, node_idx, output_port):
        dst_idx = node_wires[node_idx]["to"][output_port]
        return self._wire_from_output(node_idx, output_port, dst_idx)

    def _declare_pp(self, node_idx, wire_set: Set, node_wires: Dict):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 2
        v_src = ""
        instance_name = f"pp_{col_idx}[{idx}]"
        sum_wire = self._output_wire(node_wires, node_idx, "sum")
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

        a_wire = self._input_wire(node_wires, node_idx, "a")
        if stage_idx < self.comp_graph.stage_num:
            sum_wire = self._output_wire(node_wires, node_idx, "sum")
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

        a_wire = self._input_wire(node_wires, node_idx, "a")
        b_wire = self._input_wire(node_wires, node_idx, "b")
        c_wire = self._input_wire(node_wires, node_idx, "c")

        sum_wire = self._output_wire(node_wires, node_idx, "sum")
        if node_wires[node_idx]["to"]["carry"] is not None:
            carry_wire = self._output_wire(node_wires, node_idx, "carry")
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

        a_wire = self._input_wire(node_wires, node_idx, "a")
        b_wire = self._input_wire(node_wires, node_idx, "b")
        sum_wire = self._output_wire(node_wires, node_idx, "sum")
        if node_wires[node_idx]["to"]["carry"] is not None:
            carry_wire = self._output_wire(node_wires, node_idx, "carry")
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

    def _declare_ct42(self, node_idx, wire_set: Set, node_wires: Dict, cell_map=None):
        stage_idx, col_idx, type_idx, idx = self.comp_graph.vertex_list[node_idx]
        assert type_idx == 4
        v_src = ""
        instance_name = f"ct42_{node_idx}"
        cell = (cell_map or {}).get(node_idx) or "CT42"

        a_wire = self._input_wire(node_wires, node_idx, "a")
        b_wire = self._input_wire(node_wires, node_idx, "b")
        c_wire = self._input_wire(node_wires, node_idx, "c")
        d_wire = self._input_wire(node_wires, node_idx, "d")
        sum_wire = self._output_wire(node_wires, node_idx, "sum")
        carry_dst = node_wires[node_idx]["to"]["carry"]
        cout_dst = node_wires[node_idx]["to"]["cout"]
        if carry_dst is None or cout_dst is None:
            raise ValueError("CT42 must not be placed in the final carryless column")
        carry_wire = self._wire_from_output(node_idx, "carry", carry_dst)
        cout_wire = self._wire_from_output(node_idx, "cout", cout_dst)

        for wire in [a_wire, b_wire, c_wire, d_wire, sum_wire, carry_wire, cout_wire]:
            v, wire_set = self._declare_wire(wire, wire_set)
            v_src += v
        v_src += f"// ct42 node {(stage_idx, col_idx, type_idx, idx)}\n"
        if cell == "CT42":
            v_src += (
                f"    CT42 {instance_name} (.a({a_wire}), .b({b_wire}), .c({c_wire}), "
                f".d({d_wire}), .sum({sum_wire}), .carry({carry_wire}), .cout({cout_wire}));\n"
            )
        else:
            native4 = (
                cell in getattr(self, "_ct42_native4_names", set())
                or cell.startswith("comp42n_")
            )
            cin_part = "" if native4 else ".cin(1'b0), "
            v_src += (
                f"    {cell} {instance_name} (.a({a_wire}), .b({b_wire}), .c({c_wire}), "
                f".d({d_wire}), {cin_part}.sum({sum_wire}), "
                f".carry({carry_wire}), .cout({cout_wire}));\n"
            )
        return v_src, wire_set

    def emit_assignment(self, samples_connection, cell_map=None):
        node_wires = {}
        INPUT_PORTS = ["a", "b", "c", "d"]

        def connect(src_idx, dst_idx, dst_conc_type, src_output):
            input_port_name = INPUT_PORTS[dst_conc_type]
            assert input_port_name in node_wires[dst_idx]["from"]
            if node_wires[dst_idx]["from"][input_port_name] is not None:
                raise ValueError(
                    f"input {input_port_name} of node {dst_idx} is routed twice"
                )
            assert src_output in node_wires[src_idx]["to"]
            if node_wires[src_idx]["to"][src_output] is not None:
                raise ValueError(
                    f"output {src_output} of node {src_idx} is routed twice"
                )
            node_wires[dst_idx]["from"][input_port_name] = self._edge_ref(
                src_idx, src_output
            )
            node_wires[src_idx]["to"][src_output] = dst_idx

        for src_idx, dst_idx, dst_conc_type, meta in samples_connection:
            src_info = self.comp_graph.vertex_list[src_idx]
            dst_info = self.comp_graph.vertex_list[dst_idx]
            src_stage_idx, src_col_idx, src_type_idx, _ = src_info
            dst_stage_idx, dst_col_idx, dst_type_idx, _ = dst_info
            node_wires = self._add_node(src_idx, src_type_idx, node_wires)
            node_wires = self._add_node(dst_idx, dst_type_idx, node_wires)

            assert src_stage_idx + 1 == dst_stage_idx
            if src_col_idx == dst_col_idx:
                src_output = meta.get("src_output", "sum")
                if src_output != "sum":
                    raise ValueError(
                        f"same-column edge must use sum output, got {src_output}"
                    )
                connect(src_idx, dst_idx, dst_conc_type, src_output)
            elif src_col_idx + 1 == dst_col_idx:
                src_output = meta.get("src_output", "carry")
                connect(src_idx, dst_idx, dst_conc_type, src_output)
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
            elif type_idx == 4:
                v, wire_set = self._declare_ct42(node_idx, wire_set, node_wires, cell_map)
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
            if self.use_ct42:
                (
                    out_a,
                    out_b,
                    out_c,
                    out_d,
                    out_sum,
                    out_carry,
                    self._node_emb,
                ) = self.gcn.forward(
                    x,
                    edge_index_a,
                    edge_index_b,
                    edge_index_c,
                    return_embedding=True,
                    return_port_d=True,
                )
            else:
                out_a, out_b, out_c, out_sum, out_carry, self._node_emb = self.gcn.forward(
                    x, edge_index_a, edge_index_b, edge_index_c, return_embedding=True
                )
                out_d = None
        else:
            if self.use_ct42:
                out_a, out_b, out_c, out_d, out_sum, out_carry = self.gcn.forward(
                    x,
                    edge_index_a,
                    edge_index_b,
                    edge_index_c,
                    return_port_d=True,
                )
            else:
                out_a, out_b, out_c, out_sum, out_carry = self.gcn.forward(
                    x, edge_index_a, edge_index_b, edge_index_c
                )
                out_d = None
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
                port_outs = [out_a, out_b, out_c]
                port_keys = ["a", "b", "c"]
                if self.comp_graph.port_num > 3:
                    port_outs.append(out_d)
                    port_keys.append("d")
                for p_idx, (key, out_port) in enumerate(zip(port_keys, port_outs)):
                    z = out_sum[sum_src_indices, :] @ out_port[dst_indices, :].T
                    z = z.masked_fill(~sum_mask[p_idx, :, :], -1e9)
                    Z_mat_dict[(s, c)][f"s{key}"] = z
                if c > 0:
                    carry_sources = self.comp_graph.get_slice_carry_sources(s, c)
                    carry_src_indices = torch.tensor(
                        [src_idx for src_idx, _out_name in carry_sources],
                        device=self.device,
                        dtype=torch.long,
                    )
                    carry_mask = self.comp_graph.get_slice_carry_mask(s, c).to(
                        self.device
                    )
                    for p_idx, (key, out_port) in enumerate(zip(port_keys, port_outs)):
                        z = out_carry[carry_src_indices, :] @ out_port[dst_indices, :].T
                        z = z.masked_fill(~carry_mask[p_idx, :, :], -1e9)
                        Z_mat_dict[(s, c)][f"c{key}"] = z
        time_end = time.time()
        return Z_mat_dict
