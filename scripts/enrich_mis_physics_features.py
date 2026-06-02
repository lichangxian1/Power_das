"""
Add MIS-inspired physical features using real FA/HA data from a Liberty file.

Input dataset:
  - X[:, :13] follows the existing enriched format
  - edge_index [2, E]
  - edge_attr [E, 5] or [E, 10]

Added node features (15 dims):
  13: pin_cap_a
  14: pin_cap_b
  15: pin_cap_c
  16: input_arr_min
  17: input_arr_max
  18: input_arr_mean
  19: input_arr_range
  20: delta_ab
  21: delta_ac
  22: delta_bc
  23: simultaneity
  24: load_sum
  25: load_carry
  26: load_total
  27: cell_area

Added edge features (4 dims, appended to existing edge_attr):
  - dst_pin_cap
  - src_load_total
  - dst_cell_area
  - wire_span

All capacitance features are normalized by CAP_NORM_PF. Area is normalized
by AREA_NORM. Arrival-derived features are normalized by ARRIVAL_NORM.
"""

import argparse
import os
import re
from dataclasses import dataclass

import torch
from tqdm import tqdm


ARRIVAL_NORM = 30.0
CAP_NORM_PF = 0.05115
AREA_NORM = 3.0
SIM_TAU = 3.0
WIRE_SPAN_NORM = 8.0


@dataclass
class CellInfo:
    name: str
    area: float
    input_caps: dict
    output_max_caps: dict


def _extract_balanced_block(text, start):
    depth = 0
    end = start
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:end + 1]
        end += 1
    raise ValueError("Unbalanced Liberty block")


def parse_liberty_cell(lib_path, cell_name):
    with open(lib_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = re.search(rf"cell\s*\({re.escape(cell_name)}\)\s*\{{", text)
    if not m:
        raise ValueError(f"Cell {cell_name} not found in {lib_path}")

    block = _extract_balanced_block(text, m.start())
    area_m = re.search(r"\barea\s*:\s*([0-9.eE+-]+)\s*;", block)
    area = float(area_m.group(1)) if area_m else 0.0

    input_caps = {}
    output_max_caps = {}
    for pin_m in re.finditer(r"pin\s*\(([^)]+)\)\s*\{", block):
        pin_name = pin_m.group(1).strip()
        pin_block = _extract_balanced_block(block, pin_m.start())
        direction_m = re.search(r"\bdirection\s*:\s*(\w+)\s*;", pin_block)
        direction = direction_m.group(1) if direction_m else ""
        if direction == "input":
            cap_m = re.search(r"\bcapacitance\s*:\s*([0-9.eE+-]+)\s*;", pin_block)
            if cap_m:
                input_caps[pin_name] = float(cap_m.group(1))
        elif direction == "output":
            max_m = re.search(r"\bmax_capacitance\s*:\s*([0-9.eE+-]+)\s*;", pin_block)
            if max_m:
                output_max_caps[pin_name] = float(max_m.group(1))

    return CellInfo(cell_name, area, input_caps, output_max_caps)


def build_physical_tables(lib_path, fa_cell, ha_cell):
    fa = parse_liberty_cell(lib_path, fa_cell)
    ha = parse_liberty_cell(lib_path, ha_cell)
    print(f"  📚 Liberty cells:")
    print(f"     FA {fa.name}: area={fa.area:.4f}, caps={fa.input_caps}, max_caps={fa.output_max_caps}")
    print(f"     HA {ha.name}: area={ha.area:.4f}, caps={ha.input_caps}, max_caps={ha.output_max_caps}")
    return fa, ha


def _dst_pin_cap(type_idx, port_idx, fa, ha):
    if type_idx == 0:
        return [fa.input_caps.get("A", 0.0),
                fa.input_caps.get("B", 0.0),
                fa.input_caps.get("CI", 0.0)][port_idx]
    if type_idx == 1:
        return [ha.input_caps.get("A", 0.0),
                ha.input_caps.get("B", 0.0),
                0.0][port_idx]
    return 0.0


def add_mis_physics_features(X, edge_index, edge_attr, fa, ha):
    if X.shape[1] < 13:
        raise ValueError(f"Expected X dim >= 13, got {X.shape[1]}")
    if edge_attr.shape[1] < 5:
        raise ValueError(f"Expected edge_attr dim >= 5, got {edge_attr.shape[1]}")

    N = X.shape[0]
    E = edge_index.shape[1]
    type_idx = X[:, 3:7].argmax(dim=1).long()
    stage = X[:, 0]
    col = X[:, 1]
    arrival = X[:, 7]

    src = edge_index[0].long()
    dst = edge_index[1].long()
    port_idx = edge_attr[:, 2:5].argmax(dim=1).long() if E else torch.empty(0, dtype=torch.long)
    is_sum = edge_attr[:, 0] > 0.5 if E else torch.empty(0, dtype=torch.bool)
    is_carry = edge_attr[:, 1] > 0.5 if E else torch.empty(0, dtype=torch.bool)

    # Input pin capacitance by node type/port.
    pin_caps = torch.zeros(N, 3, dtype=torch.float32)
    fa_mask = type_idx == 0
    ha_mask = type_idx == 1
    pin_caps[fa_mask, 0] = fa.input_caps.get("A", 0.0)
    pin_caps[fa_mask, 1] = fa.input_caps.get("B", 0.0)
    pin_caps[fa_mask, 2] = fa.input_caps.get("CI", 0.0)
    pin_caps[ha_mask, 0] = ha.input_caps.get("A", 0.0)
    pin_caps[ha_mask, 1] = ha.input_caps.get("B", 0.0)

    cell_area = torch.zeros(N, dtype=torch.float32)
    cell_area[fa_mask] = fa.area
    cell_area[ha_mask] = ha.area

    # Arrival per destination port, gathered from incoming edges.
    port_arr = torch.zeros(N, 3, dtype=torch.float32)
    port_seen = torch.zeros(N, 3, dtype=torch.bool)
    for e in range(E):
        d = int(dst[e])
        p = int(port_idx[e])
        if 0 <= p < 3:
            port_arr[d, p] = arrival[int(src[e])]
            port_seen[d, p] = True

    valid_count = port_seen.float().sum(dim=1).clamp_min(1.0)
    masked_arr = torch.where(port_seen, port_arr, torch.zeros_like(port_arr))
    arr_min = torch.where(port_seen, port_arr, torch.full_like(port_arr, float("inf"))).min(dim=1).values
    arr_max = torch.where(port_seen, port_arr, torch.full_like(port_arr, float("-inf"))).max(dim=1).values
    arr_min = torch.where(torch.isinf(arr_min), torch.zeros_like(arr_min), arr_min)
    arr_max = torch.where(torch.isinf(arr_max), torch.zeros_like(arr_max), arr_max)
    arr_mean = masked_arr.sum(dim=1) / valid_count
    arr_range = arr_max - arr_min

    delta_ab = torch.where(port_seen[:, 0] & port_seen[:, 1], port_arr[:, 0] - port_arr[:, 1], torch.zeros(N))
    delta_ac = torch.where(port_seen[:, 0] & port_seen[:, 2], port_arr[:, 0] - port_arr[:, 2], torch.zeros(N))
    delta_bc = torch.where(port_seen[:, 1] & port_seen[:, 2], port_arr[:, 1] - port_arr[:, 2], torch.zeros(N))

    pair_seen = torch.stack([
        port_seen[:, 0] & port_seen[:, 1],
        port_seen[:, 0] & port_seen[:, 2],
        port_seen[:, 1] & port_seen[:, 2],
    ], dim=1)
    pair_delta = torch.stack([delta_ab, delta_ac, delta_bc], dim=1)
    pair_score = torch.exp(-pair_delta.abs() / SIM_TAU) * pair_seen.float()
    simultaneity = pair_score.sum(dim=1) / pair_seen.float().sum(dim=1).clamp_min(1.0)
    simultaneity = torch.where(pair_seen.any(dim=1), simultaneity, torch.zeros_like(simultaneity))

    # True-ish output load proxy: sum downstream input pin caps by driven output branch.
    edge_dst_cap = torch.zeros(E, dtype=torch.float32)
    for e in range(E):
        edge_dst_cap[e] = _dst_pin_cap(int(type_idx[int(dst[e])]), int(port_idx[e]), fa, ha)

    load_sum = torch.zeros(N, dtype=torch.float32)
    load_carry = torch.zeros(N, dtype=torch.float32)
    if E:
        load_sum.scatter_add_(0, src[is_sum], edge_dst_cap[is_sum])
        load_carry.scatter_add_(0, src[is_carry], edge_dst_cap[is_carry])
    load_total = load_sum + load_carry

    node_new = torch.stack([
        pin_caps[:, 0] / CAP_NORM_PF,
        pin_caps[:, 1] / CAP_NORM_PF,
        pin_caps[:, 2] / CAP_NORM_PF,
        arr_min / ARRIVAL_NORM,
        arr_max / ARRIVAL_NORM,
        arr_mean / ARRIVAL_NORM,
        arr_range / ARRIVAL_NORM,
        delta_ab / ARRIVAL_NORM,
        delta_ac / ARRIVAL_NORM,
        delta_bc / ARRIVAL_NORM,
        simultaneity,
        load_sum / CAP_NORM_PF,
        load_carry / CAP_NORM_PF,
        load_total / CAP_NORM_PF,
        cell_area / AREA_NORM,
    ], dim=1)

    edge_new = None
    if E:
        src_load_total = load_total[src] / CAP_NORM_PF
        dst_cell_area = cell_area[dst] / AREA_NORM
        wire_span = ((stage[dst] - stage[src]).abs() + (col[dst] - col[src]).abs()) / WIRE_SPAN_NORM
        edge_new = torch.stack([
            edge_dst_cap / CAP_NORM_PF,
            src_load_total,
            dst_cell_area,
            wire_span,
        ], dim=1)
    else:
        edge_new = torch.zeros(0, 4, dtype=torch.float32)

    return torch.cat([X, node_new], dim=1), torch.cat([edge_attr, edge_new], dim=1)


def print_feature_stats(data, old_x_dim, old_e_dim):
    x = torch.cat([item["X"][:, old_x_dim:] for item in data], dim=0)
    ea = torch.cat([item["edge_attr"][:, old_e_dim:] for item in data], dim=0)
    node_names = [
        "pin_cap_a", "pin_cap_b", "pin_cap_c",
        "input_arr_min", "input_arr_max", "input_arr_mean", "input_arr_range",
        "delta_ab", "delta_ac", "delta_bc", "simultaneity",
        "load_sum", "load_carry", "load_total", "cell_area",
    ]
    edge_names = ["dst_pin_cap", "src_load_total", "dst_cell_area", "wire_span"]
    print("\n  📊 新增节点特征统计:")
    for i, name in enumerate(node_names):
        v = x[:, i]
        print(f"     X+{i:02d} {name:16s}: min={v.min():+.4f} mean={v.mean():+.4f} "
              f"max={v.max():+.4f} std={v.std():.4f}")
    print("\n  📊 新增边特征统计:")
    for i, name in enumerate(edge_names):
        v = ea[:, i]
        print(f"     E+{i:02d} {name:16s}: min={v.min():+.4f} mean={v.mean():+.4f} "
              f"max={v.max():+.4f} std={v.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", default="dataset/glitch_power_data_16bit_v2_9k_edge10.pt")
    parser.add_argument("--out_path", default="dataset/glitch_power_data_16bit_v2_9k_phys.pt")
    parser.add_argument("--lib_path",
                        default="library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib")
    parser.add_argument("--fa_cell", default="FA1D0BWP12T40P140")
    parser.add_argument("--ha_cell", default="HA1D0BWP12T40P140")
    args = parser.parse_args()

    fa, ha = build_physical_tables(args.lib_path, args.fa_cell, args.ha_cell)

    data = torch.load(args.in_path, map_location="cpu", weights_only=False)
    print(f"\n  📂 加载 {len(data)} 样本: {args.in_path}")
    old_x_dim = data[0]["X"].shape[1]
    old_e_dim = data[0]["edge_attr"].shape[1]
    print(f"     old X dim={old_x_dim}, old edge_attr dim={old_e_dim}")

    for item in tqdm(data, desc="添加 MIS/physics 特征"):
        X_new, edge_attr_new = add_mis_physics_features(
            item["X"], item["edge_index"], item["edge_attr"], fa, ha
        )
        item["X"] = X_new
        item["edge_attr"] = edge_attr_new
        item["physics_feature_names"] = {
            "node_added": [
                "pin_cap_a", "pin_cap_b", "pin_cap_c",
                "input_arr_min", "input_arr_max", "input_arr_mean", "input_arr_range",
                "delta_ab", "delta_ac", "delta_bc", "simultaneity",
                "load_sum", "load_carry", "load_total", "cell_area",
            ],
            "edge_added": ["dst_pin_cap", "src_load_total", "dst_cell_area", "wire_span"],
            "lib_cells": {"FA": fa.name, "HA": ha.name},
        }

    print(f"\n  ✅ 完成: X {old_x_dim}→{data[0]['X'].shape[1]}, "
          f"edge_attr {old_e_dim}→{data[0]['edge_attr'].shape[1]}")
    print_feature_stats(data, old_x_dim, old_e_dim)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(data, args.out_path)
    print(f"\n  💾 保存到: {args.out_path}")


if __name__ == "__main__":
    main()
