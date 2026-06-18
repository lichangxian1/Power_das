#!/usr/bin/env python3
"""Append real EDA observables (STA + SAIF toggles) to graph features.

This script is intentionally post-processing only: data collection stores raw
node_timing/node_toggles tensors, and this script converts them into model input
features while preserving the original dataset.
"""
import argparse
import math
import os
import sys
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_dataset import NODE_TIMING_FEATURES  # noqa: E402

NODE_FEATURE_NAMES = [f"eda_sta_{name}_over_time" for name in NODE_TIMING_FEATURES] + [
    "eda_sta_mask",
    "eda_toggle_density",
    "eda_toggle_log",
    "eda_toggle_mask",
]

EDGE_FEATURE_NAMES = [
    "eda_src_out_arr_over_time",
    "eda_dst_in_arr_max_over_time",
    "eda_arrival_gap_over_time",
    "eda_src_toggle_density",
    "eda_dst_toggle_density",
    "eda_sta_edge_mask",
    "eda_toggle_edge_mask",
]


def _bool_any(value):
    return bool(value is not None and torch.as_tensor(value).bool().any().item())


def _canonical_timing(item, num_nodes):
    timing = item.get("node_timing")
    if timing is None:
        return torch.zeros((num_nodes, len(NODE_TIMING_FEATURES)), dtype=torch.float32)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    if timing.ndim != 2 or timing.shape[0] != num_nodes:
        return torch.zeros((num_nodes, len(NODE_TIMING_FEATURES)), dtype=torch.float32)

    names = item.get("node_timing_features") or NODE_TIMING_FEATURES
    if list(names) == list(NODE_TIMING_FEATURES) and timing.shape[1] == len(NODE_TIMING_FEATURES):
        return torch.nan_to_num(timing, nan=0.0, posinf=0.0, neginf=0.0)

    out = torch.zeros((num_nodes, len(NODE_TIMING_FEATURES)), dtype=torch.float32)
    for dst_idx, name in enumerate(NODE_TIMING_FEATURES):
        if name in names:
            src_idx = list(names).index(name)
            if src_idx < timing.shape[1]:
                out[:, dst_idx] = timing[:, src_idx]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _node_mask(item, key, num_nodes):
    mask = item.get(key)
    if mask is None:
        return torch.zeros(num_nodes, dtype=torch.bool)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.bool)
    return mask


def _node_vector(item, key, num_nodes):
    vec = item.get(key)
    if vec is None:
        return torch.zeros(num_nodes, dtype=torch.float32)
    vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
    if vec.numel() != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.float32)
    return torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _positive_denominator(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value) or value <= 0:
        return fallback
    return value


def enrich_item(item, time_norm=2.0, toggle_norm=4096.0, allow_missing=False):
    if item.get("eda_observables_enriched"):
        raise ValueError("input item already has eda_observables_enriched=True")

    out = deepcopy(item)
    x = torch.as_tensor(item["X"], dtype=torch.float32)
    edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
    edge_attr = torch.as_tensor(item["edge_attr"], dtype=torch.float32)
    num_nodes = x.shape[0]

    timing = _canonical_timing(item, num_nodes)
    timing_mask = _node_mask(item, "node_timing_mask", num_nodes)
    toggles = torch.clamp(_node_vector(item, "node_toggles", num_nodes), min=0.0)
    toggle_mask = _node_mask(item, "node_toggle_mask", num_nodes)

    has_timing = _bool_any(timing_mask)
    has_toggle = _bool_any(toggle_mask)
    if not allow_missing and not (has_timing or has_toggle):
        raise ValueError("item has neither node_timing nor node_toggles; pass --allow_missing for old datasets")

    time_scale = _positive_denominator(time_norm, 2.0)
    fallback_toggle_scale = _positive_denominator(toggle_norm, 4096.0)
    vec_cnt = _positive_denominator(item.get("vec_cnt"), fallback_toggle_scale)

    timing_norm = timing / time_scale
    toggle_density = (toggles / vec_cnt).unsqueeze(1)
    toggle_log = (torch.log1p(toggles) / math.log1p(fallback_toggle_scale)).unsqueeze(1)

    node_extra = torch.cat([
        timing_norm,
        timing_mask.float().unsqueeze(1),
        toggle_density,
        toggle_log,
        toggle_mask.float().unsqueeze(1),
    ], dim=1)
    out["X"] = torch.cat([x, node_extra], dim=1)

    if edge_index.numel() > 0:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        out_arr_idx = NODE_TIMING_FEATURES.index("out_arr_max")
        in_arr_idx = NODE_TIMING_FEATURES.index("in_arr_max")
        src_out = timing[src, out_arr_idx] / time_scale
        dst_in = timing[dst, in_arr_idx] / time_scale
        src_toggle = toggles[src] / vec_cnt
        dst_toggle = toggles[dst] / vec_cnt
        edge_extra = torch.stack([
            src_out,
            dst_in,
            src_out - dst_in,
            src_toggle,
            dst_toggle,
            (timing_mask[src] & timing_mask[dst]).float(),
            (toggle_mask[src] & toggle_mask[dst]).float(),
        ], dim=1)
    else:
        edge_extra = torch.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32)
    out["edge_attr"] = torch.cat([edge_attr, edge_extra], dim=1)

    out["eda_observables_enriched"] = True
    out["eda_observable_node_feature_names"] = list(NODE_FEATURE_NAMES)
    out["eda_observable_edge_feature_names"] = list(EDGE_FEATURE_NAMES)
    return out, has_timing, has_toggle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--time_norm", type=float, default=2.0,
                        help="STA time normalization in ns; target_delay=2.0 uses 2.0 by default")
    parser.add_argument("--toggle_norm", type=float, default=4096.0,
                        help="Fallback vector count/log scale when vec_cnt is missing")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N items for smoke tests")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Fill zero features for old samples without STA/toggle tensors")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data = torch.load(args.in_path, map_location="cpu", weights_only=False)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    enriched = []
    timing_count = 0
    toggle_count = 0
    for item in data:
        new_item, has_timing, has_toggle = enrich_item(
            item,
            time_norm=args.time_norm,
            toggle_norm=args.toggle_norm,
            allow_missing=args.allow_missing,
        )
        enriched.append(new_item)
        timing_count += int(has_timing)
        toggle_count += int(has_toggle)

    if enriched:
        print(f"input samples: {len(data)}")
        print(f"timing coverage: {timing_count}/{len(data)}")
        print(f"toggle coverage: {toggle_count}/{len(data)}")
        print(f"X dim: {data[0]['X'].shape[1]} -> {enriched[0]['X'].shape[1]}")
        print(f"edge_attr dim: {data[0]['edge_attr'].shape[1]} -> {enriched[0]['edge_attr'].shape[1]}")
        print("node extras:", ", ".join(NODE_FEATURE_NAMES))
        print("edge extras:", ", ".join(EDGE_FEATURE_NAMES))

    if args.dry_run:
        print("dry_run: not saving")
        return

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(enriched, args.out_path)
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
