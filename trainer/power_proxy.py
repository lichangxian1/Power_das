import logging
import os
import sys
from typing import Optional, Tuple

import torch

from .augment_features import DEFAULT_TYPE_DELAYS, compute_timing_features
from .enrich_features import compute_graph_features
from .proxy_mlp import ArithProxyGNN, OneHotGIN, PureGCN, PureGIN


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.enrich_mis_physics_features import (  # noqa: E402
    add_mis_physics_features,
    build_physical_tables,
)


ARRIVAL_NORM = 30.0
FANOUT_NORM = 16.0


def _resolve_repo_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def extract_x_edge(comp_graph, samples_connection) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an arith-das routed sample into the proxy dataset graph format."""
    num_nodes = len(comp_graph.vertex_list)

    x_features = []
    for stage_idx, col_idx, type_idx, idx in comp_graph.vertex_list:
        type_onehot = [0.0, 0.0, 0.0, 0.0]
        type_onehot[type_idx] = 1.0
        x_features.append([float(stage_idx), float(col_idx), float(idx)] + type_onehot)
    x = torch.tensor(x_features, dtype=torch.float32)

    src_list, dst_list, attr_list = [], [], []
    for src_idx, dst_idx, dst_connec_type, _ in samples_connection:
        if not (0 <= src_idx < num_nodes and 0 <= dst_idx < num_nodes):
            continue
        src_col = comp_graph.vertex_list[src_idx][1]
        dst_col = comp_graph.vertex_list[dst_idx][1]
        is_sum = 1.0 if src_col == dst_col else 0.0
        is_carry = 1.0 if src_col + 1 == dst_col else 0.0
        port = [0.0, 0.0, 0.0]
        if 0 <= dst_connec_type <= 2:
            port[dst_connec_type] = 1.0
        src_list.append(src_idx)
        dst_list.append(dst_idx)
        attr_list.append([is_sum, is_carry] + port)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    if attr_list:
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)
    else:
        edge_attr = torch.zeros(0, 5, dtype=torch.float32)
    return x, edge_index, edge_attr


def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_index.numel() > 0:
        adj[edge_index[0].long(), edge_index[1].long()] = 1.0
    return adj


def _ensure_x9(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if x.shape[1] >= 9:
        return x[:, :9]
    adj = _edge_index_to_dense_adj(edge_index, x.shape[0])
    arrival, skew = compute_timing_features(x[:, :7], adj, DEFAULT_TYPE_DELAYS)
    return torch.cat([x[:, :7], arrival.unsqueeze(1), skew.unsqueeze(1)], dim=1)


def _ensure_x13(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if x.shape[1] >= 13:
        return x[:, :13]
    x9 = _ensure_x9(x, edge_index)
    fanout, fanin, depth, critical = compute_graph_features(x9, edge_index)
    return torch.cat(
        [
            x9,
            fanout.unsqueeze(1),
            fanin.unsqueeze(1),
            depth.unsqueeze(1),
            critical.unsqueeze(1),
        ],
        dim=1,
    )


def _add_edge10_features(
    x13: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr5: torch.Tensor,
) -> torch.Tensor:
    src = edge_index[0].long()
    dst = edge_index[1].long()
    if edge_index.numel() == 0:
        return torch.zeros(0, 10, dtype=torch.float32)

    arrival = x13[:, 7]
    fanout = x13[:, 9]
    fanin = x13[:, 10]

    new_features = torch.stack(
        [
            (arrival[src] - arrival[dst]) / ARRIVAL_NORM,
            arrival[src] / ARRIVAL_NORM,
            arrival[dst] / ARRIVAL_NORM,
            fanout[src] / FANOUT_NORM,
            fanin[dst] / 3.0,
        ],
        dim=1,
    )
    return torch.cat([edge_attr5[:, :5], new_features], dim=1)


def _build_model_from_ckpt(ckpt, device):
    model_class = ckpt.get("model_class", "")
    if model_class == "OneHotGIN" or ckpt.get("use_onehot_only", False):
        model = OneHotGIN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
            dropout=ckpt.get("dropout", 0.0),
            onehot_start=ckpt.get("onehot_start", 3),
            onehot_dim=ckpt.get("onehot_dim", 4),
        )
    elif ckpt.get("use_pure_gcn", False):
        model = PureGCN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
        )
    elif ckpt.get("use_pure_gin", False):
        model = PureGIN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
        )
    else:
        model = ArithProxyGNN(
            node_feature_dim=ckpt.get("node_feature_dim", 13),
            hidden_dim=ckpt.get("hidden_dim", 96),
            num_gnn_layers=ckpt.get("num_gnn_layers", 4),
            dropout=ckpt.get("dropout", 0.15),
            use_mean_agg=ckpt.get("use_mean_agg", True),
            use_edge_feat=ckpt.get("use_edge_feat", True),
            external_edge_attr_dim=ckpt.get("external_edge_attr_dim", 0),
            use_typed_edges=ckpt.get("use_typed_edges", False),
            use_multitask=ckpt.get("use_multitask", False),
            use_jk_pool=ckpt.get("use_jk_pool", False),
            use_gin=bool(ckpt.get("use_gin", False)),
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        logging.warning(
            "Power proxy checkpoint loaded with missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class PowerProxyPredictor:
    """Runtime wrapper for ArithProxyGNN checkpoints used inside arith-das."""

    def __init__(
        self,
        ckpt_path: str,
        device,
        lib_path: str,
        fa_cell: str,
        ha_cell: str,
    ):
        self.device = torch.device(device)
        self.ckpt_path = _resolve_repo_path(ckpt_path)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)

        self.model = _build_model_from_ckpt(ckpt, self.device)
        self.node_feature_dim = int(ckpt.get("node_feature_dim", 7))
        self.edge_attr_dim = int(ckpt.get("external_edge_attr_dim", 0) or 0)
        self.power_mean = torch.tensor(float(ckpt["power_mean"]), device=self.device)
        self.power_std = torch.tensor(float(ckpt["power_std"]), device=self.device)
        self.target = ckpt.get("target", "power") or "power"

        self.fa_info = None
        self.ha_info = None
        if self.node_feature_dim >= 28 or self.edge_attr_dim >= 14:
            resolved_lib = _resolve_repo_path(lib_path)
            self.fa_info, self.ha_info = build_physical_tables(
                resolved_lib,
                fa_cell,
                ha_cell,
            )

        logging.info(
            "Loaded power proxy %s: X=%d edge_attr=%d target=%s",
            self.ckpt_path,
            self.node_feature_dim,
            self.edge_attr_dim,
            self.target,
        )

    def _prepare_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        needs_x9 = self.node_feature_dim >= 9
        needs_x13 = self.node_feature_dim >= 13 or self.edge_attr_dim >= 10
        needs_phys = self.node_feature_dim >= 28 or self.edge_attr_dim >= 14

        if needs_x13 or needs_phys:
            x = _ensure_x13(x, edge_index)
        elif needs_x9:
            x = _ensure_x9(x, edge_index)
        else:
            x = x[:, : self.node_feature_dim]

        edge_attr_work = edge_attr[:, :5]
        if self.edge_attr_dim >= 10 or needs_phys:
            x13 = _ensure_x13(x, edge_index)
            edge_attr_work = _add_edge10_features(x13, edge_index, edge_attr_work)
            x = x13
        if needs_phys:
            if self.fa_info is None or self.ha_info is None:
                raise RuntimeError("Physical power proxy requires Liberty cell info")
            x, edge_attr_work = add_mis_physics_features(
                x,
                edge_index,
                edge_attr_work,
                self.fa_info,
                self.ha_info,
            )

        edge_attr_out = None
        if self.edge_attr_dim > 0:
            if edge_attr_work.shape[1] < self.edge_attr_dim:
                raise ValueError(
                    f"Power proxy needs edge_attr dim {self.edge_attr_dim}, "
                    f"got {edge_attr_work.shape[1]}"
                )
            edge_attr_out = edge_attr_work[:, : self.edge_attr_dim]

        if x.shape[1] < self.node_feature_dim:
            raise ValueError(
                f"Power proxy needs X dim {self.node_feature_dim}, got {x.shape[1]}"
            )
        x = x[:, : self.node_feature_dim]
        return x, edge_attr_out

    @torch.no_grad()
    def predict_mw(self, comp_graph, samples_connection) -> float:
        x, edge_index, edge_attr = extract_x_edge(comp_graph, samples_connection)
        x, edge_attr = self._prepare_features(x, edge_index, edge_attr)

        x = x.unsqueeze(0).to(self.device)
        edge_index = edge_index.to(self.device)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        pred_norm = self.model(x, edge_index, mask, edge_attr=edge_attr)
        pred_mw = pred_norm * self.power_std + self.power_mean
        return float(pred_mw.squeeze().detach().cpu().item())
