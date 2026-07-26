"""Cross-entropy optimization over legal compressor-tree routings.

The state is a set of per-slice connection logits. Sampling adds Gumbel noise
and legalizes every slice with a maximum-weight bipartite matching (Hungarian
algorithm). Updating averages the complete matchings selected by the real
DC/Verilator objective, so the empirical target is doubly stochastic over the
usable input slots.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


RoutingLogits = Dict[Tuple[int, int], Dict[str, torch.Tensor]]


class RoutingCEM:
    """Gumbel-Hungarian sampler and smoothed elite-frequency updater."""

    _PORT_NAMES = ("a", "b", "c", "d")
    _ILLEGAL_LOGIT = -1.0e9

    @staticmethod
    def _clone_logits(logits: RoutingLogits) -> RoutingLogits:
        return {
            slice_key: {
                port: value.detach().clone()
                for port, value in value_dict.items()
            }
            for slice_key, value_dict in logits.items()
        }

    def __init__(
        self,
        engine,
        *,
        smoothing: float = 0.25,
        exploration: float = 0.05,
        temperature: float = 1.0,
        init_mode: str = "policy",
    ):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("CEM smoothing must be in (0, 1]")
        if not 0.0 <= exploration < 1.0:
            raise ValueError("CEM exploration must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("CEM temperature must be > 0")
        if init_mode not in ("policy", "uniform"):
            raise ValueError("CEM init_mode must be 'policy' or 'uniform'")
        self.engine = engine
        self.smoothing = float(smoothing)
        self.exploration = float(exploration)
        self.temperature = float(temperature)
        self.init_mode = init_mode

    def initialize(self, template: RoutingLogits) -> RoutingLogits:
        """Create detached direct logits while preserving all static masks."""
        out = self._clone_logits(template)
        for value_dict in out.values():
            for key, value in value_dict.items():
                value = value.detach().clone()
                legal = value > self._ILLEGAL_LOGIT / 2.0
                if self.init_mode == "uniform":
                    value = torch.where(
                        legal,
                        torch.zeros_like(value),
                        torch.full_like(value, self._ILLEGAL_LOGIT),
                    )
                else:
                    value = value.masked_fill(~legal, self._ILLEGAL_LOGIT)
                value_dict[key] = value
        return out

    @staticmethod
    def _route_signature(connection: Sequence[tuple]) -> tuple:
        return tuple(
            (
                int(src),
                int(dst),
                int(port),
                str((meta or {}).get("src_output", "sum")),
            )
            for src, dst, port, meta in connection
        )

    @staticmethod
    def _gumbel_like(value: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(value).clamp_(1.0e-7, 1.0 - 1.0e-7)
        return -torch.log(-torch.log(uniform))

    def sample(self, logits: RoutingLogits):
        """Sample one complete legal route with Gumbel-Hungarian matching."""
        mask_cache, z_cache = self.engine.get_cache(logits)
        connection = []
        matching_score = 0.0
        graph = self.engine.comp_graph

        for (stage, column), z_slice in z_cache.items():
            legal = mask_cache[(stage, column)]
            scores = z_slice / self.temperature + self._gumbel_like(z_slice)
            scores = scores.masked_fill(~legal, -1.0e12)
            rows, columns = linear_sum_assignment(
                -scores.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            if len(rows) != scores.shape[0] or not np.array_equal(
                rows, np.arange(scores.shape[0])
            ):
                raise RuntimeError(
                    f"CEM matching did not cover every source in slice "
                    f"{(stage, column)}: {len(rows)}/{scores.shape[0]}"
                )

            n_dst = len(graph.slice_indice_map[(stage, column)])
            sum_sources = graph.slice_indice_map[(stage - 1, column)]
            carry_sources = (
                graph.get_slice_carry_sources(stage, column) if column > 0 else []
            )
            for flat_row, flat_column in zip(rows.tolist(), columns.tolist()):
                if not bool(legal[flat_row, flat_column].item()):
                    raise RuntimeError(
                        f"CEM matching selected an illegal edge in slice "
                        f"{(stage, column)}"
                    )
                port = flat_column // n_dst
                local_dst = flat_column % n_dst
                dst_idx = graph.slice_indice_map[(stage, column)][local_dst]
                if flat_row < len(sum_sources):
                    local_src = flat_row
                    src_idx = sum_sources[local_src]
                    src_output = "sum"
                else:
                    local_src = flat_row - len(sum_sources)
                    src_idx, src_output = carry_sources[local_src]
                connection.append(
                    (
                        int(src_idx),
                        int(dst_idx),
                        int(port),
                        {
                            "cem_score": float(z_slice[flat_row, flat_column].item()),
                            "flat_row": int(flat_row),
                            "local_src_idx": int(local_src),
                            "local_dst_idx": int(local_dst),
                            "sample": int(flat_column),
                            "slice": (int(stage), int(column)),
                            "src_output": str(src_output),
                        },
                    )
                )
                matching_score += float(z_slice[flat_row, flat_column].item())

        return connection, matching_score

    def sample_many(
        self,
        logits: RoutingLogits,
        count: int,
        *,
        max_attempt_factor: int = 100,
    ):
        """Sample distinct routes so collapsed distributions do not waste DC calls."""
        if count < 1:
            return []
        sampled = []
        seen = set()
        max_attempts = max(count, int(count) * int(max_attempt_factor))
        for _ in range(max_attempts):
            connection, score = self.sample(logits)
            signature = self._route_signature(connection)
            if signature in seen:
                continue
            seen.add(signature)
            sampled.append((connection, score))
            if len(sampled) == count:
                return sampled
        raise RuntimeError(
            f"CEM produced only {len(sampled)}/{count} distinct routes after "
            f"{max_attempts} attempts; increase exploration or temperature"
        )

    def _combined_elite_counts(
        self,
        shape: torch.Size,
        slice_key: Tuple[int, int],
        elite_connections: Iterable[Sequence[tuple]],
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        counts = torch.zeros(shape, device=device, dtype=dtype)
        for connection in elite_connections:
            for _src, _dst, _port, meta in connection:
                meta = meta or {}
                if tuple(meta.get("slice", ())) != tuple(slice_key):
                    continue
                flat_row = int(meta["flat_row"])
                flat_column = int(meta["sample"])
                counts[flat_row, flat_column] += 1.0
        return counts

    def update(
        self,
        logits: RoutingLogits,
        elite_connections: Sequence[Sequence[tuple]],
    ):
        """Move row distributions toward the mean of elite legal matchings."""
        if not elite_connections:
            raise ValueError("CEM update requires at least one elite connection")
        mask_cache, z_cache = self.engine.get_cache(logits)
        updated = self._clone_logits(logits)

        for (stage, column), old_z in z_cache.items():
            legal = mask_cache[(stage, column)]
            counts = self._combined_elite_counts(
                old_z.shape,
                (stage, column),
                elite_connections,
                device=old_z.device,
                dtype=old_z.dtype,
            )
            expected = float(len(elite_connections))
            row_counts = counts.sum(dim=1)
            if not torch.allclose(
                row_counts,
                torch.full_like(row_counts, expected),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise ValueError(
                    f"incomplete CEM elite routes in slice {(stage, column)}: "
                    f"row counts range {float(row_counts.min())}.."
                    f"{float(row_counts.max())}, expected {expected}"
                )

            masked_old = old_z.masked_fill(~legal, -torch.inf)
            old_prob = torch.softmax(masked_old, dim=1).masked_fill(~legal, 0.0)
            target = counts / expected
            uniform = legal.to(old_z.dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            target = (
                (1.0 - self.exploration) * target
                + self.exploration * uniform
            )
            new_prob = (
                (1.0 - self.smoothing) * old_prob
                + self.smoothing * target
            )
            new_z = torch.log(new_prob.clamp_min(1.0e-12))
            new_z = new_z.masked_fill(~legal, self._ILLEGAL_LOGIT)

            n_sum = len(self.engine.comp_graph.slice_indice_map[(stage - 1, column)])
            n_dst = len(self.engine.comp_graph.slice_indice_map[(stage, column)])
            for port, name in enumerate(
                self._PORT_NAMES[: self.engine.comp_graph.port_num]
            ):
                block = new_z[:, port * n_dst : (port + 1) * n_dst]
                updated[(stage, column)][f"s{name}"] = block[:n_sum].clone()
                if column > 0:
                    updated[(stage, column)][f"c{name}"] = block[n_sum:].clone()

        return updated, self.stats(updated)

    def stats(self, logits: RoutingLogits) -> dict:
        """Return source-averaged entropy and concentration diagnostics."""
        mask_cache, z_cache = self.engine.get_cache(logits)
        normalized_entropies: List[float] = []
        max_probabilities: List[float] = []
        for key, z_slice in z_cache.items():
            legal = mask_cache[key]
            prob = torch.softmax(
                z_slice.masked_fill(~legal, -torch.inf), dim=1
            ).masked_fill(~legal, 0.0)
            entropy = -(prob * torch.log(prob.clamp_min(1.0e-12))).sum(dim=1)
            n_legal = legal.sum(dim=1)
            multi = n_legal > 1
            if bool(multi.any().item()):
                normalized = entropy[multi] / torch.log(
                    n_legal[multi].to(entropy.dtype)
                )
                normalized_entropies.extend(normalized.detach().cpu().tolist())
            max_probabilities.extend(prob.max(dim=1).values.detach().cpu().tolist())
        return {
            "normalized_entropy": (
                float(np.mean(normalized_entropies))
                if normalized_entropies
                else 0.0
            ),
            "mean_max_probability": (
                float(np.mean(max_probabilities))
                if max_probabilities
                else 1.0
            ),
            "sources": len(max_probabilities),
        }
