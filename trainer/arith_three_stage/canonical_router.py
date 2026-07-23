"""Deterministic per-slice Kuhn matching used by Stage 1 and Stage 2."""
from __future__ import annotations

from typing import List

import torch


class CanonicalRoutingError(RuntimeError):
    pass


class CanonicalRouter:
    """Build a stable legal routing without consulting the PPO policy."""

    @staticmethod
    def _slice_mask(graph, s: int, c: int) -> torch.Tensor:
        sum_mask = graph.get_slice_sum_mask(s, c).bool()
        blocks = []
        carry_mask = graph.get_slice_carry_mask(s, c).bool() if c > 0 else None
        for port in range(graph.port_num):
            block = sum_mask[port, :, :]
            if carry_mask is not None:
                block = torch.cat((block, carry_mask[port, :, :]), dim=0)
            blocks.append(block)
        return torch.cat(blocks, dim=1).cpu()

    @staticmethod
    def _kuhn(mask: torch.Tensor) -> List[int]:
        n_src, n_slot = mask.shape
        allowed = [torch.where(mask[i])[0].tolist() for i in range(n_src)]
        match_slot = [-1] * n_slot

        def dfs(src: int, seen: List[bool]) -> bool:
            for slot in allowed[src]:
                if seen[slot]:
                    continue
                seen[slot] = True
                if match_slot[slot] < 0 or dfs(match_slot[slot], seen):
                    match_slot[slot] = src
                    return True
            return False

        for src in range(n_src):
            if not dfs(src, [False] * n_slot):
                raise CanonicalRoutingError(
                    f"no complete matching: source {src}/{n_src}, slots={n_slot}"
                )
        source_to_slot = [-1] * n_src
        for slot, src in enumerate(match_slot):
            if src >= 0:
                source_to_slot[src] = slot
        if any(slot < 0 for slot in source_to_slot):
            raise CanonicalRoutingError("Kuhn matching left an unmatched source")
        return source_to_slot

    def route(self, graph) -> list:
        connections = []
        for s in range(graph.stage_num + 1):
            for c in range(graph.col_num):
                dst_indices = list(graph.slice_indice_map[(s, c)])
                sum_sources = list(graph.slice_indice_map[(s - 1, c)])
                carry_sources = list(graph.get_slice_carry_sources(s, c)) if c > 0 else []
                sources = [(idx, "sum") for idx in sum_sources] + [
                    (idx, output) for idx, output in carry_sources
                ]
                if not sources:
                    continue
                if not dst_indices:
                    raise CanonicalRoutingError(f"slice {(s, c)} has sources but no destinations")
                mask = self._slice_mask(graph, s, c)
                if mask.shape[0] != len(sources):
                    raise CanonicalRoutingError(
                        f"slice {(s, c)} row mismatch: mask={mask.shape[0]} sources={len(sources)}"
                    )
                matched = self._kuhn(mask)
                for local_src, ((src_idx, src_output), slot) in enumerate(
                    zip(sources, matched)
                ):
                    local_dst = int(slot) % len(dst_indices)
                    port = int(slot) // len(dst_indices)
                    dst_idx = int(dst_indices[local_dst])
                    connections.append(
                        (
                            int(src_idx),
                            dst_idx,
                            port,
                            {
                                "log_prob": 0.0,
                                "local_src_idx": local_src,
                                "local_dst_idx": local_dst,
                                "sample": int(slot),
                                "slice": (s, c),
                                "src_output": src_output,
                                "canonical": True,
                            },
                        )
                    )
        return connections
