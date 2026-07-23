"""Stable candidate representation and stage-specific hashes."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
from typing import Any, Dict, List, Optional


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class Candidate:
    k: int
    ct22: List[int]
    ct32: List[int]
    ct42: List[int]
    cells: List[List[int]] = field(default_factory=list)
    routing: Optional[List[Any]] = None
    candidate_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    stage: int = 1
    area: Optional[float] = None
    power: Optional[float] = None
    delay: Optional[float] = None
    mred: Optional[float] = None
    valid: bool = True
    failure_reason: Optional[str] = None
    rank: Optional[int] = None
    crowding_distance: float = 0.0
    operator: Optional[str] = None
    operator_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.k = int(self.k)
        self.ct22 = [int(x) for x in self.ct22]
        self.ct32 = [int(x) for x in self.ct32]
        self.ct42 = [int(x) for x in self.ct42]
        self.cells = sorted(
            [[int(x) for x in e] for e in self.cells], key=lambda e: tuple(e)
        )
        if not self.candidate_id:
            self.candidate_id = self.cell_hash[:16]

    @property
    def structure_hash(self) -> str:
        return _stable_hash({
            "k": self.k,
            "ct22": self.ct22,
            "ct32": self.ct32,
            "ct42": self.ct42,
        })

    @property
    def cell_hash(self) -> str:
        return _stable_hash({"structure": self.structure_hash, "cells": self.cells})

    @property
    def routing_hash(self) -> str:
        if self.routing is None:
            route = "canonical"
        else:
            route = []
            for edge in self.routing:
                src, dst, port, meta = edge
                route.append([
                    int(src), int(dst), int(port),
                    str((meta or {}).get("src_output", "sum")),
                ])
        return _stable_hash({"cell": self.cell_hash, "routing": route})

    @property
    def evaluated(self) -> bool:
        vals = (self.area, self.power, self.delay, self.mred)
        return all(v is not None and math.isfinite(float(v)) for v in vals)

    def clone(self, *, stage: Optional[int] = None) -> "Candidate":
        child = copy.deepcopy(self)
        child.parent_ids = [self.candidate_id]
        child.candidate_id = ""
        child.stage = self.stage if stage is None else int(stage)
        child.area = child.power = child.delay = child.mred = None
        child.valid = True
        child.failure_reason = None
        child.rank = None
        child.crowding_distance = 0.0
        child.routing = None
        child.metadata = {}
        return child

    def refresh_id(self) -> None:
        base = self.routing_hash if self.routing is not None else self.cell_hash
        self.candidate_id = base[:16]

    def set_result(self, result: Dict[str, Any]) -> None:
        self.area = float(result["area"])
        self.power = float(result["power"])
        self.delay = float(result["delay"])
        self.mred = float(result["mred"])
        self.valid = bool(result.get("valid", True))
        self.failure_reason = result.get("failure_reason")
        self.metadata.update(copy.deepcopy(result.get("metadata") or {}))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        return cls(**copy.deepcopy(data))
