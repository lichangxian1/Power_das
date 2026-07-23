"""Per-child contextual Thompson bandit with delayed survival updates."""
from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Deque, Dict, Iterable, List, Tuple


class ContextualThompsonBandit:
    def __init__(self, arms: Iterable[str], window: int = 128, explore: float = 0.03):
        self.arms = list(arms)
        self.window = int(window)
        self.explore = float(explore)
        self._history: Dict[Tuple[str, str], Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    def choose(self, context: str, legal_arms: Iterable[str], rng: random.Random) -> str:
        legal = [a for a in self.arms if a in set(legal_arms)]
        if not legal:
            raise ValueError(f"no legal Thompson arm for context={context}")
        if rng.random() < self.explore:
            return rng.choice(legal)
        scored = []
        for arm in legal:
            hist = self._history[(str(context), arm)]
            wins, n = sum(hist), len(hist)
            scored.append((rng.betavariate(1 + wins, 1 + n - wins), arm))
        return max(scored)[1]

    def update(self, context: str, arm: str, success: bool) -> None:
        if arm not in self.arms:
            raise ValueError(f"unknown Thompson arm: {arm}")
        self._history[(str(context), arm)].append(1 if success else 0)

    def stats(self, context: str, arm: str) -> Tuple[int, int]:
        hist = self._history[(str(context), arm)]
        return sum(hist), len(hist)

    def state_dict(self) -> dict:
        return {
            "arms": self.arms,
            "window": self.window,
            "explore": self.explore,
            "history": {
                f"{ctx}\t{arm}": list(hist)
                for (ctx, arm), hist in self._history.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if list(state.get("arms") or []) != self.arms:
            raise ValueError("bandit arm mismatch while restoring checkpoint")
        self._history.clear()
        for key, values in (state.get("history") or {}).items():
            ctx, arm = key.split("\t", 1)
            self._history[(ctx, arm)].extend(int(v) for v in values[-self.window :])
