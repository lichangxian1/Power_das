import heapq
import itertools
import copy
import numpy as np
import torch


class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, key, item):
        heapq.heappush(self.heap, (key, item))

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)
        raise IndexError("pop from an empty priority queue")

    def peek(self):
        if self.heap:
            return self.heap[0]
        return None

    def __len__(self):
        return len(self.heap)


class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, key, item):
        for i in range(len(self.heap)):
            if self.heap[i][0] == -key:
                return
        heapq.heappush(self.heap, (-key, item))

    def pop(self):
        if not self.heap:
            raise IndexError("pop from an empty heap")
        key, item = heapq.heappop(self.heap)
        return -key, item

    def peek(self):
        if not self.heap:
            return None
        key, item = self.heap[0]
        return -key, item

    def least(self):
        if not self.heap:
            return None
        l = copy.deepcopy(self.heap)
        l_sorted = sorted(l, key=lambda x: -x[0])
        return -l_sorted[0][0], l_sorted[0][1]

    def __len__(self):
        return len(self.heap)


def one_hot_encoder(index: int, min_index: int, max_index: int) -> np.ndarray:
    assert min_index <= index <= max_index
    n = max_index - min_index + 1
    code01 = np.zeros((n), dtype=int)
    code01[index - min_index] = 1
    return code01


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.item()
    else:
        return obj


class BoundedParetoPool:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._pool = []
        self._counter = 0  # tiebreaker: prevents heapq from comparing state objects

    def add(self, objective: float, state):
        entry = (-objective, self._counter, state)
        self._counter += 1
        if len(self._pool) < self.max_size:
            heapq.heappush(self._pool, entry)
        else:
            if -objective > self._pool[0][0]:
                for obj, _, _ in self._pool:
                    if obj == -objective:
                        return
                heapq.heapreplace(self._pool, entry)

    def get_pool(self) -> list:
        return [(-obj, state) for obj, _, state in self._pool]

    def get_best(self):
        if self._pool:
            return min(self._pool, key=lambda x: -x[0])
        return None

    def get_worst(self):
        if self._pool:
            obj, _, state = self._pool[0]
            return (-obj, state)
        return None

    def __len__(self) -> int:
        return len(self._pool)

    def is_full(self) -> bool:
        return len(self._pool) >= self.max_size

    def is_empty(self) -> bool:
        return not self._pool


def lse_gamma(x: torch.Tensor, gamma: float, dim: int = -1):
    return gamma * torch.logsumexp(x / gamma, dim=dim)
