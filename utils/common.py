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


class ParetoArchive:
    """v5 多目标档案（PARETO_ARITH_PLAN.md）：MRED 对数分箱 + 箱内 (area,power) 支配准入。

    选择层零人工系数：谁存活只由支配关系决定。分箱 = mred 轴的 ε-支配网格（容量控制 +
    全轴覆盖保证）；eps_power 是 DC 复跑噪声地板（测量分辨率，非偏好系数）——功耗差
    落在分辨率内视为同值、由面积裁决。容量超限按 NSGA-II 拥挤度淘汰（两端极值保留）。
    条目 = {"mred","area","power","payload"}；payload 与 found_best_info 同构（可导出）。"""

    def __init__(self, mred_lo=1e-7, mred_hi=2e-1, bin_ratio=2.0, bin_cap=6,
                 eps_power=0.01, binless=False):
        import math
        if not (0 < mred_lo < mred_hi and bin_ratio > 1.0 and bin_cap >= 1):
            raise ValueError(
                f"ParetoArchive 参数非法: lo={mred_lo} hi={mred_hi} "
                f"ratio={bin_ratio} cap={bin_cap}（cap 最小 1）")
        self.lo, self.hi = float(mred_lo), float(mred_hi)
        self.ratio, self.cap = float(bin_ratio), int(bin_cap)
        self.eps = float(eps_power)
        # binless 消融（07-16 用户裁定实验）：单箱 + 支配升维到 (mred,area,power)
        # 三目标 + 3D 拥挤度淘汰。cap 语义变为全档案总容量（对照组 = n_bins×cap）。
        self.binless = bool(binless)
        if self.binless:
            self.n_bins = 1
        else:
            self.n_bins = max(1, int(math.ceil(
                math.log(self.hi / self.lo) / math.log(self.ratio) - 1e-9)))
        self.bins = {i: [] for i in range(self.n_bins)}

    def bin_of(self, mred):
        """mred → 箱号；超上限返回 None（档案范围外），低于下限归 0 箱。"""
        import math
        if mred is None or mred > self.hi:
            return None
        if self.binless or mred <= self.lo:
            return 0
        b = int(math.log(mred / self.lo) / math.log(self.ratio))
        return min(b, self.n_bins - 1)

    def bin_edges(self, b):
        if self.binless:
            return (self.lo, self.hi)
        return (self.lo * self.ratio ** b,
                min(self.hi, self.lo * self.ratio ** (b + 1)))

    def _dominates(self, a, c):
        """a 支配 c？功耗轴按 eps 相对量化（分辨率内=同值）。
        binless：mred 作为第三目标进支配（分箱模式下 mred 由箱离散化承担，
        不进箱内比较；无箱后必须显式比，否则低误差端会被小面积设计灭绝）。"""
        tol = self.eps * min(a["power"], c["power"])
        if not (a["area"] <= c["area"] and a["power"] <= c["power"] + tol):
            return False
        if self.binless:
            if a["mred"] > c["mred"]:
                return False
            return (a["mred"] < c["mred"] or a["area"] < c["area"]
                    or a["power"] < c["power"] - tol)
        return a["area"] < c["area"] or a["power"] < c["power"] - tol

    def add(self, mred, area, power, payload):
        """支配准入。返回 (admitted, bin)；bin=None 表示 mred 超范围/指标非法。
        area<=0 或 power<0 视为异常测量拒收（负 power 会使 ε 容差为负，
        重复点无限入档——07-13 双评审发现）。NaN/Inf 一律拒收（r2 审查 #2：
        NaN 会在 bin_of 的 int(log) 处直接抛异常打崩训练，必须挡在门外）。"""
        import math
        if area is None or power is None or mred is None:
            return False, None
        if not all(math.isfinite(float(x)) for x in (mred, area, power)):
            return False, None
        if mred < 0 or area <= 0 or power < 0:
            return False, None
        b = self.bin_of(mred)
        if b is None:
            return False, None
        cand = {"mred": float(mred), "area": float(area),
                "power": float(power), "payload": payload}
        for e in self.bins[b]:
            if self._dominates(e, cand):
                return False, b
            if (e["area"] == cand["area"]
                    and abs(e["power"] - cand["power"]) <= self.eps * cand["power"]
                    and (not self.binless or e["mred"] <= cand["mred"])):
                return False, b   # 分辨率内重复点（binless：mred 更优不算重复）
        kept = [e for e in self.bins[b] if not self._dominates(cand, e)]
        kept.append(cand)
        while len(kept) > self.cap:
            kept = (self._evict_most_crowded_3d(kept) if self.binless
                    else self._evict_most_crowded(kept))
        self.bins[b] = kept
        return any(e is cand for e in kept), b

    @staticmethod
    def _evict_most_crowded(ents):
        """NSGA-II 拥挤度淘汰：按 area 排序，两端极值免死，删中间最挤的。
        cap=1 时（仅剩 2 个极值仍超容量）保面积最小者，避免死循环。"""
        ents = sorted(ents, key=lambda e: (e["area"], e["power"]))
        if len(ents) <= 2:
            return ents[:1]
        ar = max(ents[-1]["area"] - ents[0]["area"], 1e-12)
        pw = [e["power"] for e in ents]
        pr = max(max(pw) - min(pw), 1e-12)
        crowd = {
            i: (ents[i + 1]["area"] - ents[i - 1]["area"]) / ar
               + abs(ents[i - 1]["power"] - ents[i + 1]["power"]) / pr
            for i in range(1, len(ents) - 1)
        }
        j = min(crowd, key=lambda i: (crowd[i], i))
        return ents[:j] + ents[j + 1:]

    def _evict_most_crowded_3d(self, ents):
        """binless 淘汰：NSGA-II 拥挤度在 (log10 mred, area, power) 三维上计算。
        mred 取对数（跨 6 个数量级，线性拥挤度会把整个低误差端挤成一个点——
        binless 消融的成败就在这一处归一化）。每个目标的两端极值拥挤度 = inf
        免死，删总拥挤度最小者；全 inf（≤6 点）退化为保面积序前缀。"""
        import math
        if len(ents) <= 2:
            return ents[:1] if len(ents) > self.cap else ents
        keys = {
            "m": lambda e: math.log10(max(e["mred"], self.lo * 1e-3)),
            "a": lambda e: e["area"],
            "p": lambda e: e["power"],
        }
        crowd = {id(e): 0.0 for e in ents}
        for kf in keys.values():
            srt = sorted(ents, key=kf)
            rng = max(kf(srt[-1]) - kf(srt[0]), 1e-12)
            crowd[id(srt[0])] = crowd[id(srt[-1])] = float("inf")
            for i in range(1, len(srt) - 1):
                crowd[id(srt[i])] += (kf(srt[i + 1]) - kf(srt[i - 1])) / rng
        finite = [e for e in ents if crowd[id(e)] != float("inf")]
        if not finite:
            srt = sorted(ents, key=lambda e: (e["area"], e["power"]))
            return srt[:self.cap]
        victim = min(finite, key=lambda e: crowd[id(e)])
        return [e for e in ents if e is not victim]

    def nearest_nonempty(self, b):
        """b 空则向外找最近非空箱（同距先取更紧/更低 mred 的一侧）。"""
        if self.bins.get(b):
            return b
        for d in range(1, self.n_bins):
            for bb in (b - d, b + d):
                if 0 <= bb < self.n_bins and self.bins[bb]:
                    return bb
        return None

    def sample_parent(self, b, rng):
        """从箱 b（空则最近非空箱）均匀取一个条目；档案全空返回 None。"""
        bb = self.nearest_nonempty(b)
        if bb is None:
            return None
        return rng.choice(self.bins[bb])

    def global_min_area(self):
        ents = [e for es in self.bins.values() for e in es]
        return min(ents, key=lambda e: e["area"]) if ents else None

    def snapshot(self):
        """front.json 用的轻量快照（不含 payload 大字段）。"""
        out = []
        for b in range(self.n_bins):
            lo, hi = self.bin_edges(b)
            for e in sorted(self.bins[b], key=lambda x: x["area"]):
                ct = (e["payload"] or {}).get("ct") or {}
                out.append({
                    "bin": b, "edge_lo": lo, "edge_hi": hi,
                    "mred": e["mred"], "area": e["area"], "power": e["power"],
                    "k": int(ct.get("k", -1)),
                    "n_cells": len(ct.get("cells") or []),
                })
        return out

    def n_nonempty(self):
        return sum(1 for es in self.bins.values() if es)

    def __len__(self):
        return sum(len(es) for es in self.bins.values())


class OpBandit:
    """V6-R2：按 (箱, 臂) 条件化的 Thompson 采样骰子（Beta-Bernoulli + 滑动窗口）。

    观测 = 该 (箱, 臂) 的一个 episode 是否产生 ≥1 入档（二值；admit 数不进观测，
    避免种子期大丰收把先验拉爆）。窗口 W 丢弃陈旧观测——箱会饱和，产出非平稳。
    保底探索：以 floor×len(可用臂) 概率均匀抽臂，保证每臂选中概率 ≥floor。
    r2 依据：keep 臂 33% 预算 ↔ 20% 产出；zero 臂尾箱冠军/低箱必亏——臂的好坏
    强依赖箱位置，静态概率必然全局次优（V6_SEARCH_PLAN.md §R2）。"""

    def __init__(self, arms, window=12, floor=0.05):
        if not arms or window < 1 or not (0 <= floor * len(arms) < 1):
            raise ValueError(f"OpBandit 参数非法: arms={arms} window={window} floor={floor}")
        self.arms = list(arms)
        self.window = int(window)
        self.floor = float(floor)
        self.hist = {}   # (bin, arm) -> [0/1,...] 最近 window 个

    def choose(self, b, available, rng):
        """rng = np.random.Generator。available 为本轮可用臂子集（如箱内无同 k
        第二亲本时 crossover 不可用）。返回臂名；无可用臂返回 None。"""
        avail = [a for a in self.arms if a in available]
        if not avail:
            return None
        if rng.random() < self.floor * len(avail):
            return avail[int(rng.integers(len(avail)))]
        best, best_v = None, -1.0
        for a in avail:
            h = self.hist.get((int(b), a), [])
            wins = sum(h)
            v = float(rng.beta(1 + wins, 1 + len(h) - wins))
            if v > best_v:
                best, best_v = a, v
        return best

    def update(self, b, arm, success):
        h = self.hist.setdefault((int(b), str(arm)), [])
        h.append(1 if success else 0)
        del h[:-self.window]

    def stats_of(self, b, arm):
        h = self.hist.get((int(b), str(arm)), [])
        return sum(h), len(h)

    def to_json(self):
        return {f"{b}|{a}": h for (b, a), h in self.hist.items()}

    def load(self, d):
        for key, h in (d or {}).items():
            b, a = key.split("|", 1)
            self.hist[(int(b), a)] = [int(x) for x in h][-self.window:]


def lse_gamma(x: torch.Tensor, gamma: float, dim: int = -1):
    return gamma * torch.logsumexp(x / gamma, dim=dim)
