"""每 slot logits 的梯度 cell 求解器（hard 前向 + STE 反向）。

候选集 = trainer 类型表（index 0 = exact，与内/外环同一菜单），LUT 取自
library.json / library42_native.json（精确张量化，不用 diffam 的 MLP surrogate）。

MRED 分层估计器（关键设计）：MRED = mean(|e|/golden) 被极少数小乘积样本主导
（k12 floor 下 g<2^22 的 ~0.7% 样本贡献绝大部分质量；200k 均匀前缀与 16M 估计可差
一个量级）。故对固定 16M xorshift 流分层：
  S12 = {0<g<2^22}（~107k 个）→ 每步全量精确；
  S3  = {g≥2^22} → 固定前缀子样本 + 权重 |S3|/n_sub（确定性、低方差：r≤|e|/2^22）。
估计值 ≈ 16M 全量口径，且跨 repair 步确定性单调可比。终验仍走 verilator 16M。

loss = 面积项（Σ 所选 cell 面积 / Σ exact 锚点面积）+ λ·relu(MRED_est/budget − 1)，
λ 自适应；离散化贪心修复（摘 wae·2^col 最大 slot），floor 不可行时回退历史最优。
"""
import json
import os

import numpy as np
import torch

from . import sim as S

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASK31 = 0x7FFFFFFF


def load_libs():
    lib = json.load(open(os.path.join(REPO, "Appr_Comp/library.json")))["cells"]
    p42 = os.path.join(REPO, "Appr_Comp/library42_native.json")
    lib42 = json.load(open(p42)) if os.path.exists(p42) else {"cells": {}, "meta": {}}
    return lib, lib42


class SlotSpace:
    """把 exp 类型表 + 库 LUT 组装成每 slot 的候选栈。"""

    def __init__(self, exp, tree, device="cpu"):
        self.device = device
        lib, lib42 = load_libs()
        self.exact_area = {}
        for name, c in lib.items():
            if c.get("is_exact"):
                self.exact_area[{"32": 0, "22": 1}[c["type"]]] = float(c["area"])
        anchors = (lib42.get("meta") or {}).get("anchors") or {}
        if "CT42_BAL" in anchors:
            self.exact_area[4] = float(anchors["CT42_BAL"].get("area") or 0.0)

        tables = {0: exp.type_table_32, 1: exp.type_table_22,
                  4: exp.type_table_42 or []}
        self.stacks = {}   # t -> (tt [K,2^n,n_out] float32, area [K] float64)
        self.tables = tables
        for t, table in tables.items():
            if len(table) <= 1:
                continue
            tts, areas = [], []
            for k, entry in enumerate(table):
                if k == 0:
                    luts = S.exact_luts(t)
                    area = self.exact_area.get(t, 0.0)
                else:
                    cell = (lib42["cells"] if t == 4 else lib)[entry["name"]]
                    luts = S.approx_luts_from_lib(t, cell)
                    area = float(entry.get("area") or cell.get("area") or 0.0)
                    if not area:
                        area = self.exact_area.get(t, 0.0)  # 无表征 → 零收益中性
                outs = [luts["sum"], luts["carry"]] + ([luts["cout"]] if t == 4 else [])
                tts.append(np.stack(outs, axis=1))  # [2^n, n_out]
                areas.append(area)
            self.stacks[t] = (
                torch.from_numpy(np.stack(tts)).to(torch.float32).to(device),
                torch.tensor(areas, dtype=torch.float64, device=device),
            )

        # 合法 slot：树上 t∈{0,1,4} 且列在近似资格带、有候选表
        self.slots = []
        for node, kind, t, col, ref, aux in tree.plan:
            if kind != "cell" or t not in self.stacks:
                continue
            if not exp._is_approx_col_allowed(col):
                continue
            self.slots.append((node, t, col))

    def wae_of(self, t, k):
        return float(self.tables[t][k].get("wae", 0.0))

    def cell_luts_of(self, config):
        lib, lib42 = load_libs()
        out = {}
        for node, (t, k) in config.items():
            entry = self.tables[t][k]
            cell = (lib42["cells"] if t == 4 else lib)[entry["name"]]
            out[int(node)] = S.approx_luts_from_lib(t, cell)
        return out


class MredEstimator:
    """固定 16M 流上的分层 MRED 估计器（确定性）。"""

    def __init__(self, pool_a, pool_b, device, small_thresh=1 << 22,
                 s3_sub=262_144, s3_batch=32_768, screen=24_576, seed=7):
        golden = (pool_a.astype(np.int64) * pool_b.astype(np.int64)) & MASK31
        self.n_rel = int((golden != 0).sum())
        i12 = np.flatnonzero((golden > 0) & (golden < small_thresh))
        i3 = np.flatnonzero(golden >= small_thresh)
        self.a12, self.b12 = pool_a[i12], pool_b[i12]
        sub = i3[:s3_sub]                      # 固定前缀子样本（确定性 gate）
        self.a3s, self.b3s = pool_a[sub], pool_b[sub]
        self.w3 = len(i3) / len(sub)
        self.i3_all = i3
        self.pool_a, self.pool_b = pool_a, pool_b
        self.s3_batch = s3_batch
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.g12 = torch.from_numpy(golden[i12]).to(device)
        self.g3s = torch.from_numpy(golden[sub]).to(device)
        self.w3_train = len(i3)  # 训练批权重按批大小折算
        # 筛选子样本（仅用于贪心排序/lazy 比较,非可行性判定）：小乘积层里确定性取
        # screen 个,MRED 质量集中于此,排序代理足够；可行性判定始终走全量 gate。
        ns = min(screen, len(i12))
        ssub = np.linspace(0, len(i12) - 1, ns).astype(np.int64) if len(i12) else \
            np.array([], dtype=np.int64)
        self.a_sc, self.b_sc = self.a12[ssub], self.b12[ssub]
        self.g_sc = self.g12[torch.from_numpy(ssub).to(device)] if len(ssub) else \
            self.g12[:0]

    @staticmethod
    def _ratio_sum_exact(out, golden):
        e = S.wrap31_int((out & MASK31) - golden)
        return (e.abs().to(torch.float64) / golden.to(torch.float64)).sum()

    @staticmethod
    def _ratio_sum_diff(out, golden):
        gf = golden.to(torch.float64)
        e = torch.remainder(out, float(1 << 31)) - gf
        HALF, FULL = float(1 << 30), float(1 << 31)
        e = torch.where(e > HALF, e - FULL, e)
        e = torch.where(e < -HALF, e + FULL, e)
        return (e.abs() / gf).sum()

    def gate(self, tree, specs, bit_width, cell_luts):
        """离散配置的确定性 MRED 估计（≈16M 口径）。"""
        s = torch.zeros((), dtype=torch.float64, device=self.device)
        for (a, b, g, w) in ((self.a12, self.b12, self.g12, 1.0),
                             (self.a3s, self.b3s, self.g3s, self.w3)):
            pp = S.compute_pp_bits(specs, a, b, bit_width, self.device)
            out = tree.eval_exact(pp, cell_luts)
            s = s + w * self._ratio_sum_exact(out, g)
        return float(s.item() / self.n_rel)

    def train_batch(self):
        """S12 全量 + S3 随机切片（每步换）。返回 (a,b,golden,weight) 两段。"""
        j = self.rng.integers(0, len(self.i3_all) - self.s3_batch)
        idx = self.i3_all[j:j + self.s3_batch]
        a3, b3 = self.pool_a[idx], self.pool_b[idx]
        g3 = torch.from_numpy(
            (a3.astype(np.int64) * b3.astype(np.int64)) & MASK31).to(self.device)
        w3 = len(self.i3_all) / self.s3_batch
        return ((self.a12, self.b12, self.g12, 1.0), (a3, b3, g3, w3))


class GradientCellSolver:
    def __init__(self, exp, tree, pp_specs, budget, device="cpu",
                 pool_vectors=16_000_000, seed=12345, cache_dir=None, est=None):
        self.exp, self.tree, self.budget = exp, tree, float(budget)
        self.device = device
        self.space = SlotSpace(exp, tree, device)
        self.pp_specs = pp_specs
        if est is not None:
            # 复用外部估计器（池/分层与设备绑定,与结构无关——训练内跨 episode 缓存）
            self.est = est
            self.pool_a, self.pool_b = est.pool_a, est.pool_b
        else:
            a, b = S.xorshift_ab(pool_vectors, seed=seed, cache_dir=cache_dir)
            self.pool_a, self.pool_b = a, b
            self.est = MredEstimator(a, b, device)
        n_slots = len(self.space.slots)
        kmax = max((self.space.stacks[t][0].shape[0]
                    for _n, t, _c in self.space.slots), default=0)
        self.logits = torch.full((n_slots, max(kmax, 1)), 0.0, device=device)
        if n_slots:
            self.logits[:, 0] = 2.0    # exact 正偏置起步
        self.logits.requires_grad_(True)
        self.mask = torch.full_like(self.logits, float("-inf"))
        for i, (_n, t, _c) in enumerate(self.space.slots):
            self.mask[i, : self.space.stacks[t][0].shape[0]] = 0.0

    # -------------------------------------------------------- 选择权重
    def weights(self, tau=1.0):
        w_soft = torch.softmax((self.logits + self.mask) / tau, dim=-1)
        hard = torch.zeros_like(w_soft)
        hard.scatter_(1, w_soft.argmax(dim=-1, keepdim=True), 1.0)
        return hard + w_soft - w_soft.detach()

    def sel_dict(self, w):
        sel = {}
        for i, (node, t, _c) in enumerate(self.space.slots):
            tt, _a = self.space.stacks[t]
            sel[node] = (tt, w[i, : tt.shape[0]])
        return sel

    def area_term(self, w):
        tot = 0.0
        sel_area = torch.zeros((), dtype=torch.float64, device=self.device)
        for i, (_n, t, _c) in enumerate(self.space.slots):
            _tt, areas = self.space.stacks[t]
            sel_area = sel_area + (w[i, : areas.shape[0]].to(torch.float64) @ areas)
            tot += float(areas[0])
        return sel_area / max(tot, 1e-9)

    def area_saving(self, config):
        sv = 0.0
        for _n, (t, k) in config.items():
            _tt, areas = self.space.stacks[t]
            sv += float(areas[0] - areas[k])
        return sv

    # -------------------------------------------------------- 离散评估
    def hard_config(self):
        with torch.no_grad():
            arg = (self.logits + self.mask).argmax(dim=-1)
        return {node: (t, int(arg[i]))
                for i, (node, t, _c) in enumerate(self.space.slots)
                if int(arg[i]) != 0}

    def gate_mred(self, config):
        return self.est.gate(self.tree, self.pp_specs, self.exp.bit_width,
                             self.space.cell_luts_of(config))

    def gate_fast(self, config):
        """S12-only 快速口径（S3 贡献视作常数），用于贪心打分/排序。"""
        e = self.est
        pp = getattr(self, "_pp12", None)
        if pp is None:
            pp = self._pp12 = S.compute_pp_bits(
                self.pp_specs, e.a12, e.b12, self.exp.bit_width, self.device)
        out = self.tree.eval_exact(pp, self.space.cell_luts_of(config))
        return float(MredEstimator._ratio_sum_exact(out, e.g12).item()) / e.n_rel

    def gate_screen(self, config):
        """筛选子样本口径（~24k 小乘积）——仅贪心排序/lazy 比较用,不判可行性。"""
        e = self.est
        pp = getattr(self, "_pp_sc", None)
        if pp is None:
            pp = self._pp_sc = S.compute_pp_bits(
                self.pp_specs, e.a_sc, e.b_sc, self.exp.bit_width, self.device)
        out = self.tree.eval_exact(pp, self.space.cell_luts_of(config))
        return float(MredEstimator._ratio_sum_exact(out, e.g_sc).item()) / e.n_rel

    def measure_full(self, config, chunk=1_000_000):
        """全池精确测量（终验前的内部核对，≈verilator 16M 口径）。"""
        luts = self.space.cell_luts_of(config)
        n = len(self.pool_a)
        outs = []
        for i in range(0, n, chunk):
            pp = S.compute_pp_bits(self.pp_specs, self.pool_a[i:i + chunk],
                                   self.pool_b[i:i + chunk],
                                   self.exp.bit_width, self.device)
            outs.append(self.tree.eval_exact(pp, luts))
        return S.error_stats(torch.cat(outs), self.pool_a, self.pool_b)

    # -------------------------------------------------------- 主循环
    def solve(self, steps=300, lr=0.05, lam0=50.0, log_every=10, log=print,
              lam_step=100.0, init_config=None):
        """对偶上升版：λ ← max(5, λ + lam_step·(util−1))，温度 1→0.3 退火。
        跟踪历史最优（可行优先、并列取面积节省大者），修复输出与其对比取优。
        init_config 给定时从该离散配置温启动 logits（hybrid：贪心解→梯度精调）。"""
        if not self.space.slots:
            log("[solver] 无合法 slot")
            return {}, []
        if init_config is not None:
            with torch.no_grad():
                self.logits.zero_()
                for i, (node, t, _c) in enumerate(self.space.slots):
                    k = init_config.get(node, (t, 0))[1]
                    self.logits[i, k] = 3.0
        opt = torch.optim.Adam([self.logits], lr=lr)
        lam = lam0
        hist = []
        best = None   # (key, config)
        for step in range(steps):
            tau = max(0.3, 1.0 - 0.7 * step / max(steps - 1, 1))
            w = self.weights(tau)
            sel = self.sel_dict(w)
            rs = torch.zeros((), dtype=torch.float64, device=self.device)
            for a, b, g, wgt in self.est.train_batch():
                pp = S.compute_pp_bits(self.pp_specs, a, b, self.exp.bit_width,
                                       self.device)
                out = self.tree.eval_diff(pp, sel)
                rs = rs + wgt * MredEstimator._ratio_sum_diff(out, g)
            mred = rs / self.est.n_rel
            area = self.area_term(w)
            loss = area + lam * torch.relu(mred / self.budget - 1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % log_every == 0 or step == steps - 1:
                cfg = self.hard_config()
                gm = self.gate_mred(cfg)
                util = gm / self.budget
                key = (max(gm, self.budget), -self.area_saving(cfg))
                if best is None or key < best[0]:
                    best = (key, dict(cfg))
                lam = max(5.0, lam + lam_step * (util - 1.0))
                hist.append((step + 1, len(cfg), gm, float(area.detach())))
                if (step + 1) % (log_every * 5) == 0 or step == steps - 1:
                    log(f"[solver] step {step+1:4d} n_cells={len(cfg):3d} "
                        f"gate_mred={gm:.3e} util={util:6.1%} "
                        f"area_frac={float(area.detach()):.4f} lam={lam:.1f}")
        cfg = self.repair(self.hard_config(), log)
        gm_final = self.gate_mred(cfg)
        key_final = (max(gm_final, self.budget), -self.area_saving(cfg))
        if best is not None and best[0] < key_final:
            log(f"[solver] 回退历史最优 n_cells={len(best[1])} "
                f"(saving {-best[0][1]:.2f} vs {self.area_saving(cfg):.2f})")
            cfg = best[1]
        return cfg, hist

    # -------------------------------------------------------- ③ 贪心加法基线
    def greedy_add(self, log=print, rescore_tol=0.7, upgrade=True):
        """实测口径的 lazy greedy：从 floor 出发，按 面积节省/实测Δmred 性价比加 cell。
        打分用 gate_fast（S12-only，秒级），验收用完整 gate；lazy 堆——弹出堆顶先
        用当前配置重测其 Δ，仍居前才接受（捕捉 cell 间交互）。
        upgrade=False 跳过升级扫描（训练内鲁棒模式用：贴线换面积的增量会被跨布线
        修复摘除,白做,省 ~1/3 求解时间）。"""
        import heapq
        cfg = {}
        gm = self.gate_mred(cfg)
        base_fast = self.gate_fast(cfg)
        log(f"[greedy] floor mred={gm:.3e} util={gm/self.budget:6.1%}")
        heap = []
        for node, t, _col in self.space.slots:
            _tt, areas = self.space.stacks[t]
            for k in range(1, areas.shape[0]):
                saving = float(areas[0] - areas[k])
                if saving <= 0:
                    continue
                dm = self.gate_fast({node: (t, k)}) - base_fast
                ratio = saving / max(dm, 1e-12) if dm > 0 else float("inf")
                heapq.heappush(heap, (-ratio, dm, node, t, k, saving))
        n_eval = len(heap)
        used = set()
        while heap:
            neg_r, dm_old, node, t, k, saving = heapq.heappop(heap)
            if node in used:
                continue
            # 当前配置下重测该候选的真实 Δ（lazy 校验,完整 S12 口径）
            cur_fast = self.gate_fast(cfg)
            dm_new = self.gate_fast({**cfg, node: (t, k)}) - cur_fast
            ratio_new = saving / max(dm_new, 1e-12) if dm_new > 0 else float("inf")
            if heap and dm_new > 0 and ratio_new < -heap[0][0] * rescore_tol:
                heapq.heappush(heap, (-ratio_new, dm_new, node, t, k, saving))
                continue
            trial = {**cfg, node: (t, k)}
            gm_t = self.gate_mred(trial)
            n_eval += 2
            if gm_t <= self.budget:
                cfg = trial
                used.add(node)
                gm = gm_t
                log(f"[greedy] + node {node} {self.space.tables[t][k]['name']} "
                    f"(save {saving:.2f}) mred={gm:.3e} "
                    f"util={gm/self.budget:6.1%} n_cells={len(cfg)}")
        log(f"[greedy] 填充完 n_cells={len(cfg)} mred={gm:.3e} "
            f"util={gm/self.budget:6.1%} saving={self.area_saving(cfg):.2f} "
            f"(gate evals≈{n_eval})")
        if upgrade:
            cfg = self.greedy_upgrade(cfg, log=log)
        return cfg

    def greedy_upgrade(self, cfg, log=print, sweeps=2):
        """升级扫描：已定型的 slot 尝试换更省面积的 cell（fast 口径预筛 +
        完整 gate 确认），把剩余误差 slack 换成面积。"""
        gm = self.gate_mred(cfg)
        fast_off = gm - self.gate_fast(cfg)   # 完整 S12→full 口径校正（准，用于升级预筛）
        for sweep in range(sweeps):
            improved = False
            for node, t, _col in self.space.slots:
                cur_k = cfg.get(node, (t, 0))[1]
                _tt, areas = self.space.stacks[t]
                cur_sv = float(areas[0] - areas[cur_k])
                cands = sorted(
                    ((float(areas[0] - areas[k]), k)
                     for k in range(1, areas.shape[0])
                     if float(areas[0] - areas[k]) > cur_sv + 1e-9),
                    reverse=True)
                for sv, k in cands:
                    trial = {**cfg, node: (t, k)}
                    if self.gate_fast(trial) + fast_off > self.budget:
                        continue
                    gm_t = self.gate_mred(trial)
                    if gm_t <= self.budget:
                        cfg, gm = trial, gm_t
                        fast_off = gm - self.gate_fast(cfg)
                        improved = True
                        break
            log(f"[upgrade] sweep{sweep+1} n_cells={len(cfg)} mred={gm:.3e} "
                f"util={gm/self.budget:6.1%} saving={self.area_saving(cfg):.2f}")
            if not improved:
                break
        return cfg

    def repair(self, config, log=print):
        """离散化贪心修复：超预算 → 摘解析贡献 wae·2^col 最大的 slot。
        floor（全摘光）也超预算时返回历史途中最优由 solve() 兜底。"""
        col_of = {node: col for node, _t, col in self.space.slots}
        while True:
            gm = self.gate_mred(config)
            if gm <= self.budget or not config:
                log(f"[repair] 结束 mred={gm:.3e} util={gm/self.budget:6.1%} "
                    f"n_cells={len(config)}")
                return config
            worst = max(config,
                        key=lambda nd: self.space.wae_of(*config[nd])
                        * 2.0 ** col_of[nd])
            t, k = config.pop(worst)
            log(f"[repair] 超预算({gm:.3e}) → 摘 node {worst} "
                f"({self.space.tables[t][k]['name']})")
