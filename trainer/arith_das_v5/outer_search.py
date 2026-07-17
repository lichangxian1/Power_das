"""外环 cell 配置搜索：add/remove/swap/zero/crossover 变异算子、
bandit 骰子、预算闭式过滤、errgate/TT-oracle 实测预筛、greedy 求解器。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
import random
import logging


import numpy as np

from utils import (
    get_initial_partial_product,
    CompressorTree,
    Mul,
)


class OuterSearchMixin:
    """外环 cell 配置搜索：变异/杂交/bandit/预算过滤/预筛门/greedy 求解。"""

    # ===== 外环 cell 搜索（Appr_Comp/OUTER_CELL_SEARCH.md）=====
    # state["cells"] = [[s, c, t, idx, k], ...]（slot 坐标 = assignment 顶点身份，跨结构
    # 变异稳定；k = 类型表索引 ≥1）。变异 = 解析提议 + 闭式可行性过滤 + resample-K；
    # 内环所有样本共用同一配置，PPO 只学布线。

    def _cells_error_totals(self, cells):
        items = [(int(e[1]), int(e[2]), int(e[4])) for e in (cells or [])]
        return self._error_totals_from_cols(items)

    def _outer_med_slack(self):
        """cell 部分允许的 Σwae·2^col 总额度（LSB）；None = 该口径无闭式约束。
        MRED 模式假设「cell 对 MRED 的推动 ≈ 对 MED 的相对推动」（一阶近似，
        outer_med_slack_scale 留作保守化旋钮）；trunc=0 时无 MRED floor 可用 → 放行，
        交给 verilator 闸门（诚实声明的空洞）。"""
        if self.med_budget is not None:
            med0, _b, _w = self._error_totals_from_cols([])
            return max(0.0, float(self.med_budget) - med0)
        if (getattr(self, "error_metric", "med") == "mred"
                and getattr(self, "mred_budget", None)
                and self.trunc_cols > 0 and self._trunc_model_mred):
            ratio = float(self.mred_budget) / float(self._trunc_model_mred) - 1.0
            return max(0.0, ratio * float(self._trunc_med) * self.outer_med_slack_scale)
        return None

    def _cells_budget_ok(self, cells):
        med, _bias, wce = self._cells_error_totals(cells)
        if self.med_budget is not None and med > float(self.med_budget):
            return False
        if self.wce_budget is not None and wce > float(self.wce_budget):
            return False
        if self.med_budget is None:
            slack = self._outer_med_slack()
            if slack is not None:
                med0, _b0, _w0 = self._error_totals_from_cols([])
                if med - med0 > slack:
                    return False
        return True

    def _enumerate_type_slots(self, assignment):
        """合法 cell slot 列表 [(s,c,t,idx)]：列在近似窗口内、该型类型表非平凡。"""
        if not self.use_approx_types or not self.type_table_32:
            return []
        slots = []
        for s in range(len(assignment)):
            for c in range(len(assignment[s])):
                if not self._is_approx_col_allowed(c):
                    continue
                for v in assignment[s][c]:
                    t = int(v[2])
                    if t not in (0, 1, 4) or (t == 4 and not self.use_ct42):
                        continue
                    _h, table = self._type_head_and_table(t)
                    if len(table) <= 1:
                        continue
                    slots.append((int(v[0]), int(v[1]), t, int(v[3])))
        return slots

    def _cells_prune_stale(self, cells, assignment):
        """结构变异后丢掉 slot 已消失的 cell（只减不增 → 误差只降，无需复检预算）。"""
        valid = set(self._enumerate_type_slots(assignment))
        kept = [e for e in (cells or [])
                if (int(e[0]), int(e[1]), int(e[2]), int(e[3])) in valid]
        n_drop = len(cells or []) - len(kept)
        if n_drop:
            logging.info("[outer] 结构变异后 %d 个 cell 的 slot 失效被丢弃", n_drop)
        return kept

    def _propose_cell_add(self, cells, slots, rng):
        """解析提议加一个 cell；无可行候选返回 None。
        符号偏好压残差偏置（正负抵消的硬逻辑）；softmax 打分 =
        outer_w_area·省面积占比 − outer_w_err·误差代价/slack（保留随机性，非 argmax）。"""
        occupied = {tuple(int(x) for x in e[:4]) for e in (cells or [])}
        free = [sl for sl in slots if sl not in occupied]
        if not free:
            return None
        med0, bias0, wce0 = self._cells_error_totals(cells)
        med_base, _b, _w = self._error_totals_from_cols([])
        slack = self._outer_med_slack()
        remain = None if slack is None else max(0.0, slack - (med0 - med_base))
        want = "P" if bias0 < -0.5 else ("N" if bias0 > 0.5 else None)

        def _collect(want_sign):
            cands, scores = [], []
            for sl in free:
                _s, c, t, _i = sl
                _h, table = self._type_head_and_table(t)
                # M2：省面积用在环境锚点（standalone 锚点虚高 ~4× 会排错序）
                ex_area = self._EXACT_AREA_INCTX.get(t) or table[0].get("area")
                w = float(1 << c)
                for k in range(1, len(table)):
                    entry = table[k]
                    cost = entry["wae"] * w
                    if remain is not None and cost > remain:
                        continue
                    if (self.wce_budget is not None
                            and wce0 + entry.get("maxe", 0.0) * w > float(self.wce_budget)):
                        continue
                    if want_sign and entry.get("group") not in (want_sign, "Z"):
                        continue
                    area_frac = 0.0
                    # is not None：ZERO cell 的 area=0.0 是合法表征（省下整个 exact cell）
                    if ex_area and entry.get("area") is not None:
                        area_frac = (float(ex_area) - float(entry["area"])) / float(ex_area)
                    denom = remain if (remain is not None and remain > 0) \
                        else float(1 << max(self.trunc_cols, 1))
                    score = (self.outer_w_area * area_frac
                             - self.outer_w_err * cost / max(denom, 1e-9))
                    cands.append((sl, k))
                    scores.append(score)
            return cands, scores

        cands, scores = _collect(want)
        if not cands and want is not None:
            cands, scores = _collect(None)
        if not cands:
            return None
        z = np.asarray(scores, dtype=np.float64)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        j = int(rng.choice(len(cands), p=p))
        sl, k = cands[j]
        return [sl[0], sl[1], sl[2], sl[3], int(k)]

    def _propose_cell_remove(self, cells, rng):
        if not cells:
            return None
        j = int(rng.integers(len(cells)))
        return cells[:j] + cells[j + 1:]

    def _propose_cell_swap(self, cells, slots, rng):
        if not cells:
            return None
        j = int(rng.integers(len(cells)))
        rest = cells[:j] + cells[j + 1:]
        slot = tuple(int(x) for x in cells[j][:4])
        add = self._propose_cell_add(rest, [slot], rng)
        return None if add is None else rest + [add]

    def _op_resample_k(self, slots, rng):
        """大步：K~cardinality choices 均匀先验（mask 到 ≤ 空闲 slot 数），清空后串行
        贪心加 K 个——每步重算残差偏置/slack 再提议下一个（正负抵消内建）；
        无可行候选提前停（K_actual < K）。"""
        choices = [k for k in self.approx_cardinality_choices if k <= len(slots)]
        if not choices:
            return []
        K = int(choices[int(rng.integers(len(choices)))])
        cells = []
        for _ in range(K):
            add = self._propose_cell_add(cells, slots, rng)
            if add is None:
                break
            cells.append(add)
        if len(cells) < K:
            logging.info("[outer] resample-K 提前停：目标 K=%d 实际 %d（可行性耗尽）",
                         K, len(cells))
        return cells

    def _zero_entry_of(self, t):
        """type t 的恒零输出 cell（=槽位级截断）表索引；无则 None。
        检测：菜单 const_zero 标志（add_zero_cells_unified.py 注入）或名字 *_zero 兜底。
        ⚠ 不能用 group=='Z'——N/Z/P 是偏置符号分组（Z=零偏置的功能 cell），恒零 cell
        因丢值恒为负偏置、按库约定挂 N 组（07-13 双评审发现 #1）。"""
        _h, table = self._type_head_and_table(t)
        zs = [(k, e) for k, e in enumerate(table)
              if k > 0 and (e.get("const_zero")
                            or str(e.get("name", "")).endswith("_zero"))]
        if not zs:
            return None
        return int(min(zs, key=lambda kv: float(kv[1].get("wae", 0.0)))[0])

    def _op_zero_col(self, cells, slots, rng):
        """M2 批量算子 zero-col：把最低的未清列（近似窗口内）整列填 ZERO
        = 分数截断一步，k 与 k+2 之间的连续插值（budget_sweep 证明 ZERO 是
        离线密集包主力，每包 3~23 个）。返回新 cells；无可操作列返回 None。
        注意：调用方跳过闭式预算过滤——解析模型对边界列 ZERO 失真
        （实测 bias 是解析 3.7×），可行性交给 TT oracle/errgate/v5 档案准入。"""
        occupied = {tuple(int(x) for x in e[:4]): int(e[4]) for e in (cells or [])}
        bycol = {}
        for sl in slots:
            bycol.setdefault(int(sl[1]), []).append(sl)
        for c in sorted(bycol):
            todo = []
            for sl in bycol[c]:
                kz = self._zero_entry_of(int(sl[2]))
                if kz is None:
                    continue
                if occupied.get(tuple(int(x) for x in sl)) != kz:
                    todo.append((sl, kz))
            if not todo:
                continue   # 该列已清满（或无 Z 型），看更高一列
            drop = {tuple(int(x) for x in sl) for sl, _ in todo}
            new = [list(e) for e in (cells or [])
                   if tuple(int(x) for x in e[:4]) not in drop]
            new += [[int(sl[0]), int(sl[1]), int(sl[2]), int(sl[3]), kz]
                    for sl, kz in todo]
            logging.info("[outer] zero-col c=%d：+%d ZERO（列清空），n_cells %d→%d",
                         c, len(todo), len(cells or []), len(new))
            return new
        return None

    def _op_unzero_col(self, cells, rng):
        """M2 反向算子 unzero-col：撤掉最低的成组 ZERO 列（误差只降，免检）。"""
        zero_cols = {}
        for e in (cells or []):
            kz = self._zero_entry_of(int(e[2]))
            if kz is not None and int(e[4]) == kz:
                zero_cols.setdefault(int(e[1]), []).append(
                    tuple(int(x) for x in e[:4]))
        if not zero_cols:
            return None
        c = min(zero_cols)
        drop = set(zero_cols[c])
        new = [list(e) for e in cells
               if tuple(int(x) for x in e[:4]) not in drop]
        logging.info("[outer] unzero-col c=%d：-%d ZERO，n_cells %d→%d",
                     c, len(drop), len(cells), len(new))
        return new

    def _is_zero_cell(self, e):
        """cells 条目 e=[s,c,t,idx,k] 是否常数零 cell（按类型表 const_zero 旗标）。"""
        _h, table = self._type_head_and_table(int(e[2]))
        return bool(table[int(e[4])].get("const_zero"))

    def _v5_crossover_candidates(self):
        """V6-R3 第二亲本池，三级放宽（07-15：同箱同 k 太苛——seg_lo 170 集仅凑齐
        8 次，臂长期不可用）：①同箱同 k（坐标天然对齐）②同箱近 k（|Δk|≤max_dk，
        杂交时 B 的列坐标按截断边界重映射 c+Δk = mini-R4 提升）③相邻箱（±bin_span）
        近 k。取最先非空层级；结构永远取 A，子代仍以当前箱为目标。
        返回 [(entry, kB)]。非 v5 / 档案未建返回空（臂自动不可用）。"""
        if not getattr(self, "pareto_v5", False):
            return []
        arch = getattr(self, "_v5_archive", None)
        if arch is None or self.state is None:
            return []
        kA = int(self.state.get("k", self.trunc_cols))
        max_dk = int(getattr(self, "outer_xover_max_dk", 2))
        span = int(getattr(self, "outer_xover_bin_span", 1))
        b0 = getattr(self, "_v5_bin", 0)

        def pool(bs, dk_lim):
            out = []
            for b in bs:
                for e in arch.bins.get(b) or []:
                    ct = (e.get("payload") or {}).get("ct") or {}
                    kB = int(ct.get("k", -1))
                    if kB >= 0 and abs(kB - kA) <= dk_lim:
                        out.append((e, kB))
            return out

        near = [b for b in range(max(0, b0 - span), min(arch.n_bins, b0 + span + 1))
                if b != b0]
        for tier in (pool([b0], 0), pool([b0], max_dk), pool(near, max_dk)):
            if tier:
                return tier
        return []

    def _op_crossover(self, cells, rng):
        """V6-R3 档案内杂交：cell 配置逐槽位均匀重组（结构保持亲本 A=self.state）。

        B 亲本槽位先按 A 当前结构过滤（结构已变异一步，B 的 slot 可能不存在）；
        A∪B 槽位逐个掷硬币继承。子代与 A 相同 → 返回 None（克隆不值 33 路 DC）。
        预算修复：无零 cell 的子代按闭式贪心摘（复用 _outer_drop_worst_cell）；
        含零 cell 跳过闭式（解析对边界列 ZERO 失真 3.7×，与 M2 zero 算子同语义），
        可行性交给 TT oracle / v5 档案准入。"""
        cands = self._v5_crossover_candidates()
        if not cands:
            return None
        ent, k_b = cands[int(rng.integers(len(cands)))]
        k_a = int(self.state.get("k", self.trunc_cols))
        dk = k_a - k_b
        # 异 k 亲本：B 的 cell 列坐标按截断边界重映射（c−kB 相对位形不变），
        # 移出资格窗/A 结构不存在的槽位由下方 slotset 过滤自然丢弃
        other = []
        for e in ((ent["payload"].get("ct") or {}).get("cells") or []):
            e = list(e)
            e[1] = int(e[1]) + dk
            other.append(e)
        slotset = {tuple(int(x) for x in s[:4])
                   for s in self._enumerate_type_slots(self._current_assignment())}
        a_map = {tuple(int(x) for x in e[:4]): list(e) for e in (cells or [])}
        b_map = {tuple(int(x) for x in e[:4]): list(e) for e in other
                 if tuple(int(x) for x in e[:4]) in slotset}
        if not a_map and not b_map:
            return None
        child = []
        for key in sorted(set(a_map) | set(b_map)):
            # 均匀杂交把"该槽无 cell"当等位基因：单边槽位也掷硬币决定去留，
            # 否则子代恒为并集，系统性偏向更多 cell/更大误差。
            allele = a_map.get(key) if rng.random() < 0.5 else b_map.get(key)
            if allele is not None:
                child.append(list(allele))
        same_as_a = ({tuple(int(x) for x in e) for e in child}
                     == {tuple(int(x) for x in e) for e in (cells or [])})
        if same_as_a:
            return None
        if not any(self._is_zero_cell(e) for e in child):
            for _ in range(self.outer_errgate_max_repairs):
                if not child or self._cells_budget_ok(child):
                    break
                child, dropped = self._outer_drop_worst_cell(child)
        logging.info("[outer] crossover：A=%d B=%d(kB=%d dk=%+d 候选%d) → 子代 %d cells",
                     len(a_map), len(b_map), k_b, dk, len(cands), len(child))
        return child

    def _current_assignment(self):
        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct = CompressorTree(
            pp, self.state["ct32"], self.state["ct22"], self.state.get("ct42")
        )
        return ct.compressor_assignment_fused()

    def _outer_mutate(self):
        """外环变异 v1.1：结构变异每轮必做（v1.0 的算子骰子把结构搜索强度砍到 40%，
        v2align 验证显示同误差下 power 系统性劣化 ~16%——结构/布线搜索被饿着了）；
        cell 层是闭式免费的叠加层，按 outer_p_struct/cell/resample 比例做
        不动/单 cell op/resample-K。直接更新 self.state（含 cells）。"""
        rng = np.random.default_rng(random.getrandbits(32))
        cells = [list(e) for e in (self.state.get("cells") or [])]

        # 1) 结构变异（必做，与非 outer 模式 reset 同强度同口径）
        action_mask = self.get_action_mask()
        action = random.choice(np.where(action_mask == 1)[0])
        self.transition(action)
        cells = self._cells_prune_stale(cells, self._current_assignment())

        # 2) cell 叠加层：keep / 单 cell op / resample-K / M2 批量 zero / V6 杂交
        has_types = bool(self.use_approx_types and self.type_table_32)
        use_zero = bool(getattr(self, "outer_zero_ops", False)) and has_types
        use_xover = (bool(getattr(self, "outer_crossover", False)) and has_types
                     and len(self._v5_crossover_candidates()) >= 2)
        arms = ["keep"]
        if has_types:
            arms += ["cell", "resample"]
        if use_zero:
            arms.append("zero")
        if use_xover:
            arms.append("crossover")
        if getattr(self, "outer_bandit", False) and self._v5_bandit is not None:
            # V6-R2：按 (箱,臂) Thompson 采样。种子集不走本函数，_v5_bin 必有效。
            op = self._v5_bandit.choose(self._v5_bin, arms, rng) or "keep"
            w, n = self._v5_bandit.stats_of(self._v5_bin, op)
            logging.info("[outer][bandit] bin=%d arm=%s (近%d次中%d)",
                         self._v5_bin, op, n, w)
        else:
            weights = {"keep": self.outer_p_struct,
                       "cell": self.outer_p_cell,
                       "resample": self.outer_p_resample,
                       "zero": self.outer_p_zero,
                       "crossover": self.outer_p_crossover}
            p = np.array([weights[a] for a in arms], dtype=float)
            p /= p.sum()
            op = arms[int(rng.choice(len(arms), p=p))]

        # V6-R1：被选臂掷 G 次收集去重配置（keep/求解器/非 v5 恒 G=1=旧行为）；
        # 组0 为主掷同步进 state["cells"]，组间仅 cell 层不同（结构/布线共享）。
        G = self.outer_multi_config
        if (op == "keep" or G <= 1 or self._cell_solver_active()
                or not getattr(self, "pareto_v5", False)):
            G = 1
        slots = (self._enumerate_type_slots(self._current_assignment())
                 if op in ("zero", "cell", "resample") else None)
        configs, seen = [], set()
        for _trial in range(3 * G):
            if len(configs) >= G:
                break
            cand = (None if op == "keep"
                    else self._outer_roll_cells(op, [list(e) for e in cells],
                                                slots, rng))
            if cand is None:
                continue
            key = frozenset(tuple(int(x) for x in e) for e in cand)
            if key in seen:
                continue
            seen.add(key)
            configs.append([list(e) for e in cand])
        if not configs:
            if op != "keep":
                logging.info("[outer] %s 多次提议均不可行，本轮 cell 维度不变", op)
            configs = [[list(e) for e in cells]]
        self._episode_cell_configs = configs
        self.state["cells"] = [list(e) for e in configs[0]]
        self._outer_last_op = op   # bandit 归因：_v5_admit_samples 按本臂回填输赢
        med, bias, wce = self._cells_error_totals(configs[0])
        logging.info("[outer] op=struct+%s n_cells=%d med=%.1f bias=%+.1f wce=%.0f"
                     " cfgs=%d%s",
                     op, len(configs[0]), med, bias, wce, len(configs),
                     "" if len(configs) == 1 else
                     f" sizes={[len(c) for c in configs]}")

    def _outer_roll_cells(self, op, cells, slots, rng):
        """单次掷臂：给定 op 在 cells 基础上产出一个候选配置；不可行返回 None。
        （V6-R1 从 _outer_mutate 抽出以支持多掷；语义与单掷版逐臂一致。）"""
        if op == "crossover":
            # V6-R3：三级放宽亲本池，cell 配置逐槽位均匀重组。
            cand = self._op_crossover(cells, rng)
            if cand is None:
                logging.info("[outer] crossover 无可行子代（无亲本/克隆）")
            return cand
        if op == "zero":
            # M2：批量 ZERO 跳过闭式预算过滤（解析对边界列 ZERO 失真 3.7×）；
            # 超标由 TT oracle/errgate 修、v5 里只是落进更松的箱去竞争。
            cand = self._op_zero_col(cells, slots, rng)
            if cand is None:
                cand = self._op_unzero_col(cells, rng)
            if cand is None:
                logging.info("[outer] zero 算子无可操作列")
            return cand
        for _ in range(self.outer_proposal_retries):
            if op == "cell":
                dice = rng.random()
                if dice < 0.5 or not cells:
                    add = self._propose_cell_add(cells, slots, rng)
                    cand = None if add is None else cells + [add]
                elif dice < 0.75:
                    cand = self._propose_cell_remove(cells, rng)
                else:
                    cand = self._propose_cell_swap(cells, slots, rng)
            else:
                cand = self._op_resample_k(slots, rng)
            if cand is not None and self._cells_budget_ok(cand):
                return cand
        return None

    def _outer_gate_active(self):
        """预筛门只在有硬预算可判定时开：MRED 模式看 mred_budget，否则看 med_budget。
        error_as_metric（无预算）没有"超标整集浪费"问题，不开。"""
        if not (self.outer_cell_search and self.outer_errgate):
            return False
        if getattr(self, "error_metric", "med") == "mred":
            return bool(getattr(self, "mred_budget", None))
        return self.med_budget is not None

    def _gate_budget_exceeded(self, measured):
        """与 get_objective 的离散预算判据同口径：MRED 模式比 measured mred，MED 模式
        比 measured med（均含截断 floor）。WCE 不进门（MC 尾部不收敛，沿用解析上界）。
        实测字段缺失 → 判不超（回退旧行为，别误杀）。返回 (超标?, 描述串)。"""
        if measured is None:
            return False, "no-measurement"
        if (getattr(self, "error_metric", "med") == "mred"
                and getattr(self, "mred_budget", None)):
            mred = measured.get("mred")
            if mred is None:
                return False, "mred-missing"
            # v5：超伪预算不是废样本（落进更松的箱竞争），门/修复只挡真出界
            # （mred > 档案上限 = 无箱可落）。预算模式维持旧语义。
            bud = (float(self._v5_archive.hi)
                   if getattr(self, "pareto_v5", False)
                   else float(self.mred_budget))
            return (mred > bud, f"mred={mred:.3e}/limit={bud:.3e}")
        if self.med_budget is not None:
            med = measured.get("med")
            if med is None:
                return False, "med-missing"
            return (med > float(self.med_budget),
                    f"med={med:.1f}/budget={float(self.med_budget):.1f}")
        return False, "no-budget"

    def _outer_drop_worst_cell(self, cells):
        """修复算子：摘掉解析误差贡献 wae·2^col 最大的 cell。每步总误差严格下降，
        且尽量多保留 cell（相对清空重摆，保住外环已积累的摆放）。"""
        if not cells:
            return cells, None

        def contrib(e):
            _h, table = self._type_head_and_table(int(e[2]))
            return float(table[int(e[4])]["wae"]) * float(1 << int(e[1]))

        j = max(range(len(cells)), key=lambda i: contrib(cells[i]))
        return cells[:j] + cells[j + 1:], cells[j]

    def _refresh_episode_cell_types(self):
        """state["cells"] → self._episode_cell_types {node_idx:(t,k)}（reset 同口径；
        预筛修复摘 cell 后重建映射）。返回未映射条数（正常应为 0）。"""
        self._episode_cell_types = {}
        unmapped = 0
        for e in self.state.get("cells") or []:
            key = (int(e[0]), int(e[1]), int(e[2]), int(e[3]))
            node_idx = self.comp_graph.indice_map.get(key)
            if node_idx is None:
                unmapped += 1
                continue
            self._episode_cell_types[int(node_idx)] = (int(e[2]), int(e[4]))
        return unmapped

    def _outer_errgate_screen(self, mul, samples_connection, rtl_path):
        """外环实测误差预筛门：sample-0 RTL 已发射，先 verilator 实测再放行 DC。
        超预算 → _outer_drop_worst_cell 修复 → 重发射重测；步数耗尽仍超 → 清空
        cells（floor 配置必可行）。verilator 探测失败 → 放行（与 error_gate 回退
        策略一致，不因门本身故障丢整集）。返回修复后的 (type_choices, cell_map)。"""
        probe_build = os.path.join(self.build_dir, "outer_errgate_probe")
        os.makedirs(probe_build, exist_ok=True)
        for rep in range(self.outer_errgate_max_repairs + 1):
            m = self._measure_error_verilator(
                rtl_path, probe_build, self.outer_errgate_vectors
            )
            if m is None:
                logging.warning("[outer-gate] verilator 探测失败，本轮跳过预筛")
                break
            over, detail = self._gate_budget_exceeded(m)
            if rep == 0:
                # 标定数据：解析 vs 实测（修 §3.1 slack 一阶近似用）
                a_med, _ab, _aw = self._cells_error_totals(self.state.get("cells"))
                logging.info(
                    "[outer-gate] probe n_cells=%d %s | analytic_med=%.1f measured_med=%.1f",
                    len(self.state.get("cells") or []), detail,
                    a_med, m.get("med") if m.get("med") is not None else float("nan"),
                )
            if not over:
                if rep:
                    logging.info("[outer-gate] 修复 %d 步后可行 (%s)", rep, detail)
                break
            cells = [list(e) for e in (self.state.get("cells") or [])]
            if rep >= self.outer_errgate_max_repairs or len(cells) <= 1:
                new_cells, dropped = [], None  # 保底：清空必可行
            else:
                new_cells, dropped = self._outer_drop_worst_cell(cells)
            self.state["cells"] = new_cells
            logging.info(
                "[outer-gate] 超budget(%s) → %s n_cells=%d→%d",
                detail,
                ("清空 cells 保底" if dropped is None else
                 f"摘除 (s{int(dropped[0])},c{int(dropped[1])},t{int(dropped[2])},"
                 f"#{int(dropped[3])},k{int(dropped[4])})"),
                len(cells), len(new_cells),
            )
            self._refresh_episode_cell_types()
            cell_map = self._cell_map_from_types(self._episode_cell_types)
            mul.emit_verilog(
                rtl_path,
                assignment=self.emit_assignment(samples_connection, cell_map=cell_map),
                extra_modules_src=self._approx_modules_src(cell_map),
            )
            if not new_cells:
                break  # 空配置必可行，不再复测
        type_choices = dict(self._episode_cell_types)
        return type_choices, self._cell_map_from_types(type_choices)

    def _outer_tt_oracle_screen(self, mul, samples_connection, rtl_path):
        """M2 TT oracle 预筛（PARETO_ARITH_PLAN.md §7.2）：sample-0 布线上用 cellsolver
        张量化仿真器 + 分层估计器实测本集 cell 配置 mred——与 16M verilator 闸门同流
        逐位一致（oracle 说可行 ⇒ 闸门必过），秒级；治解析一阶 slack 对密集 ZERO 包
        的误杀与边界列 bias 3.7× 失真。超上限按解析贡献 wae·2^col 降序二分前缀摘除
        （robust greedy 同款）。上限：v5 = 档案 mred 上限（伪预算内外都放行、落箱
        竞争）；预算模式 = mred_budget。oracle 异常回退 errgate（若开）或放行。"""
        def _current():
            tc = dict(self._episode_cell_types)
            return tc, self._cell_map_from_types(tc)

        if getattr(self, "pareto_v5", False):
            limit = float(self._v5_archive.hi)
        elif getattr(self, "mred_budget", None):
            limit = float(self.mred_budget)
        else:
            return _current()
        if not self._episode_cell_types:
            return _current()
        try:
            import torch as _torch
            from Appr_Comp.cellsolver import sim as _cs
            from Appr_Comp.cellsolver.solver import GradientCellSolver

            dev = getattr(self, "device", "cpu")
            if (isinstance(dev, str) and dev.startswith("cuda")
                    and not _torch.cuda.is_available()):
                dev = "cpu"
            specs = _cs.parse_pp_specs(mul.emit_pp_encoder())
            tree = _cs.TreeSim(self.comp_graph, samples_connection, specs, dev)
            cache = self.outer_solver_cache or os.path.join(
                self.build_dir, "solver_pool")
            os.makedirs(cache, exist_ok=True)
            solver = GradientCellSolver(
                self, tree, specs, limit, device=dev,
                pool_vectors=self.outer_solver_vectors, cache_dir=cache,
                est=getattr(self, "_cell_solver_est", None),
            )
            self._cell_solver_est = solver.est
            cfg = {int(n): (int(t), int(k))
                   for n, (t, k) in self._episode_cell_types.items()}

            def _mred_of(c):
                return solver.est.gate(tree, specs, self.bit_width,
                                       solver.space.cell_luts_of(c))

            mred0 = _mred_of(cfg)
            a_med, _ab, _aw = self._cells_error_totals(self.state.get("cells"))
            logging.info(
                "[tt-oracle] n_cells=%d mred_sim=%.3e limit=%.3e (util=%.0f%%) "
                "analytic_med=%.1f", len(cfg), mred0, limit,
                mred0 / limit * 100, a_med,
            )
            if mred0 <= limit or not cfg:
                return _current()
            colmap = {n: c for n, _t, c in solver.space.slots}

            def _contrib(n):
                return solver.space.wae_of(*cfg[n]) * (2 ** colmap.get(n, 0))

            order = sorted(cfg, key=_contrib, reverse=True)

            def _prefix(m):
                c = dict(cfg)
                for n in order[:m]:
                    c.pop(n)
                return c

            lo, hi = 1, len(order)
            while lo < hi:
                mid = (lo + hi) // 2
                if _mred_of(_prefix(mid)) <= limit:
                    hi = mid
                else:
                    lo = mid + 1
            cfg = _prefix(lo)
            drops = lo
            while cfg and _mred_of(cfg) > limit:   # 非单调兜底
                node = max(cfg, key=_contrib)
                cfg.pop(node)
                drops += 1
        except Exception as e:  # noqa: BLE001
            logging.warning("[tt-oracle] 异常(%s)，回退 %s", e,
                            "errgate" if self._outer_gate_active() else "放行")
            if self._outer_gate_active():
                return self._outer_errgate_screen(mul, samples_connection, rtl_path)
            return _current()
        # 修剪写回 + 重发射 sample-0 RTL（后续样本自然沿用修剪后的配置）
        vlist = self.comp_graph.vertex_list
        cells = []
        for node, (t, k) in cfg.items():
            s, c, _t, idx = vlist[int(node)]
            cells.append([int(s), int(c), int(t), int(idx), int(k)])
        self.state["cells"] = cells
        self._refresh_episode_cell_types()
        type_choices, cell_map = _current()
        mul.emit_verilog(
            rtl_path,
            assignment=self.emit_assignment(samples_connection, cell_map=cell_map),
            extra_modules_src=self._approx_modules_src(cell_map),
        )
        logging.info("[tt-oracle] 超上限修剪：摘 %d 个 → n_cells=%d", drops, len(cells))
        return type_choices, cell_map

    def _outer_screen_group(self, g, mul, samples_connection, rtl_path):
        """V6-R1：组 g 过预筛门。TT oracle/errgate 读写 state["cells"] 与
        _episode_cell_types（单配置遗留接口）——临时切到组 g 视图，修剪结果
        写回该组，完事恢复组0 默认视图。G=1 时等价旧的 sample-0 单次过门。"""
        cfgs = self._episode_cell_configs
        self.state["cells"] = [list(e) for e in cfgs[g]]
        self._episode_cell_types = dict(self._episode_ct_groups[g])
        if getattr(self, "outer_tt_oracle", False):
            tc, cm = self._outer_tt_oracle_screen(mul, samples_connection, rtl_path)
        else:
            tc, cm = self._outer_errgate_screen(mul, samples_connection, rtl_path)
        cfgs[g] = [list(e) for e in (self.state.get("cells") or [])]
        self._episode_ct_groups[g] = dict(self._episode_cell_types)
        self.state["cells"] = [list(e) for e in cfgs[0]]
        self._episode_cell_types = dict(self._episode_ct_groups[0])
        return tc, cm

    def _cell_solver_active(self):
        """greedy 求解器是否生效：需 outer_cell_search + solver="greedy" + MRED 预算模式。
        v5 种子集（_v5_seeding）不生效——种子是纯截断 Dadda 基线本体，greedy 填 cell 会
        破坏可复现基线（与 GA 变体种子集不变异对等）。"""
        return bool(
            self.outer_cell_search
            and self.outer_cell_solver == "greedy"
            and getattr(self, "error_metric", None) == "mred"
            and self.mred_budget
            and self.trunc_cols > 0
            and not getattr(self, "_v5_seeding", False)
        )

    def _outer_greedy_solve_robust(self, connections):
        """外环 greedy 求解器（鲁棒版,替代进化 cell 变异）：在 sample-0 布线上用张量化
        仿真器实测 Δmred 打分解 cell 包(到 budget×margin),再对整集**全部布线**复测
        (sim 与 verilator gate 同 16M 流逐位一致 → sim 全合规 = gate 必过),任一布线
        超全额 budget → 摘解析贡献 wae·2^col 最大的 cell,直至全布线合规。
        动机:密集包(50~65 cell)的误差抵消强依赖布线,只按 sample-0 解会跨布线偏高
        10%+,07-10 首集 3 个 k 全部 7/9 样本越线报废。
        成功后更新 state["cells"]/_episode_cell_types(发射由 get_samples 主循环做);
        求解失败/无 slot → 空 cells 放行(纯截断必可行),不丢整集。"""
        self.state["cells"] = []
        self._refresh_episode_cell_types()
        try:
            import torch as _torch
            from Appr_Comp.cellsolver import sim as _cs
            from Appr_Comp.cellsolver.solver import GradientCellSolver
        except Exception as e:  # noqa: BLE001
            logging.warning("[outer-solver] 导入 cellsolver 失败,跳过求解: %s", e)
            return
        dev = getattr(self, "device", "cpu")
        if isinstance(dev, str) and dev.startswith("cuda") and not _torch.cuda.is_available():
            dev = "cpu"
        try:
            ct = CompressorTree(
                self.initial_pp, self.state["ct32"], self.state["ct22"],
                self.state.get("ct42"),
            )
            if self.trunc_cols > 0:
                ct.trunc_cols = self.trunc_cols
                ct.trunc_bits = self._trunc_bits
            mul = Mul(self.bit_width, self.encode_type, ct)
            specs = _cs.parse_pp_specs(mul.emit_pp_encoder())
            trees = [_cs.TreeSim(self.comp_graph, conn, specs, dev)
                     for conn in connections]
            cache = self.outer_solver_cache or os.path.join(self.build_dir, "solver_pool")
            os.makedirs(cache, exist_ok=True)
            solver = GradientCellSolver(
                self, trees[0], specs,
                float(self.mred_budget) * self.outer_solver_margin, device=dev,
                pool_vectors=self.outer_solver_vectors, cache_dir=cache,
                est=getattr(self, "_cell_solver_est", None),  # 跨 episode 复用池/分层
            )
            self._cell_solver_est = solver.est
            if not solver.space.slots:
                logging.info("[outer-solver] 无合法 slot（资格带空）→ 纯截断")
                return
            # 鲁棒模式跳过升级扫描：贴线换面积的增量会被跨布线修复摘掉,白做
            cfg = solver.greedy_add(log=lambda *a, **k: None, upgrade=False)
            # 鲁棒修复：全布线合规（对全额 budget,非 margin 后的）。
            # 二分批量摘除：按解析贡献 wae·2^col 降序排出摘除顺序,二分最小前缀
            # (log2(n)×trees 次 gate,替代逐个摘的 n×trees 次);非单调兜底逐摘。
            budget = float(self.mred_budget)
            colmap = {n: c for n, _t, c in solver.space.slots}

            def _contrib(n):
                return solver.space.wae_of(*cfg_full[n]) * (2 ** colmap.get(n, 0))

            def _worst(c):
                luts = solver.space.cell_luts_of(c)
                return max(solver.est.gate(t, specs, self.bit_width, luts)
                           for t in trees)

            cfg_full = dict(cfg)
            drops = 0
            worst_val = _worst(cfg)
            if worst_val > budget and cfg:
                order = sorted(cfg_full, key=_contrib, reverse=True)

                def _drop_prefix(m):
                    c = dict(cfg_full)
                    for n in order[:m]:
                        c.pop(n)
                    return c

                lo, hi = 1, len(order)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _worst(_drop_prefix(mid)) <= budget:
                        hi = mid
                    else:
                        lo = mid + 1
                cfg = _drop_prefix(lo)
                drops = lo
                worst_val = _worst(cfg)
                while cfg and worst_val > budget:  # 非单调兜底
                    node = max(cfg, key=lambda n: solver.space.wae_of(*cfg[n])
                               * (2 ** colmap.get(n, 0)))
                    cfg.pop(node)
                    drops += 1
                    worst_val = _worst(cfg)
        except Exception as e:  # noqa: BLE001
            logging.warning("[outer-solver] 求解异常,空 cells 放行: %s", e)
            self.state["cells"] = []
            self._refresh_episode_cell_types()
            return
        # cfg {node_idx:(t,k)} → state["cells"] [[s,c,t,idx,k]]（vertex_list 反查坐标）
        vlist = self.comp_graph.vertex_list
        cells = []
        for node, (t, k) in cfg.items():
            s, c, _t, idx = vlist[int(node)]
            cells.append([int(s), int(c), int(t), int(idx), int(k)])
        self.state["cells"] = cells
        self._refresh_episode_cell_types()
        logging.info(
            "[outer-solver] robust greedy n_cells=%d(摘%d) worst_mred=%.3e "
            "worst_util=%.1f%% slots=%d trees=%d",
            len(cfg), drops, worst_val, worst_val / budget * 100,
            len(solver.space.slots), len(trees),
        )
