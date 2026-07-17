"""v5 非支配档案：按 mred 分箱的 ParetoArchive 准入/亲代采样/代表解，
前沿状态加载与导出。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
import random
import copy
import json
import logging


import numpy as np

from utils import (
    CompressorTree,
    convert_to_serializable,
    ParetoArchive,
    OpBandit,
)


class ParetoFrontMixin:
    """v5 非支配档案：分箱准入、亲代采样、代表解与前沿导入导出。"""

    def _dump_front_snapshot(self, episode_idx):
        """轻量前沿快照：与 save_experiment 的 front.json 同构（不含 payload），
        tmp+rename 原子写，rsync 半途拉不到残文件。"""
        d = os.path.join(self.log_dir, "front_hist")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"front_ep{episode_idx + 1:04d}.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._v5_archive.snapshot(), f, indent=1,
                      default=convert_to_serializable)
        os.replace(tmp, p)

    def _seed_pool_from_best_info(self, ref_shape):
        """温启动：把已有 run 的 best_info.json 种进初始池 + found_best_info。
        返回成功加载的条数；0 表示未启用/全失败（调用方回退默认种子）。
        口径要求：同 bit_width/encode_type/trunc_cols/objective scales（objective 直接复用）。"""
        if not self.init_pool_best_info:
            return 0
        n_loaded = 0
        for path in self.init_pool_best_info:
            try:
                with open(self._resolve_path(path)) as f:
                    bi = json.load(f)
                for kk in ("ct32", "ct22", "ct42"):
                    if bi["ct"].get(kk) is not None:
                        bi["ct"][kk] = np.asarray(bi["ct"][kk], dtype=int)
                if bi["ct"]["ct32"].shape != tuple(ref_shape):
                    raise ValueError(
                        f"ct32 shape {bi['ct']['ct32'].shape} != env {tuple(ref_shape)}"
                        " (bit_width/encode_type/trunc_cols 不一致?)")
                if isinstance(bi.get("assignment"), list):
                    bi["assignment"] = [[[tuple(v) for v in col] for col in stage]
                                        for stage in bi["assignment"]]
                state = {"ct32": bi["ct"]["ct32"].copy(),
                         "ct22": bi["ct"]["ct22"].copy()}
                if self.use_ct42:
                    state["ct42"] = (bi["ct"]["ct42"].copy()
                                     if bi["ct"].get("ct42") is not None
                                     else np.zeros_like(state["ct32"]))
                if self.outer_cell_search:
                    # 外环模式 cells 存在 ct["cells"]（state 整体 deepcopy 进 best_info["ct"]）
                    cells_src = bi["ct"].get("cells") or bi.get("cells") or []
                    state["cells"] = [list(c) for c in cells_src]
                objective = float(bi["objective"])
                self.pool.add(objective, state)
                # 种 found_best_info：本 run 的 best 单调不劣于温启动（rank 同 objective 口径）
                if (bi.get("connection") is not None
                        and objective < self.found_best_info["objective"]):
                    bi["ct"] = dict(state)
                    self.found_best_info = bi
                n_loaded += 1
                logging.info(
                    f"warm-start pool <- {path}: objective={objective:.4f} "
                    f"area={bi.get('area')} med={((bi.get('measured_error') or {}).get('med'))}")
            except Exception as e:  # noqa: BLE001
                logging.error(f"warm-start load failed {path}: {e}")
        return n_loaded

    # ===================== v5 多目标：混合 k 种群 + 非支配档案 =====================
    # PARETO_ARITH_PLAN.md。默认关（不调 enable_pareto_v5 = 逐位旧行为）。
    # 选择层零系数：存活/亲代只由 ParetoArchive 支配关系决定；标量 objective 退到
    # PPO 提议启发层（伪预算 = 亲代箱上沿，铰链无量纲化，不再决定谁存活）。

    @property
    def found_best_info(self):
        if getattr(self, "pareto_v5", False):
            rep = self._v5_representative()
            if rep is not None:
                return rep
        return self._found_best_info

    @found_best_info.setter
    def found_best_info(self, value):
        self._found_best_info = value

    def enable_pareto_v5(self, mred_lo=1e-7, mred_hi=2e-1, bin_ratio=2.0,
                         bin_cap=6, eps_power=0.01, seed_ks=None, binless=False):
        """v5 入口（train_dc.py 在 error_metric=mred 接线后调用）。
        seed_ks = 初始种群的截断深度集合（不同 k 截断的 Dadda 树，论文可复现基线）。
        binless = 无箱消融（07-16）：全局 3 目标支配档案，bin_cap 语义 = 总容量；
        调度改为"随机亲代 + 伪预算 = 亲代 mred × ratio"（无箱轮询可轮）。"""
        if getattr(self, "error_metric", "med") != "mred":
            raise ValueError("pareto_v5 需要 error_metric='mred'")
        self.pareto_v5 = True
        self._v5_binless = bool(binless)
        self._v5_archive = ParetoArchive(
            mred_lo=mred_lo, mred_hi=mred_hi, bin_ratio=bin_ratio,
            bin_cap=bin_cap, eps_power=eps_power, binless=binless,
        )
        if seed_ks is not None and not list(seed_ks):
            raise ValueError("pareto_v5 seed_ks 为空（解析失败？）——显式传 None 用默认 2-30")
        self._v5_seed_queue = sorted({int(k) for k in (seed_ks or range(2, 31))})
        self._v5_seed_tries = {}
        self._v5_state_override = None
        self._v5_seeding = False
        self._v5_seed_k = None
        self._v5_bin = 0
        if getattr(self, "outer_bandit", False):
            # V6-R2：臂集合固定建全（可用性在 _outer_mutate 逐轮判定），统计按需增长
            self._v5_bandit = OpBandit(
                ["keep", "cell", "resample", "zero", "crossover"],
                window=self.outer_bandit_window, floor=self.outer_bandit_floor,
            )
        if not hasattr(self, "_trunc_profiles"):
            self._trunc_profiles = {}
            if self.trunc_cols > 0 and self._trunc_bits:
                self._trunc_profiles[self.trunc_cols] = self._capture_trunc_profile()
        logging.info(
            "[v5] enabled: bins=%d%s [%.1e, %.1e] ratio=%.2f cap=%d eps_pwr=%.3f "
            "seed_ks=%s", self._v5_archive.n_bins,
            " (BINLESS 全局3目标档案)" if binless else "",
            mred_lo, mred_hi, bin_ratio,
            bin_cap, eps_power, self._v5_seed_queue,
        )

    def v5_load_front_state(self, path):
        """V6 温启动：把已完成 run 的滚动档案 front_state.json 全量载入 ParetoArchive。

        条目走 add() 正常准入（isfinite/支配/去重/容量全套把关），所以跨 bin 配置
        （lo/hi/ratio/cap 不同）也安全——按 mred 重新分箱。已被档案覆盖的 k 从种子
        队列剔除（阶梯已在，不重评）；bandit 统计（若有）一并恢复。
        口径要求：同 bit_width/encode_type/菜单文件（payload 的 cell 类型索引直接复用）。"""
        assert getattr(self, "pareto_v5", False), "先 enable_pareto_v5 再温启动"

        def _tupleize(x):
            # JSON 往返把元组变列表；发射/诊断路径拿 vertex_info 当 dict key
            # （CompressorGraph.indice_map）——不还原会 TypeError: unhashable
            # （r3 首个 save 实测）。dict 保形，值递归。
            if isinstance(x, list):
                return tuple(_tupleize(v) for v in x)
            if isinstance(x, dict):
                return {k: _tupleize(v) for k, v in x.items()}
            return x

        with open(path) as f:
            st = json.load(f)
        n_in = n_ok = 0
        for _b, ents in sorted((st.get("bins") or {}).items(), key=lambda x: int(x[0])):
            for e in ents:
                n_in += 1
                pl = e.get("payload")
                if pl:
                    for fld in ("assignment", "connection"):
                        if pl.get(fld) is not None:
                            pl[fld] = [_tupleize(v) for v in pl[fld]]
                ok, _bb = self._v5_archive.add(
                    e.get("mred"), e.get("area"), e.get("power"), pl)
                n_ok += bool(ok)
        seeds_before = list(self._v5_seed_queue)
        if "seed_queue" in st:
            # 忠实续跑：直接继承上一战役结束时的种子队列（含它已消费/已放弃的判定）。
            # 否则被支配挤出档案的 k 会按覆盖剪枝复活，白烧最多 3 轮/个重评。
            self._v5_seed_queue = sorted({int(k) for k in st["seed_queue"]})
        else:
            have_ks = {int(((e.get("payload") or {}).get("ct") or {}).get("k", -1))
                       for es in self._v5_archive.bins.values() for e in es}
            self._v5_seed_queue = [k for k in self._v5_seed_queue
                                   if k not in have_ks]
        if self._v5_bandit is not None and st.get("bandit"):
            self._v5_bandit.load(st["bandit"])
        logging.info(
            "[v5] warm-start %s: 载入 %d/%d 条目 -> archive=%d pts/%d bins; "
            "seeds %s -> %s (档案已覆盖 k 不重评)",
            path, n_ok, n_in, len(self._v5_archive), self._v5_archive.n_nonempty(),
            seeds_before, self._v5_seed_queue,
        )
        if n_ok == 0:
            raise ValueError(f"warm-start 载入 0 条目（{path} 空/口径不符？）——"
                             "拒绝静默冷启动，请检查路径或去掉 --v5_warm_state")
        return n_ok

    def _v5_dadda_state(self, k):
        """初始种群个体：标准 Dadda 树 + 截断 k + 零近似 cell（从零确定性构造）。"""
        ct = CompressorTree.dadda(self.initial_pp)
        st = {"ct32": ct.ct32.astype(int), "ct22": ct.ct22.astype(int)}
        if self.use_ct42:
            st["ct42"] = np.zeros_like(st["ct32"])
        if self.outer_cell_search:
            st["cells"] = []
        st["k"] = int(k)
        return st

    def _v5_begin_episode(self, episode_idx):
        """每集编排：种子集（初始种群逐 k 评估，不变异）优先，之后箱轮询。
        伪预算 = 本集箱上沿；mred_scale = 伪预算（铰链无量纲化，跨箱同尺度）。"""
        self._v5_state_override = None
        self._v5_seeding = False
        self._v5_seed_k = None
        self._outer_last_op = None   # 种子集不掷骰子；防止 bandit 归因到上集的臂
        arch = self._v5_archive
        while self._v5_seed_queue:
            # 双评审发现 #3 修复：peek 不 pop——种子只在评估成功（_v5_admit_samples
            # 至少 1 个入档）后才出队；全批 DC/verilator 失败 → 下集自动重试（上限 3 次）
            k = self._v5_seed_queue[0]
            self._activate_trunc_profile(k)
            floor = self._trunc_model_mred
            if floor is not None and floor > arch.hi:
                logging.info("[v5] seed k=%d 模型floor=%.3e 超档案上限 %.1e，跳过",
                             k, floor, arch.hi)
                self._v5_seed_queue.pop(0)
                continue
            tries = self._v5_seed_tries.get(k, 0)
            if tries >= 3:
                logging.error("[v5] seed k=%d 连续 %d 次评估失败，放弃该基线", k, tries)
                self._v5_seed_queue.pop(0)
                continue
            self._v5_seed_tries[k] = tries + 1
            b = arch.bin_of(floor if floor is not None else arch.lo)
            self._v5_bin = arch.n_bins - 1 if b is None else b
            self._v5_state_override = self._v5_dadda_state(k)
            self._v5_seeding = True
            self._v5_seed_k = k
            break
        binless = getattr(self, "_v5_binless", False)
        if not self._v5_seeding:
            self._v5_bin = 0 if binless else episode_idx % arch.n_bins
        if binless:
            # binless 消融：伪预算跟亲代走（= 亲代 mred × ratio，clip 到 [lo·ratio, hi]）。
            # 亲代在此处就采定并覆写（预算和实际亲代必须一致）；种子集用模型 floor。
            if self._v5_seeding:
                anchor = self._trunc_model_mred or arch.lo
            else:
                ent = arch.sample_parent(0, random)
                if ent is not None:
                    st = copy.deepcopy(ent["payload"]["ct"])
                    for kk in ("ct32", "ct22", "ct42"):
                        if st.get(kk) is not None:
                            st[kk] = np.asarray(st[kk], dtype=int)
                    self._v5_state_override = st
                    anchor = ent["mred"]
                else:
                    anchor = arch.lo   # 档案空：回退池路径，预算给最紧档
            hi_e = min(arch.hi, max(float(anchor), arch.lo) * arch.ratio)
        else:
            _lo_e, hi_e = arch.bin_edges(self._v5_bin)
        self.mred_budget = hi_e
        self.mred_scale = hi_e
        logging.info(
            "[v5] ep %d bin=%s pseudo_budget=%.3e%s archive=%d pts/%d bins",
            episode_idx,
            "G" if binless else f"{self._v5_bin}/{arch.n_bins}", hi_e,
            f" SEED(dadda k={self._v5_seed_k})" if self._v5_seeding else "",
            len(arch), arch.n_nonempty(),
        )

    def _v5_sample_parent_state(self):
        """亲代：种子覆写优先；否则本集箱（空则最近非空箱）均匀取。档案全空返回 None
        （reset 回退旧池路径，池里有初始 Dadda）。"""
        if self._v5_state_override is not None:
            st, self._v5_state_override = self._v5_state_override, None
            return copy.deepcopy(st)
        ent = self._v5_archive.sample_parent(self._v5_bin, random)
        if ent is None:
            return None
        st = copy.deepcopy(ent["payload"]["ct"])
        for kk in ("ct32", "ct22", "ct42"):
            if st.get(kk) is not None:
                st[kk] = np.asarray(st[kk], dtype=int)
        return st

    def _v5_admit_samples(self, sample_info_list):
        """支配准入（替代 legacy found_best 标量更新）：每个实测过 mred 的样本按
        (mred→箱, area, power) 进档案；无实测 mred（verilator 失败）不入箱。"""
        arch = self._v5_archive
        n_ok, n_skip = 0, 0
        n_arm_ok = 0     # bandit 归因口径：排除 all-exact 保底等 baseline 样本
        admitted_bins = []
        for s in sample_info_list:
            me = s.get("measured_error")
            mred = (me or {}).get("mred")
            if mred is None:
                n_skip += 1
                continue
            summary = self._summarize_result(s["result"])
            payload = {
                "objective": s["objective"],
                "connection": s["connection"],
                "ct": copy.deepcopy(self.state),
                "assignment": copy.deepcopy(self.assignment),
                "simulated_result": s["result"],
                "cell_types": copy.deepcopy(s.get("cell_types")),
                "cell_type_info": copy.deepcopy(s.get("cell_type_info")),
                "measured_error": copy.deepcopy(me),
                "error_source": me.get("source", "verilator"),
            }
            payload["ct"]["k"] = int(self.trunc_cols)
            if s.get("outer_cells") is not None:
                # V6-R1：分组样本的 ct.cells = 本组配置（state 里只有组0）
                payload["ct"]["cells"] = copy.deepcopy(s["outer_cells"])
            if not s.get("cell_types"):
                # exact-inject 基线等无 cell 样本：payload 结构如实反映该设计
                # （state 的 cells 不属于它，作亲代复用时不应复活）
                payload["ct"]["cells"] = []
            payload.update({k2: summary[k2] for k2 in ("area", "delay", "power")})
            ok, b = arch.add(mred, summary["area"], summary["power"], payload)
            if ok:
                n_ok += 1
                admitted_bins.append(b)
                if not s.get("baseline_only"):
                    n_arm_ok += 1
        if self._v5_seeding and self._v5_seed_k is not None and n_ok > 0:
            # 种子评估成功 → 此刻才正式出队（配合 begin 的 peek+重试）
            try:
                self._v5_seed_queue.remove(self._v5_seed_k)
            except ValueError:
                pass
        if (self._v5_bandit is not None and not self._v5_seeding
                and self._outer_last_op is not None):
            # V6-R2 归因：本臂本箱是否 ≥1 入档（二值，防种子期大丰收拉爆先验）。
            # codex 弱点1修复：all-exact 保底样本（baseline_only）与臂无关，其入档
            # 不得记为臂的胜利——低箱里 exact 基线常入档，会把臂统计系统性喂成
            # 假赢。归因用 n_arm_ok；种子出队仍看 n_ok（种子集本就全是纯截断）。
            self._v5_bandit.update(self._v5_bin, self._outer_last_op, n_arm_ok > 0)
        logging.info(
            "[v5] admit %d/%d (arm_ok %d, no-mred skip %d) -> bins %s"
            " | archive=%d pts/%d bins",
            n_ok, len(sample_info_list), n_arm_ok, n_skip, sorted(set(admitted_bins)),
            len(arch), arch.n_nonempty(),
        )

    def _v5_representative(self):
        """箱代表（供 legacy 日志/save/export 路径读 found_best_info）：当前箱内最小
        面积条目，箱空则全档案最小面积条目。档案空返回 None（回退 _found_best_info）。"""
        arch = getattr(self, "_v5_archive", None)
        if arch is None or len(arch) == 0:
            return None
        ents = arch.bins.get(getattr(self, "_v5_bin", 0)) or []
        ent = min(ents, key=lambda e: e["area"]) if ents else arch.global_min_area()
        return ent["payload"] if ent is not None else None

    def export_front(self, export_dir):
        """导出整个前沿：每条目一个 k*/ 子目录（MUL.v + best_info.json），目录名以 k
        开头以兼容 reeval_xa_glob_tmpbuild.py 的 k* glob；外加 front.json 清单。"""
        assert getattr(self, "pareto_v5", False)
        os.makedirs(export_dir, exist_ok=True)
        n_exp = 0
        for b in range(self._v5_archive.n_bins):
            for j, e in enumerate(
                sorted(self._v5_archive.bins[b], key=lambda x: x["area"])
            ):
                kk = int((e["payload"].get("ct") or {}).get("k", self.trunc_cols))
                name = f"k{kk:02d}_bin{b:02d}_{j}"
                try:
                    self.export_best_candidate(
                        os.path.join(export_dir, name), info=e["payload"]
                    )
                    n_exp += 1
                except Exception as ex:  # noqa: BLE001
                    logging.error("[v5] export %s failed: %s", name, ex)
        with open(os.path.join(export_dir, "front.json"), "w") as f:
            json.dump(self._v5_archive.snapshot(), f, indent=2,
                      default=convert_to_serializable)
        logging.info("[v5] front exported: %d designs -> %s", n_exp, export_dir)
        return n_exp
