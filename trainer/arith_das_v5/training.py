"""训练机制（RL 核心）：reward/objective 计算、advantage 分组归一、
PPO loss、run_episode 主循环、episode 日志。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
from typing import Dict, List, Tuple
import copy
import time
import logging

import torch

import numpy as np


class TrainingMixin:
    """RL 训练核心：objective、advantage 分组、PPO loss、episode 主循环。"""

    def run_experiment(self):
        for episode_idx in range(self.num_episodes):
            self.run_episode(episode_idx)
            if (getattr(self, "pareto_v5", False) and self.front_dump_freq
                    and (episode_idx + 1) % self.front_dump_freq == 0):
                self._dump_front_snapshot(episode_idx)
            if (episode_idx + 1) % self.save_freq == 0:
                self.save_experiment(episode_idx)
        self.save_experiment(self.num_episodes)

    def get_rule_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        l = 0.0
        time_start = time.time()
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            keys = ["a", "b", "c", "d"][: self.comp_graph.port_num]
            z_ports = []
            for key in keys:
                z = Z_mat_slice[f"s{key}"]
                if c > 0:
                    z = torch.cat([z, Z_mat_slice[f"c{key}"]], dim=0)
                z_ports.append(z)
            Z = torch.cat(z_ports, dim=1)
            mask = Z > -1e6
            p = torch.softmax(Z, dim=0).masked_fill(~mask, 0.0)
            row_sum = torch.sum(p, dim=1)
            row_sum_target = (torch.sum(mask.float(), dim=1) > 0).float()
            l += torch.sum(torch.pow(row_sum - row_sum_target, 2))

        time_end = time.time()
        return l

    def get_discrete_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        l = 0.0
        time_start = time.time()
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            keys = ["a", "b", "c", "d"][: self.comp_graph.port_num]
            z_ports = []
            for key in keys:
                z = Z_mat_slice[f"s{key}"]
                if c > 0:
                    z = torch.cat([z, Z_mat_slice[f"c{key}"]], dim=0)
                z_ports.append(z)
            Z = torch.cat(z_ports, dim=1)
            mask = Z > -1e6
            p = torch.softmax(Z, dim=0).masked_fill(~mask, 0.0)
            l += torch.sum(torch.pow((p * (1 - p)), 2))
        time_end = time.time()
        return l

    def get_objective(self, simulated_result, cell_types=None, measured_error=None):
        summary = self._summarize_result(simulated_result)

        # Phase B/C：近似+截断误差成本（约束式 A，LSB 单位）。两者全关时为 0，不影响。
        # codex review(medium)：截断误差不应被 use_approx_types/cell_types 门控——纯截断
        # （trunc_cols>0 但无类型搜索）也要计入，否则误差预算骗不过。
        err_term = 0.0
        if (self.use_approx_types and cell_types) or self.trunc_cols > 0:
            med, abs_bias, _nmed, wce = self._analytic_error(cell_types or {})
            # 误差闸门（codex 审过）：verilator 模式且实测可用 → med/bias 用 circular-wrap 真实值
            # （解析 proxy 系统性低估 0–30%）；wce 始终用解析上界（MC 尾部不收敛、不可信）。
            # verilator 失败(measured_error=None) → 回退解析 med/bias（不浪费已花的 DC）。
            if self.error_gate == "verilator" and measured_error is not None:
                med = measured_error["med"]
                abs_bias = abs(measured_error["bias"])
            # ── MRED 模式：误差用相对误差 MRED 当软罚闸门（verilator 实测）。深截断毁小积→
            # MRED 被重罚。无 bias 项（相对误差无干净的 bias 分解）。error_metric 默认 "med"
            # 保持向后兼容；mred 项尺度由 mred_scale（默认 0.01）归一。
            if getattr(self, "error_metric", "med") == "mred":
                mred = (measured_error or {}).get("mred")
                if mred is not None:   # verilator 失败(罕见，2 次重试)时 mred=None → 不罚该样本
                    budget = getattr(self, "mred_budget", 0.0) or 0.0
                    scale = getattr(self, "mred_scale", 0.01) or 0.01
                    err_term += self.med_violation_weight * max(0.0, mred - budget) / scale
            # 点2：除 error_scale 把 LSB 绝对值压到和 PPA 同量级。
            elif self.error_as_metric:
                # 误差作为普通目标项，评估方式同 area/power：error_weight*med/error_scale
                # （线性计入，无 budget 铰链；med_budget/med_violation_weight 此模式忽略）。
                err_term += self.error_weight * med / self.error_scale
            elif self.med_budget is not None:
                err_term += (
                    self.med_violation_weight
                    * max(0.0, med - self.med_budget)
                    / self.error_scale
                )
            if getattr(self, "error_metric", "med") != "mred":
                err_term += self.bias_weight * abs_bias / self.error_scale
            # ④ 尾部/WCE 约束（默认关）：超出 wce_budget 才罚，压重尾/最坏情况误差。
            if self.wce_budget is not None:
                err_term += (
                    self.wce_violation_weight
                    * max(0.0, wce - self.wce_budget)
                    / self.error_scale
                )

        # （area_budget 约束模式的目标分支已剪除：train_dc 固定 area_budget=None。）
        if self.delay_as_constraint:
            # P0(codex)：delay≤target 不奖励也不罚（不再把预算花在压 delay）；超 target 才罚。
            dtarget = (
                self.delay_target_ns
                if self.delay_target_ns is not None
                else self.fixed_target_delay
            )
            delay_cost = 0.0
            if dtarget is not None:
                delay_cost = (
                    self.delay_violation_weight
                    * max(0.0, summary["delay"] - float(dtarget))
                    / self.delay_scale
                )
            objective = (
                delay_cost
                + self.area_weight * summary["area"] / self.area_scale
                + self.power_weight * summary["power"] / self.power_scale
            )
        else:
            objective = (
                self.delay_weight * summary["delay"] / self.delay_scale
                + self.area_weight * summary["area"] / self.area_scale
                + self.power_weight * summary["power"] / self.power_scale
            )
        return objective + err_term

    def _candidate_rank(self, sample_info):
        result = sample_info.get("result", sample_info.get("simulated_result"))
        summary = self._summarize_result(result)
        # 方案1：MRED ε-硬约束。可行(mred≤budget)优先，可行里按 PPA(area+power 归一)最小；
        # 不可行按超额量排。mred 没测到(verilator 失败)→ 最差档，避免被当"可行"误选为 best。
        if getattr(self, "error_metric", "med") == "mred" and getattr(self, "mred_budget", None):
            me = sample_info.get("measured_error") or {}
            mred = me.get("mred")
            ppa = (self.area_weight * summary["area"] / self.area_scale
                   + self.power_weight * summary["power"] / self.power_scale)
            if mred is None:
                return (2, float("inf"), ppa)
            return (0 if mred <= self.mred_budget else 1,
                    max(0.0, mred - self.mred_budget), ppa)
        # Unconstrained (EDA) mode mirrors Arith-DAS exactly: rank purely by the
        # scalar objective so the exported "best" design matches the baseline.
        # （area-budget 运行的可行性/违规量/功耗三元排序已随该模式剪除。）
        return (0, 0.0, sample_info["objective"])

    @staticmethod
    def _adv_group_of(si):
        return (si.get("cell_type_info") or {}).get("cfg_group", 0)

    def _adv_group_stats(self, sample_info_list):
        """V6-R1：advantage 组内归一。不同 cell 配置的 objective 有系统性位差
        （配置好坏），全批归一会让"组间差"淹没"组内布线差"——PPO 只学布线，
        信号必须来自同配置样本之间。组内 <2 样本回退全批统计。
        返回 {group: (mean, std)}，含全批兜底键。"""
        import collections as _c
        objs = _c.defaultdict(list)
        for si in sample_info_list:
            objs[self._adv_group_of(si)].append(float(si["objective"]))
        allv = np.array([float(si["objective"]) for si in sample_info_list],
                        dtype=np.float64)
        fallback = (float(allv.mean()), float(allv.std()) + 1e-8)
        out = {}
        for gk, v in objs.items():
            a = np.array(v, dtype=np.float64)
            out[gk] = ((float(a.mean()), float(a.std()) + 1e-8)
                       if len(a) >= 2 else fallback)
        return out

    def get_ppo_loss(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
        sample_info_list: List[Dict],
    ):
        l = torch.tensor([0.0], device=self.device)
        # 点1：advantage 归一。把 A 从原始 -obj 改成 -(obj-mean)/(std+eps)，
        # 减均值给出"比本批平均好/差多少"的相对信号，除标准差把量纲压到 O(1)，
        # 使学习信号不再被 obj 绝对大小（误差尺度）主导。normalize_advantage=False 时为旧行为。
        if self.normalize_advantage and len(sample_info_list) < 2:
            # 单样本无相对信号（A≡0，梯度为零）——显式跳过并说明，而非静默零更新
            logging.info("[ppo] 本批仅 %d 个策略样本，normalize_advantage 无相对信号，"
                         "跳过 PPO 更新", len(sample_info_list))
            return l
        if self.normalize_advantage:
            _gstats = self._adv_group_stats(sample_info_list)
        for sample_info in sample_info_list:
            old_log_prob = sample_info["overall_log_prob"]
            new_log_prob = 0.0
            sample_id = 0
            mask_cache, Z_cache = self.get_cache(Z_mat_dict)
            for (s, c), Z_mat_slice in Z_mat_dict.items():
                Z = Z_cache[(s, c)]
                M = mask_cache[(s, c)]
                sum_src_indices = torch.tensor(
                    self.comp_graph.slice_indice_map[(s - 1, c)], device=self.device
                )
                for local_src_idx, src_idx in enumerate(sum_src_indices):
                    sample = sample_info["connection"][sample_id][3]
                    sample_id += 1
                    logits = Z[local_src_idx, :].masked_fill(~M[local_src_idx, :], -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(
                        torch.tensor([sample["sample"]], device=self.device)
                    )
                    new_log_prob += log_prob
                    M[:, sample["sample"]] = False

                if c > 0:
                    carry_sources = self.comp_graph.get_slice_carry_sources(s, c)
                    for local_src_idx, (_src_idx, _src_output) in enumerate(carry_sources):
                        sample = sample_info["connection"][sample_id][3]
                        sample_id += 1
                        logits = Z[local_src_idx + len(sum_src_indices), :].masked_fill(
                            ~M[local_src_idx + len(sum_src_indices), :], -1e9
                        )
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(
                            torch.tensor([sample["sample"]], device=self.device)
                        )
                        new_log_prob += log_prob
                        M[:, sample["sample"]] = False

            # Phase B：重算类型采样 log_prob。旧模式为逐 slot 独立 categorical；
            # cardinality 模式为 P(K) + P(slot sequence | K) + P(cell | slot)。
            if self.use_approx_types:
                type_log_prob = self._cell_type_log_prob(sample_info)
                if type_log_prob is not None:
                    new_log_prob = new_log_prob + type_log_prob

            if self.normalize_advantage:
                _m, _s = _gstats[self._adv_group_of(sample_info)]
                A = -(sample_info["objective"] - _m) / _s
            else:
                A = -sample_info["objective"]
            ratio = torch.exp(new_log_prob - old_log_prob)
            loss_1 = A * ratio
            loss_2 = A * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss_1, loss_2)
            l += -loss
        l /= len(sample_info_list)
        return l

    def update_found_best_info(self, sample_info_list):
        if getattr(self, "pareto_v5", False):
            # v5：标量 rank 不再决定存活，全部样本走档案支配准入
            self._v5_admit_samples(sample_info_list)
            return
        for sample_info in sample_info_list:
            is_better = self.found_best_info["connection"] is None
            if not is_better:
                is_better = self._candidate_rank(sample_info) < self._candidate_rank(
                    self.found_best_info
                )
            if is_better:
                self.found_best_info["objective"] = sample_info["objective"]
                self.found_best_info["connection"] = sample_info["connection"]
                self.found_best_info["ct"] = copy.deepcopy(self.state)
                self.found_best_info["assignment"] = copy.deepcopy(self.assignment)
                self.found_best_info["simulated_result"] = sample_info["result"]
                self.found_best_info["cell_types"] = copy.deepcopy(
                    sample_info.get("cell_types")
                )
                self.found_best_info["cell_type_info"] = copy.deepcopy(
                    sample_info.get("cell_type_info")
                )
                # 可审计：记录该最优点的误差闸门来源（verilator 实测 / analytic 回退）
                me = sample_info.get("measured_error")
                self.found_best_info["measured_error"] = copy.deepcopy(me)
                self.found_best_info["error_source"] = (
                    me["source"] if me else
                    ("analytic_fallback" if self.error_gate == "verilator" else "analytic")
                )
                self.found_best_info.update(self._best_info_metadata())

    def _effective_med(self, d):
        """与 get_objective 同口径的 MED（供日志/上报）：verilator 实测优先，失败回退
        解析；无误差源（无近似 cell 且无截断）= 0。d 可为单个 sample_info 或 found_best_info。"""
        me = d.get("measured_error")
        if (self.error_gate == "verilator"
                and me is not None and me.get("med") is not None):
            return float(me["med"])
        if (self.use_approx_types and d.get("cell_types")) or self.trunc_cols > 0:
            med, _b, _n, _w = self._analytic_error(d.get("cell_types") or {})
            return float(med)
        return 0.0

    def log_episode(self, episode_idx, info):
        if self.found_best_info.get("simulated_result") is None:
            # r2 审查 #3 修复：v5 档案为空（首集全批测量失败）时代表回退到初始
            # 占位（simulated_result=None），下面 found_best 段会迭代 None 崩溃。
            # 只跳日志，不跳训练。
            logging.warning("[log] found_best 尚无有效设计（档案空？），跳过本轮 tb 日志")
            return
        self.tb_logger.add_scalar("objective", info["objective"], episode_idx)
        self.tb_logger.add_scalar(
            "weight/disc_loss_weight", self.disc_loss_weight, episode_idx
        )
        self.tb_logger.add_scalar(
            "weight/rule_loss_weight", self.rule_loss_weight, episode_idx
        )
        for epoch_loss_info in info["epoch_loss"]:
            for loss_key in epoch_loss_info.keys():
                self.tb_logger.add_scalar(
                    f"epoch_loss/{loss_key}",
                    epoch_loss_info[loss_key],
                    self.total_epoch_num,
                )
            self.total_epoch_num += 1

        for loss_key in info["epoch_loss"][0].keys():
            loss_value = 0.0
            for epoch_loss_info in info["epoch_loss"]:
                loss_value += epoch_loss_info[loss_key]
            loss_value /= len(info["epoch_loss"])
            self.tb_logger.add_scalar(
                f"episode_loss/{loss_key}", loss_value, episode_idx
            )

        for ppa_key in ["area", "delay", "power"]:
            ppa_value = 0.0
            for simulated_result in info["simulated_result"]:
                ppa_value += simulated_result[ppa_key]
            ppa_value /= len(info["simulated_result"])
            self.tb_logger.add_scalar(f"ppa/{ppa_key}", ppa_value, episode_idx)
        self.tb_logger.add_scalar("ppa/med", info.get("med", 0.0), episode_idx)
        self.tb_logger.add_scalar("lr", self.scheduler.get_last_lr()[0], episode_idx)

        self.tb_logger.add_scalar(
            "found_best/objective",
            self.found_best_info["objective"],
            episode_idx,
        )
        for ppa_key in ["area", "delay", "power"]:
            ppa_value = 0.0
            for simulated_result in self.found_best_info["simulated_result"]:
                ppa_value += simulated_result[ppa_key]
            ppa_value /= len(self.found_best_info["simulated_result"])
            self.tb_logger.add_scalar(f"found_best/{ppa_key}", ppa_value, episode_idx)
        best_med = self._effective_med(self.found_best_info)
        self.tb_logger.add_scalar("found_best/med", best_med, episode_idx)
        self.tb_logger.add_scalar("lr", self.scheduler.get_last_lr()[0], episode_idx)

        # ── real-time console progress ──────────────────────────────────────
        cur  = self._summarize_result(info["simulated_result"])
        best = self._best_info_metadata()
        # v5：area_budget 恒 None、power_source 恒 eda → 原 vio_str/proxy_str 恒为空串，已剪除
        best_mred = (self.found_best_info.get("measured_error") or {}).get("mred")
        cur_mred = info.get("mred")
        extra = ""
        if cur_mred is not None:
            extra += f"  mred={cur_mred * 100:.4f}%"
        if "n_over" in info:
            extra += f"  over_budget={info['n_over']}/{info['n_total']}"
        best_str = f"  best_mred={best_mred * 100:.4f}%" if best_mred is not None else ""
        if info.get("n_approx") is not None:
            best_str += f"  n_approx={info['n_approx']}"
        if info.get("n_ct42") is not None:
            best_str += f"  n_ct42={info['n_ct42']}"
        logging.info(
            "[ep %4d/%d]  obj=%.6f  area=%.1f  delay=%.4fns"
            "  pwr=%.4fmW[%s]  med=%.1f%s"
            "  || best: obj=%.6f  area=%.1f  pwr=%.4fmW  med=%.1f%s",
            episode_idx, self.num_episodes, info["objective"],
            cur["area"], cur["delay"], cur["power"] * 1000,
            self.power_source, info.get("med", float("nan")), extra,
            best["objective"], best["area"], best["power"] * 1000, best_med, best_str,
        )

    def run_episode(self, episode_idx):
        logging.info(f"Episode {episode_idx} start")
        if getattr(self, "pareto_v5", False):
            self._v5_begin_episode(episode_idx)

        logging.info(f"sampling")
        self.reset()
        sample_info_list = self.get_samples()
        if not sample_info_list:
            # P0(codex)：整批远端 DC 都失败 → 跳过本 episode（不更新策略/best），保持 LR schedule 对齐。
            logging.warning(f"Episode {episode_idx}: 全批远端 DC 失败, 跳过本轮更新")
            self.scheduler.step()
            return
        self.update_found_best_info(sample_info_list)
        ppo_sample_info_list = [
            item for item in sample_info_list if not item.get("baseline_only")
        ]

        min_idx = np.argmin([item["objective"] for item in sample_info_list])
        info = {}
        info["epoch_loss"] = []
        info["objective"] = sample_info_list[min_idx]["objective"]
        info["simulated_result"] = sample_info_list[min_idx]["result"]
        info["med"] = self._effective_med(sample_info_list[min_idx])
        # 训练监控：当前候选 mred、best 设计的近似压缩器数量、本轮超 budget 的样本数。
        info["mred"] = (sample_info_list[min_idx].get("measured_error") or {}).get("mred")
        info["n_approx"] = sum(
            1 for tk in (self.found_best_info.get("cell_types") or {}).values()
            if tk and tk[1] != 0
        )
        info["n_ct42"] = int(np.sum((self.found_best_info.get("ct") or {}).get("ct42", [])))
        _bud = getattr(self, "mred_budget", None)
        if getattr(self, "error_metric", "med") == "mred" and _bud:
            _ms = [(s.get("measured_error") or {}).get("mred") for s in sample_info_list]
            info["n_over"] = sum(1 for m in _ms if m is not None and m > _bud)
            info["n_total"] = len(sample_info_list)

        if self.outer_cell_search:
            # 变异配置与 all-exact 配对候选各以其最优 objective 入池（两个变体都保留；
            # 好的 cell 摆放随状态继承，不再每轮丢失）
            non_base = [s for s in sample_info_list if not s.get("baseline_only")]
            if non_base:
                self.update_pool(min(s["objective"] for s in non_base), self.state)
            base = [s for s in sample_info_list
                    if s.get("candidate_kind") == "all_exact"]
            if base and (self.state.get("cells") or []):
                exact_state = copy.deepcopy(self.state)
                exact_state["cells"] = []
                self.update_pool(min(s["objective"] for s in base), exact_state)
        else:
            self.update_pool(sample_info_list[min_idx]["objective"], self.state)

        logging.info(f"updating")
        for epoch_idx in range(self.num_epochs):
            Z_mat_dict = self.get_Z_mat()

            loss_info = {}
            l = torch.tensor([0.0], device=self.device)
            if self.use_ppo_loss and ppo_sample_info_list:
                l_ppo = self.get_ppo_loss(Z_mat_dict, ppo_sample_info_list)
                l += self.ppo_loss_weight * l_ppo
                loss_info["l_ppo"] = l_ppo.item()
            elif self.use_ppo_loss:
                loss_info["l_ppo"] = 0.0
            if self.use_disc_loss:
                l_discrete = self.get_discrete_loss(Z_mat_dict)
                l += self.disc_loss_weight * l_discrete
                loss_info["l_discrete"] = l_discrete.item()
                self.disc_loss_weight += self.disc_loss_weight_incr
            if self.use_rule_loss:
                l_rule = self.get_rule_loss(Z_mat_dict)
                l += self.rule_loss_weight * l_rule
                loss_info["l_rule"] = l_rule.item()
                self.rule_loss_weight += self.rule_loss_weight_incr
            # （原版此处还有 use_delay_loss/use_error_loss 两个可微 surrogate 分支：
            #   v5 下恒关，已随 get_delay_loss/get_error_loss 一并剪除。）

            self.optim.zero_grad()
            l.backward()
            # 双评审发现 #7 修复：裁剪盖全部被训参数（原来只裁 GCN，类型头/
            # cardinality logits 无界——Phase B 以来的老洞）
            torch.nn.utils.clip_grad_norm_(self._opt_params, self.max_grad_norm)
            self.optim.step()

            loss_info["l"] = l.item()
            info["epoch_loss"].append(loss_info)
        if episode_idx % self.log_freq == 0:
            self.log_episode(episode_idx, info)
        self.scheduler.step()
