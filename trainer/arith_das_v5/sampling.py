"""内环采样：GCN 前向 + 按 action mask 逐步采样压缩树布线，
组装 episode 样本（get_samples 是内环主入口）。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
from typing import Dict, Tuple
import copy
import logging

import torch

import multiprocessing

from utils import (
    CompressorTree,
    Mul,
)


class SamplingMixin:
    """内环采样：GCN 前向 + 逐步布线采样，组装 episode 样本。"""

    def get_cache(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ):
        mask_cache: Dict[Tuple, torch.Tensor] = {}
        Z_cache: Dict[Tuple, torch.Tensor] = {}
        stage_num, col_num = self.comp_graph.stage_num, self.comp_graph.col_num
        for s in range(stage_num + 1):
            for c in range(col_num):
                sum_mask = self.comp_graph.get_slice_sum_mask(s, c).to(self.device)
                masks = [sum_mask[p, :, :] for p in range(self.comp_graph.port_num)]
                if c > 0:
                    carry_mask = self.comp_graph.get_slice_carry_mask(s, c).to(
                        self.device
                    )
                    masks = [
                        torch.cat((sum_mask[p, :, :], carry_mask[p, :, :]), dim=0)
                        for p in range(self.comp_graph.port_num)
                    ]
                M = torch.cat(masks, dim=1)
                mask_cache[(s, c)] = M
        for (s, c), Z_mat_slice in Z_mat_dict.items():
            keys = ["a", "b", "c", "d"][: self.comp_graph.port_num]
            z_ports = []
            for key in keys:
                z = Z_mat_slice[f"s{key}"]
                if c > 0:
                    z = torch.cat([z, Z_mat_slice[f"c{key}"]], dim=0)
                z_ports.append(z)
            Z = torch.cat(z_ports, dim=1)
            Z_cache[(s, c)] = Z
        return mask_cache, Z_cache

    def sample_from_logits(
        self,
        Z_mat_dict: Dict[Tuple, torch.Tensor],
    ):
        samples_connection = []
        overall_log_prob = 0.0
        mask_cache, Z_cache = self.get_cache(Z_mat_dict)

        for (s, c), Z_mat_slice in Z_mat_dict.items():
            Z = Z_cache[(s, c)]
            M = mask_cache[(s, c)]
            sum_src_indices = self.comp_graph.slice_indice_map[(s - 1, c)]
            dst_indices = self.comp_graph.slice_indice_map[(s, c)]
            for local_src_idx, src_idx in enumerate(sum_src_indices):
                logits = Z[local_src_idx, :].masked_fill(~M[local_src_idx, :], -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                sample = dist.sample()
                log_prob = dist.log_prob(sample)
                overall_log_prob += log_prob.item()

                local_dst_idx = sample.item() % len(dst_indices)
                dst_connec_type = sample.item() // len(dst_indices)
                dst_idx = dst_indices[local_dst_idx]

                dst_info = self.comp_graph.vertex_list[dst_idx]
                src_info = self.comp_graph.vertex_list[src_idx]

                assert dst_info[0] == src_info[0] + 1
                assert dst_info[1] == src_info[1]

                M[:, sample.item()] = False

                samples_connection.append(
                    (
                        src_idx,
                        dst_idx,
                        dst_connec_type,
                        {
                            "log_prob": log_prob.item(),
                            "local_src_idx": local_src_idx,
                            "local_dst_idx": local_dst_idx,
                            "sample": sample.item(),
                            "slice": (s, c),
                            "src_output": "sum",
                        },
                    )
                )
            if c > 0:
                carry_sources = self.comp_graph.get_slice_carry_sources(s, c)
                for local_src_idx, (src_idx, src_output) in enumerate(carry_sources):
                    src_info = self.comp_graph.vertex_list[src_idx]
                    logits = Z[local_src_idx + len(sum_src_indices), :].masked_fill(
                        ~M[local_src_idx + len(sum_src_indices), :], -1e9
                    )
                    dist = torch.distributions.Categorical(logits=logits)
                    sample = dist.sample()
                    log_prob = dist.log_prob(sample)
                    overall_log_prob += log_prob.item()
                    local_dst_idx = sample.item() % len(dst_indices)
                    dst_connec_type = sample.item() // len(dst_indices)
                    dst_idx = dst_indices[local_dst_idx]
                    dst_info = self.comp_graph.vertex_list[dst_idx]
                    assert dst_info[0] == src_info[0] + 1
                    assert dst_info[1] == src_info[1] + 1
                    M[:, sample.item()] = False
                    samples_connection.append(
                        (
                            src_idx,
                            dst_idx,
                            dst_connec_type,
                            {
                                "log_prob": log_prob.item(),
                                "local_src_idx": local_src_idx,
                                "local_dst_idx": local_dst_idx,
                                "sample": sample.item(),
                                "slice": (s, c),
                                "src_output": src_output,
                            },
                        )
                    )

        return samples_connection, overall_log_prob

    def get_samples(self):
        with torch.no_grad():
            sample_info = []
            Z_mat_dict = self.get_Z_mat()
            pre_samples = None
            if self._cell_solver_active():
                # 求解器模式：先采完整集布线,greedy 包对全部布线做鲁棒修复后再发射
                # （只按 sample-0 解会跨布线偏高 10%+ → 7/9 越线报废,07-10 实测）
                pre_samples = [
                    self.sample_from_logits(Z_mat_dict)
                    for _ in range(self.num_samples)
                ]
                self._outer_greedy_solve_robust([c for c, _lp in pre_samples])
            for sample_idx in range(self.num_samples):
                if pre_samples is not None:
                    samples_connection, overall_log_prob = pre_samples[sample_idx]
                else:
                    samples_connection, overall_log_prob = self.sample_from_logits(
                        Z_mat_dict
                    )
                if self.outer_cell_search:
                    # 外环模式：cell 配置由外环决定、不进 log_prob（PPO 只学布线）。
                    # V6-R1：G>1 时样本按组均分（sample_idx % G），各组一个配置；
                    # 求解器模式恒单组（cells 由求解器在本函数内填充，读实时视图）
                    if self._cell_solver_active():
                        n_grp, g = 1, 0
                        type_choices = dict(self._episode_cell_types)
                    else:
                        groups = self._episode_ct_groups or [dict(self._episode_cell_types)]
                        n_grp = len(groups)
                        g = sample_idx % n_grp
                        type_choices = dict(groups[g])
                    cell_map = self._cell_map_from_types(type_choices)
                    type_sample_info = {"mode": "outer", "cfg_group": g}
                elif getattr(self, "pareto_v5", False) and self._v5_seeding:
                    # 双评审发现 #2 修复：非 outer 模式的 v5 种子集不做类型采样——
                    # 种子必须是纯截断 Dadda（可复现基线）；"outer" 模式标记复用
                    # PPO 侧"无类型 log_prob"处理
                    cell_map, type_choices, type_sample_info = {}, {}, {"mode": "outer"}
                else:
                    # Phase B：采样每个压缩器槽的 cell 类型（关时返回空，行为不变）
                    cell_map, type_choices, type_log_prob, type_sample_info = (
                        self.sample_cell_types()
                    )
                    overall_log_prob += type_log_prob
                assignment = self.emit_assignment(samples_connection, cell_map=cell_map)

                ct = CompressorTree(
                    self.initial_pp,
                    self.state["ct32"],
                    self.state["ct22"],
                    self.state.get("ct42"),
                )
                if self.trunc_cols > 0:           # ① 把截断信息挂到 ct，emit_pp_encoder 会读
                    ct.trunc_cols = self.trunc_cols
                    ct.trunc_bits = self._trunc_bits
                mul = Mul(self.bit_width, self.encode_type, ct)
                rtl_path = os.path.join(self.build_dir, f"MUL-{sample_idx}.v")
                mul.emit_verilog(
                    rtl_path,
                    assignment=assignment,
                    extra_modules_src=self._approx_modules_src(cell_map),
                )
                if (
                    self.outer_cell_search
                    and sample_idx < n_grp          # 每组首样本过一次门（旧行为=组0）
                    and type_choices
                    and not self._cell_solver_active()
                    and (getattr(self, "outer_tt_oracle", False)
                         or self._outer_gate_active())
                ):
                    # M2 预筛门（TT oracle 或 errgate）：超预算就地修剪并重发射本样本
                    # RTL；V6-R1 下逐组独立过门，修剪结果写回该组供同组后续样本沿用
                    type_choices, cell_map = self._outer_screen_group(
                        g, mul, samples_connection, rtl_path
                    )
                entry = {
                    "rtl_path": rtl_path,
                    "connection": samples_connection,
                    "overall_log_prob": overall_log_prob,
                    "cell_types": type_choices,
                    "cell_type_info": type_sample_info,
                }
                if self.outer_cell_search and not self._cell_solver_active():
                    # V6-R1：payload 保真用——admit 时该样本的 ct.cells 必须是
                    # 本组配置（过门后的），而非 state 里的组0
                    entry["outer_cells"] = copy.deepcopy(
                        self._episode_cell_configs[g])
                sample_info.append(entry)
                if (
                    self.inject_exact_candidate
                    and self.use_approx_types
                    and sample_idx == 0
                ):
                    exact_rtl_path = os.path.join(
                        self.build_dir, f"MUL-{self.num_samples}-exact.v"
                    )
                    mul.emit_verilog(
                        exact_rtl_path,
                        assignment=self.emit_assignment(samples_connection, cell_map={}),
                        extra_modules_src="",
                    )
                    sample_info.append(
                        {
                            "rtl_path": exact_rtl_path,
                            "connection": samples_connection,
                            "overall_log_prob": overall_log_prob,
                            "cell_types": {},
                            "cell_type_info": {"mode": "exact_baseline"},
                            "baseline_only": True,
                            "candidate_kind": "all_exact",
                        }
                    )

            # v5：synth 恒 "dc" 且固定单一 DC 周期（构造守卫保证）→ 每样本只跑 1 次 DC
            target_delay_list = [self.fixed_target_delay]
            params_list = [
                (
                    self.bit_width,
                    self.encode_type,
                    copy.deepcopy(ct),
                    sample["rtl_path"],
                    os.path.join(self.build_dir, f"worker_{i}_{target_delay_id}"),
                    target_delay,
                    i,
                    target_delay_id,
                    self.synth,
                    self.error_gate,
                    self.error_gate_vectors,
                )
                for i, sample in enumerate(sample_info)
                for target_delay_id, target_delay in enumerate(target_delay_list)
            ]
            logging.info(f"processings: {self.n_processing}")
            if self.n_processing == 1:
                results = [
                    self.parallel_simulate_worker(*param) for param in params_list
                ]
            else:
                with multiprocessing.Pool(self.n_processing) as pool:
                    results = pool.starmap(self.parallel_simulate_worker, params_list)
            processed_results = {}
            measured_errors = {}      # rid -> verilator 实测误差 dict（误差闸门，与 PPA 分开聚合）
            failed_ids = set()
            for result in results:
                rid = result["id"]
                if (
                    result.get("failed")
                    or not result.get("result")
                    or result["result"][0] is None
                ):
                    failed_ids.add(rid)
                    continue
                processed_results.setdefault(rid, []).append(result["result"][0])
                # verilator 失败 ≠ DC 失败：只在成功时记录，None 时上层回退解析（不丢样本）
                me = result.get("measured_error")
                if me is not None:
                    measured_errors.setdefault(rid, me)

            # P0(codex)：远端 DC 失败的样本直接踢出本批（不回退 ABC），避免量纲污染 PPO/best。
            kept_sample_info = []
            for i in range(len(sample_info)):
                if i not in processed_results:
                    continue
                result_list = self._apply_power_proxy_to_results(
                    processed_results[i],
                    sample_info[i]["connection"],
                )
                sample_info[i]["result"] = result_list
                sample_info[i]["measured_error"] = measured_errors.get(i)
                sample_info[i]["objective"] = self.get_objective(
                    result_list,
                    cell_types=sample_info[i].get("cell_types"),
                    measured_error=sample_info[i]["measured_error"],
                )
                kept_sample_info.append(sample_info[i])
            if failed_ids:
                logging.warning(
                    "[dc] %d/%d 样本远端 DC 失败已丢弃(不混 ABC); 本批保留 %d",
                    len(failed_ids), len(sample_info), len(kept_sample_info),
                )
            # 误差闸门健康度：verilator 失败回退解析会给该样本虚假优势（解析低估），
            # codex 建议监控回退率；过高(>50%)则本批 verilator 闸门失真，告警。
            if self.error_gate == "verilator" and kept_sample_info:
                n_fb = sum(1 for s in kept_sample_info if s.get("measured_error") is None)
                if n_fb:
                    lvl = logging.ERROR if n_fb * 2 > len(kept_sample_info) else logging.INFO
                    logging.log(lvl, "[errgate] verilator 回退解析 %d/%d 样本",
                                n_fb, len(kept_sample_info))
            if self.inject_exact_candidate:
                exact_kept = sum(
                    1 for s in kept_sample_info
                    if s.get("candidate_kind") == "all_exact"
                )
                logging.info("[exact] all-exact baseline candidates kept: %d", exact_kept)
        return kept_sample_info
