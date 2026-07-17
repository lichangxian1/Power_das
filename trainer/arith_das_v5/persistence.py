"""落盘与恢复：策略权重加载、实验目录保存、PPA 诊断、最优解导出。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
import copy
import json
import logging

import torch

from pygmo import hypervolume
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    CompressorTree,
    Mul,
    convert_to_serializable,
)

from .compressor_graph import CompressorGraph


class PersistenceMixin:
    """策略权重加载、实验保存、PPA 诊断与最优解导出。"""

    def _load_policy(self, path):
        """策略持久化 LOAD：path = save_iterNN 目录或 gcn.pth 文件。
        组件级 try/except：GCN 必载（失败告警回退随机初始化）；类型头/基数 logits
        仅当本 run 相应组件存在且形状匹配时载入（跨库表大小可能不同）。"""
        p = self._resolve_path(path)
        gcn_path = p if p.endswith(".pth") else os.path.join(p, "gcn.pth")
        heads_path = os.path.join(os.path.dirname(gcn_path), "type_heads.pth")
        try:
            state = torch.load(gcn_path, map_location=self.device)
            self.gcn.load_state_dict(state)
            logging.info(f"policy warm-start: GCN <- {gcn_path}")
        except Exception as e:  # noqa: BLE001
            logging.error(f"policy warm-start GCN 加载失败(回退随机初始化): {e}")
            return
        if not (self.use_approx_types and os.path.exists(heads_path)):
            return
        try:
            ts = torch.load(heads_path, map_location=self.device)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"policy warm-start type_heads 读取失败(仅载 GCN): {e}")
            return
        for name in ("type_head_32", "type_head_22", "type_head_42"):
            head = getattr(self, name, None)
            if head is None or name not in ts:
                continue
            try:
                head.load_state_dict(ts[name])
                logging.info(f"policy warm-start: {name} <- {heads_path}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"policy warm-start {name} 形状不合跳过: {e}")
        if (self.approx_cardinality_logits is not None
                and "approx_cardinality_logits" in ts
                and list(ts.get("approx_cardinality_choices", []))
                == list(self.approx_cardinality_choices)):
            with torch.no_grad():
                self.approx_cardinality_logits.copy_(
                    ts["approx_cardinality_logits"].to(self.device))
            logging.info("policy warm-start: approx_cardinality_logits 已载入")

    def save_experiment(self, episode_idx):
        logging.info(f"saving experiment at episode {episode_idx}")
        save_dir = os.path.join(self.log_dir, f"save_iter{episode_idx}")
        os.makedirs(save_dir, exist_ok=True)
        gcn_save_path = os.path.join(save_dir, "gcn.pth")
        torch.save(self.gcn.state_dict(), gcn_save_path)
        # Phase B：类型头不在 gcn 内，单独存，否则 resume/重载会丢失已学的类型策略
        if self.use_approx_types:
            type_state = {
                "type_head_32": self.type_head_32.state_dict(),
                "type_head_22": self.type_head_22.state_dict(),
            }
            if self.type_head_42 is not None:
                type_state["type_head_42"] = self.type_head_42.state_dict()
            if self.approx_cardinality_logits is not None:
                type_state["approx_cardinality_logits"] = (
                    self.approx_cardinality_logits.detach().cpu()
                )
                type_state["approx_cardinality_choices"] = (
                    self.approx_cardinality_choices
                )
            torch.save(type_state, os.path.join(save_dir, "type_heads.pth"))
        with open(os.path.join(save_dir, "best_info.json"), "w") as f:
            json.dump(
                self.found_best_info, f, indent=4, default=convert_to_serializable
            )
        if getattr(self, "pareto_v5", False):
            with open(os.path.join(save_dir, "front.json"), "w") as f:
                json.dump(self._v5_archive.snapshot(), f, indent=2,
                          default=convert_to_serializable)
            # 双评审发现 #4 修复：滚动全量档案（含 payload），崩溃后可收割/重播；
            # tmp+rename 原子写，单文件滚动不膨胀磁盘
            state_path = os.path.join(self.log_dir, "front_state.json")
            tmp = state_path + ".tmp"
            state_blob = {
                "episode": episode_idx,
                "seed_queue": list(getattr(self, "_v5_seed_queue", [])),
                "bins": {str(b): es for b, es in self._v5_archive.bins.items()},
            }
            if self._v5_bandit is not None:
                state_blob["bandit"] = self._v5_bandit.to_json()
            with open(tmp, "w") as f:
                json.dump(state_blob, f, default=convert_to_serializable)
            os.replace(tmp, state_path)
            if len(self._v5_archive) == 0:
                logging.info("[v5] 档案空，跳过 full-PPA 诊断")
                return
            # 代表设计可能来自任意 k：full-PPA 发射前激活其截断档
            self._activate_trunc_profile(
                int(self.found_best_info["ct"].get("k", self.trunc_cols))
            )

        if getattr(self, "pareto_v5", False):
            # r2 审查 #1 修复：以下 full-PPA/HV/画图全是诊断——checkpoint（gcn/heads/
            # best_info/front/front_state）已全部落盘，诊断失败只弃诊断，绝不许打崩
            # 32-40h 战役（DC 瞬时失败 × 每 20ep 一次 save = ~72 个曝险窗口）。
            try:
                self._save_full_ppa_diagnostics(save_dir, episode_idx)
            except Exception:
                logging.exception("[v5] full-PPA 诊断失败（已跳过，checkpoint 完好）")
            return

        self._save_full_ppa_diagnostics(save_dir, episode_idx)

    def _save_full_ppa_diagnostics(self, save_dir, episode_idx):
        """save 期诊断段（full-PPA + hypervolume + pareto 图）。从 save_experiment
        拆出以便 v5 整段异常隔离；非 v5 维持原语义（异常照旧上抛）。"""
        self.state = self.found_best_info["ct"]
        self.assignment = self.found_best_info["assignment"]
        pp = self.initial_pp
        self.comp_graph = CompressorGraph(
            pp, self.assignment, num_node_types=self.num_node_types
        )

        logging.info(f"testing full target delay at episode {episode_idx}")
        simulated_result = self.get_full_target_delay_result()
        pareto_points = self.get_full_target_delay_pareto(
            simulated_result, self.pareto_target
        )
        pareto_value_0 = [point[0] for point in pareto_points]
        pareto_value_1 = [point[1] for point in pareto_points]
        hv = hypervolume(pareto_points)
        try:
            # Unconstrained (EDA) mode mirrors Arith-DAS: fixed reference point.
            # （area_budget 约束模式的自适应参考点已随该模式一并剪除。）
            ref = list(self.reference_point)
            hv_value = hv.compute(ref)
            self.tb_logger.add_scalar("hv_value", hv_value, episode_idx)
        except Exception as e:
            logging.error(f"Error computing hypervolume: {e}")
            hv_value = None
        with open(os.path.join(save_dir, "pareto.json"), "w") as f:
            json.dump(
                {
                    "hv_value": hv_value,
                    "pareto_target": self.pareto_target,
                    "pareto_value_0": pareto_value_0,
                    "pareto_value_1": pareto_value_1,
                },
                f,
                indent=4,
            )

        fig = plt.figure()
        pareto_value_0 = np.array(pareto_value_0)
        pareto_value_1 = np.array(pareto_value_1)
        sorted_indices = np.argsort(pareto_value_0)
        pareto_value_0 = pareto_value_0[sorted_indices]
        pareto_value_1 = pareto_value_1[sorted_indices]
        simulated_value_0 = np.asarray(
            [item[self.pareto_target[0]] for item in simulated_result]
        )
        simulated_value_1 = np.asarray(
            [item[self.pareto_target[1]] for item in simulated_result]
        )
        plt.scatter(
            simulated_value_0, simulated_value_1, label="Simulated Result", alpha=0.5
        )
        plt.plot(pareto_value_0, pareto_value_1, "--o", label="Pareto Front")
        plt.xlabel(self.pareto_target[0])
        plt.ylabel(self.pareto_target[1])
        plt.legend()
        self.tb_logger.add_figure("pareto_front", fig, episode_idx)

    def _best_info_metadata(self, info=None):
        info = info if info is not None else self.found_best_info
        if info["simulated_result"] is None:
            return {}
        summary = self._summarize_result(info["simulated_result"])
        summary["objective"] = info["objective"]
        return summary

    def export_best_candidate(self, export_dir, info=None):
        info = info if info is not None else self.found_best_info
        if info["connection"] is None:
            raise ValueError("No best candidate has been found; run training first")

        os.makedirs(export_dir, exist_ok=True)
        old_state = self.state
        old_assignment = self.assignment
        old_comp_graph = self.comp_graph
        try:
            if getattr(self, "pareto_v5", False):
                # v5：条目可能来自任意 k，发射前激活其截断档
                self._activate_trunc_profile(
                    int((info.get("ct") or {}).get("k", self.trunc_cols))
                )
            self.state = copy.deepcopy(info["ct"])
            self.assignment = copy.deepcopy(info["assignment"])
            self.comp_graph = CompressorGraph(
                self.initial_pp,
                self.assignment,
                num_node_types=self.num_node_types,
            )
            # Phase B：从最优设计的 cell_types 复原近似 cell（comp_graph 同序，node_idx 一致）
            cell_types = info.get("cell_types") or {}
            cell_map = self._cell_map_from_types(cell_types)
            routing_assignment = self.emit_assignment(
                info["connection"], cell_map=cell_map
            )
            ct = CompressorTree(
                self.initial_pp,
                self.state["ct32"],
                self.state["ct22"],
                self.state.get("ct42"),
            )
            if self.trunc_cols > 0:               # ① 导出也带截断
                ct.trunc_cols = self.trunc_cols
                ct.trunc_bits = self._trunc_bits
            mul = Mul(self.bit_width, self.encode_type, ct)
            rtl_path = os.path.join(export_dir, "MUL.v")
            mul.emit_verilog(
                rtl_path,
                assignment=routing_assignment,
                extra_modules_src=self._approx_modules_src(cell_map),
            )

            best_info = {
                **self._best_info_metadata(info),
                "connection": info["connection"],
                "ct": info["ct"],
                "assignment": info["assignment"],
                "cell_types": cell_types,
                "cell_type_info": info.get("cell_type_info"),
                "approx_cells": {str(k): v for k, v in cell_map.items()},
                "routing_assignment": routing_assignment,
                "rtl_path": rtl_path,
                "simulated_result": info["simulated_result"],
                # 误差闸门审计：该最优点用的是 verilator 实测还是解析回退
                "error_source": info.get("error_source"),
                "measured_error": info.get("measured_error"),
            }
            with open(os.path.join(export_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=4, default=convert_to_serializable)
            return rtl_path
        finally:
            self.state = old_state
            self.assignment = old_assignment
            self.comp_graph = old_comp_graph
