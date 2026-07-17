"""动作空间环境：reset/legalize/transition/action mask（压缩树布线的
合法动作生成与状态转移）。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
from typing import Dict
import random
import copy
import logging


import numpy as np

from utils import (
    get_initial_partial_product,
    CompressorTree,
)

from .compressor_graph import CompressorGraph


class EnvironmentMixin:
    """动作空间环境：reset/legalize/transition/action mask。"""

    def _start_reset(self):
        self.initial_pp = get_initial_partial_product(
            self.bit_width, self.encode_type
        ).astype(int)
        if self.trunc_cols > 0 and not self._trunc_bits:
            self._setup_truncation()
        if self.ct_arch == "wallace":
            ct = CompressorTree.wallace(self.initial_pp)
        elif self.ct_arch == "dadda":
            ct = CompressorTree.dadda(self.initial_pp)
        else:
            raise ValueError("Invalid compressor tree architecture")
        init_objective = self.get_objective(
            [
                {
                    "delay": self.delay_scale,
                    "area": self.area_scale,
                    "power": self.power_scale,
                }
            ]
        )
        init_state = {
            "ct32": ct.ct32.astype(int),
            "ct22": ct.ct22.astype(int),
        }
        if self.use_ct42:
            init_state["ct42"] = np.zeros_like(ct.ct32, dtype=int)
        if self.outer_cell_search:
            init_state["cells"] = []
        n_warm = self._seed_pool_from_best_info(ct.ct32.shape)
        if n_warm == 0:
            self.pool.add(init_objective, init_state)

    def reset(self):
        v5 = getattr(self, "pareto_v5", False)
        # V6-R1：组状态每集清零——种子集/求解器/greedy 清 cells 路径不掷骰子，
        # 不清会把上一集的多配置带进本集
        self._episode_cell_configs = None
        self._episode_ct_groups = None
        v5_state = self._v5_sample_parent_state() if v5 else None
        if v5_state is not None:
            self.state = v5_state          # 已 deepcopy（种子覆写或档案亲代）
        else:
            pool_list = self.pool.get_pool()
            logging.info(f"pool size: {len(pool_list)}")
            if len(pool_list) == 0:
                raise ValueError("Pool is empty, cannot reset environment.")
            sampled_item = random.choice(pool_list)
            random_objective, random_state = sampled_item
            self.state = copy.deepcopy(random_state)
        if v5:
            # M0 k 线程化：k 是设计属性（state["k"]），激活其截断档（常数/资格窗/floor）
            self.state.setdefault("k", int(self.trunc_cols))
            self._activate_trunc_profile(self.state["k"])
        if v5 and self._v5_seeding:
            # 初始种群个体（不同 k 截断的 Dadda 树）：原样评估不变异——
            # 这是论文可复现基线点本体，第 0 遍课程 = 逐 k 建阶梯
            if self.outer_cell_search:
                self.state.setdefault("cells", [])
        elif self.outer_cell_search:
            self.state.setdefault("cells", [])
            self._outer_mutate()
            if self.outer_cell_solver == "greedy":
                # cells 由 get_samples 在 sample-0 布线上 greedy 求解填充；
                # 结构变异照常（外环仍搜结构），cell 维度交给求解器。
                self.state["cells"] = []
                self._episode_cell_configs = None   # 骰子结果已作废，别留多配置
        else:
            action_mask = self.get_action_mask()
            action = random.choice(np.where(action_mask == 1)[0])
            self.transition(action)

        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct = CompressorTree(pp, self.state["ct32"], self.state["ct22"], self.state.get("ct42"))
        self.assignment = ct.compressor_assignment_fused()
        self.comp_graph = CompressorGraph(
            pp, self.assignment, num_node_types=self.num_node_types
        )
        if self.outer_cell_search:
            # slot 坐标 → 本 episode 图的 node_idx（发射/objective/best 复用现有路径）
            unmapped = self._refresh_episode_cell_groups()
            if unmapped:
                logging.warning("[outer] %d 个 cell 未映射到图节点（prune 应已保证为 0）",
                                unmapped)

    def _refresh_episode_cell_groups(self):
        """V6-R1：为本集每个 cell 配置建 slot→node 映射（_episode_ct_groups），
        组0 同步到 _episode_cell_types（TT 门/求解器等单配置遗留接口默认读组0）。
        无多配置（种子集/求解器/旧行为）→ 单组=state["cells"]，语义与旧版一致。
        返回未映射条数。"""
        cfgs = self._episode_cell_configs
        if not cfgs:
            cfgs = [[list(e) for e in (self.state.get("cells") or [])]]
            self._episode_cell_configs = cfgs
        self._episode_ct_groups = []
        unmapped = 0
        saved = self.state.get("cells")
        for cfg in cfgs:
            self.state["cells"] = cfg
            unmapped += self._refresh_episode_cell_types()
            self._episode_ct_groups.append(dict(self._episode_cell_types))
        self.state["cells"] = saved
        self._episode_cell_types = dict(self._episode_ct_groups[0])
        return unmapped

    def _ct42_effective_pp(self, ct42):
        """把固定的 ct42 背景折进等效初始列高：每列消耗 3、给下一列 2。
        0..3 经典动作的合法性/合法化在等效列高上判定即与无 ct42 时同构。"""
        eff = np.asarray(copy.deepcopy(self.initial_pp), dtype=int)
        ct42 = np.asarray(ct42, dtype=int)
        eff -= 3 * ct42
        eff[1:] += 2 * ct42[:-1]
        return eff

    def legalize_ct_architecture(self, ct32: np.ndarray, ct22: np.ndarray, initial_pp=None):
        if initial_pp is None:
            initial_pp = copy.deepcopy(self.initial_pp)
        else:
            initial_pp = np.asarray(copy.deepcopy(initial_pp), dtype=int)
        assert len(ct32) == len(initial_pp) and len(ct22) == len(initial_pp)
        ct32 = copy.deepcopy(ct32).astype(int)
        ct22 = copy.deepcopy(ct22).astype(int)
        for column_index in range(0, len(initial_pp)):
            ct32[column_index] = max(0, ct32[column_index])
            ct22[column_index] = max(0, ct22[column_index])
            if column_index == 0:
                remain_pp = (
                    initial_pp[column_index]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            else:
                remain_pp = (
                    initial_pp[column_index]
                    + ct32[column_index - 1]
                    + ct22[column_index - 1]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            if remain_pp < 1:
                if ct22[column_index] + remain_pp >= 1:
                    ct22[column_index] += remain_pp - 1
                else:
                    remain_pp += ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] -= (2 - remain_pp) // 2
                    else:
                        ct32[column_index] -= (1 - remain_pp) // 2
            elif remain_pp > 2:
                if remain_pp - ct22[column_index] <= 2:
                    ct22[column_index] -= remain_pp - 2
                    ct32[column_index] += remain_pp - 2
                else:
                    ct32[column_index] += ct22[column_index]
                    remain_pp -= ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] += (remain_pp - 2) / 2
                    else:
                        ct32[column_index] += (remain_pp - 1) / 2

        remain_pp = copy.deepcopy(initial_pp)
        remain_pp[0] = initial_pp[0] - 2 * ct32[0] - ct22[0]
        for column_index in range(1, len(initial_pp)):
            remain_pp[column_index] = (
                initial_pp[column_index]
                + ct32[column_index - 1]
                + ct22[column_index - 1]
                - 2 * ct32[column_index]
                - ct22[column_index]
            )
        remain_pp = np.asarray(remain_pp)
        return ct32, ct22

    def transition(self, action: int) -> np.ndarray:
        if self.use_ct42:
            # 6 动作/列：0..3 = 经典 ±HA / FA<->HA（ct42 折进等效列高后合法化），
            #            4 = promote42 (FA+HA -> CT42)，5 = demote42 (CT42 -> FA+HA)。
            # 动作空间是无 ct42 搜索的严格超集，纯截断解不再不可达。
            action_type_num = 6
            action_column = action // action_type_num
            action_type = action % action_type_num
            ct32 = copy.deepcopy(self.state["ct32"])
            ct22 = copy.deepcopy(self.state["ct22"])
            ct42 = copy.deepcopy(
                self.state.get("ct42", np.zeros_like(ct32, dtype=int))
            )
            if action_type == 4:
                if ct32[action_column] <= 0 or ct22[action_column] <= 0:
                    raise ValueError(f"illegal CT42 promote action at column {action_column}")
                ct32[action_column] -= 1
                ct22[action_column] -= 1
                ct42[action_column] += 1
            elif action_type == 5:
                if ct42[action_column] <= 0:
                    raise ValueError(f"illegal CT42 demote action at column {action_column}")
                ct32[action_column] += 1
                ct22[action_column] += 1
                ct42[action_column] -= 1
            elif action_type in (0, 1, 2, 3):
                if action_type == 0:
                    ct22[action_column] += 1
                elif action_type == 1:
                    ct22[action_column] -= 1
                elif action_type == 2:
                    ct22[action_column] += 1
                    ct32[action_column] -= 1
                elif action_type == 3:
                    ct22[action_column] -= 1
                    ct32[action_column] += 1
                eff_pp = self._ct42_effective_pp(ct42)
                ct32, ct22 = self.legalize_ct_architecture(ct32, ct22, initial_pp=eff_pp)
            else:
                raise NotImplementedError
            self.state["ct32"] = np.asarray(ct32).astype(int)
            self.state["ct22"] = np.asarray(ct22).astype(int)
            self.state["ct42"] = ct42.astype(int)
            return self.state

        action_column = action // 4
        action_type = action % 4
        ct_32 = copy.deepcopy(self.state["ct32"])
        ct_22 = copy.deepcopy(self.state["ct22"])

        if action_type == 0:
            ct_22[action_column] += 1
        elif action_type == 1:
            ct_22[action_column] -= 1
        elif action_type == 2:
            ct_22[action_column] += 1
            ct_32[action_column] -= 1
        elif action_type == 3:
            ct_22[action_column] -= 1
            ct_32[action_column] += 1
        else:
            raise NotImplementedError

        legalized_ct32, legalized_ct22 = self.legalize_ct_architecture(ct_32, ct_22)
        self.state["ct32"] = legalized_ct32
        self.state["ct22"] = legalized_ct22

    def get_action_mask(self):
        if self.use_ct42:
            # 6 动作/列：0..3 经典 ±HA / FA<->HA（在 ct42 等效列高上判定），
            # 4 promote42 / 5 demote42（列高不变；末列禁用，两个 carry 无处去）。
            action_type_num = 6
            ct32 = self.state["ct32"]
            ct22 = self.state["ct22"]
            ct42 = self.state.get("ct42", np.zeros_like(ct32, dtype=int))
            mask = np.zeros([action_type_num * len(self.initial_pp)])
            for column_index in range(0, len(self.initial_pp) - 1):
                if ct32[column_index] > 0 and ct22[column_index] > 0:
                    mask[column_index * action_type_num + 4] = 1
                if ct42[column_index] > 0:
                    mask[column_index * action_type_num + 5] = 1
            eff_pp = self._ct42_effective_pp(ct42)
            mask = self._fill_classic_action_mask(eff_pp, mask, action_type_num)
            if not np.any(mask):
                raise ValueError("use_ct42=True but no legal action exists")
            return mask != 0

        action_type_num = 4
        initial_pp = self.initial_pp
        mask = np.zeros([action_type_num * len(initial_pp)])
        mask = self._fill_classic_action_mask(initial_pp, mask, action_type_num)
        mask = mask != 0
        return mask

    def _fill_classic_action_mask(self, initial_pp, mask, action_type_num):
        """0..3 经典动作（+HA/-HA/FA->HA/HA->FA）的合法性判定，写入并返回 mask。
        use_ct42 时 initial_pp 传 ct42 等效列高即可复用同一判定。"""
        ct_32 = self.state["ct32"]
        ct_22 = self.state["ct22"]

        remain_pp = copy.deepcopy(initial_pp)
        for column_index in range(len(remain_pp)):
            if column_index > 0:
                remain_pp[column_index] += (
                    ct_32[column_index - 1] + ct_22[column_index - 1]
                )
            remain_pp[column_index] += -2 * ct_32[column_index] - ct_22[column_index]

        legal_act = []
        for column_index in range(2, len(initial_pp)):
            if remain_pp[column_index] == 2:
                legal_act.append((column_index, 0))
                if ct_22[column_index] >= 1:
                    legal_act.append((column_index, 3))
            if remain_pp[column_index] == 1:
                if ct_32[column_index] >= 1:
                    legal_act.append((column_index, 2))
                if ct_22[column_index] >= 1:
                    legal_act.append((column_index, 1))

        for act_col, action in legal_act:
            pp = copy.deepcopy(remain_pp)
            ct_32 = copy.deepcopy(self.state["ct32"])
            ct_22 = copy.deepcopy(self.state["ct22"])

            if action == 0:
                ct_22[act_col] = ct_22[act_col] + 1
                pp[act_col] = pp[act_col] - 1
                if act_col + 1 < len(pp):
                    pp[act_col + 1] = pp[act_col + 1] + 1
            elif action == 1:
                ct_22[act_col] = ct_22[act_col] - 1
                pp[act_col] = pp[act_col] + 1
                if act_col + 1 < len(pp):
                    pp[act_col + 1] = pp[act_col + 1] - 1
            elif action == 2:
                ct_22[act_col] = ct_22[act_col] + 1
                ct_32[act_col] = ct_32[act_col] - 1
                pp[act_col] = pp[act_col] + 1
            elif action == 3:
                ct_22[act_col] = ct_22[act_col] - 1
                ct_32[act_col] = ct_32[act_col] + 1
                pp[act_col] = pp[act_col] - 1

            for i in range(act_col + 1, len(pp) + 1):
                if i == len(pp):
                    mask[act_col * action_type_num + action] = 1
                    break
                elif pp[i] == 1 or pp[i] == 2:
                    mask[act_col * action_type_num + action] = 1
                    break
                elif pp[i] == 3:
                    ct_32[i] = ct_32[i] + 1
                    if i + 1 < len(pp):
                        pp[i + 1] = pp[i + 1] + 1
                    pp[i] = 1
                elif pp[i] == 0:
                    if ct_22[i] >= 1:
                        ct_22[i] = ct_22[i] - 1
                        if i + 1 < len(pp):
                            pp[i + 1] = pp[i + 1] - 1
                        pp[i] = 1
                    else:
                        ct_32[i] = ct_32[i] - 1
                        if i + 1 < len(pp):
                            pp[i + 1] = pp[i + 1] - 1
                        pp[i] = 2
        return mask

    def update_pool(
        self,
        objective: float,
        state: Dict[str, np.ndarray],
    ):
        self.pool.add(objective, state)
