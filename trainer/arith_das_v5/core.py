"""CompressorRouting 类本体：__init__（全部超参与状态定义）+ 通用小工具。

巨型类按主题拆成了 10 个 mixin 切面（见各模块 docstring 与包 README），
在这里多继承拼装回唯一的 CompressorRouting，行为与拆分前逐行一致。"""
import os
from typing import Dict
import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from utils import (
    BoundedParetoPool,
)

from .compressor_graph import CompressorGraph
from .networks import MultiChannelResGCN
from .cell_types import CellTypeMixin
from .outer_search import OuterSearchMixin
from .truncation import TruncationMixin
from .rtl_emit import RtlEmitMixin
from .simulate import SimulateMixin
from .sampling import SamplingMixin
from .training import TrainingMixin
from .pareto_front import ParetoFrontMixin
from .persistence import PersistenceMixin
from .environment import EnvironmentMixin


class CompressorRouting(CellTypeMixin, OuterSearchMixin, TruncationMixin, RtlEmitMixin, SimulateMixin, SamplingMixin, TrainingMixin, ParetoFrontMixin, PersistenceMixin, EnvironmentMixin):
    """ARITH-DAS v5 训练闭环：外环（k 截断档 + 近似 cell 配置的进化/求解）+
    内环（GCN 策略采样压缩树布线，PPO 更新），评估走远端 DC 直出 PPA、
    verilator 实测误差，样本按 (mred→分箱, area, power) 进 ParetoArchive
    非支配档案（enable_pareto_v5 开启）。

    与 trainer.arith_das.CompressorRouting 的 v5 行为逐行一致；区别仅在
    剪除了模块 docstring 所列的不可达遗留路径，并在构造时显式守卫。
    """

    def __init__(
        self,
        bit_width,
        encode_type,
        ct_arch,
        use_ppo_loss,
        ppo_loss_weight,
        use_delay_loss,
        delay_loss_weight,
        lse_gamma_val,
        use_rule_loss,
        rule_loss_weight,
        use_disc_loss,
        disc_loss_weight,
        num_episodes,
        num_samples,
        num_epochs,
        log_dir,
        build_dir,
        save_freq,
        log_freq,
        device,
        optim_name,
        optim_kwargs,
        scheduler_name,
        scheduler_kwargs,
        gcn_kwargs,
        delay_weight,
        area_weight,
        power_weight,
        delay_scale,
        area_scale,
        power_scale,
        clip_range,
        max_grad_norm,
        n_processing,
        reference_point,
        pareto_target,
        pool_size,
        rule_loss_wight_incr,
        disc_loss_weight_incr,
        use_power_proxy=False,
        power_proxy_ckpt=None,
        power_proxy_lib_path="library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib",
        power_proxy_fa_cell="FA1D0BWP12T40P140",
        power_proxy_ha_cell="HA1D0BWP12T40P140",
        power_proxy_output_scale=1e-3,
        fixed_target_delay=None,
        area_budget=None,
        area_violation_weight=2.0,
        delay_violation_weight=2.0,
        power_source=None,
        gomil_path=None,
        # 温启动：best_info.json 路径（str 或 list）。成功加载则替代默认 wallace/dadda
        # 种子进池，且同时种 found_best_info（本 run 报告的 best 单调不劣于温启动点）。
        # 要求与本 run 同 bit_width/encode_type/trunc_cols/objective 口径。
        init_pool_best_info=None,
        # 策略持久化 LOAD 侧（save 侧 = save_experiment 的 gcn.pth/type_heads.pth）：
        # 指向 save_iterNN 目录（或 gcn.pth 文件）。组件级加载，形状不合逐项跳过并告警。
        init_policy_from=None,
        synth="openroad",
        # v5 前沿回放颗粒度：每 N ep 把档案 snapshot 落到 logs/front_hist/（KB 级，
        # 独立于 save_freq 的全量存档；gen_v5_front_viewer.py 会扫）。0/None=关。
        front_dump_freq=5,
        # ===== 阶段3 Phase B：近似压缩器类型搜索（全部默认关，关时行为不变）=====
        use_approx_types=False,
        approx_lib_path="Appr_Comp/selected_compressors.json",
        approx_library_path="Appr_Comp/library.json",
        approx42_library_path="Appr_Comp/library42_pair32_func.json",
        approx42_rtl_path="Appr_Comp/rtl/comp42_lib.v",
        approx42_max_types=16,
        approx_max_col=6,
        # 近似 cell 只在截断边界上方的窗口 [trunc_cols, trunc_cols+window) 内可选（None=旧行为，
        # 用 approx_max_col 作上界）。cell 误差 ∝ wae·2^col，高列代价指数增长几乎永远不划算；
        # 把动作集中到边界附近的廉价低列，省掉无用探索、提高发现有益 cell 的概率。
        approx_col_window=None,
        # 误差 reward（约束式 A）。med/bias 用 LSB 绝对单位（跨位宽稳定、梯度 O(1)）；
        # NMED=med/maxprod 仅用于上报。budget/weight 都以 LSB 计。
        med_budget=None,
        med_violation_weight=0.1,
        bias_weight=0.0,
        # 误差项归一尺度（点2）：把 med/bias 的 LSB 绝对值除以 error_scale，使 err_term
        # 落到和 PPA(~O(1)) 同量级。默认 1.0 = 不归一（旧行为，向后兼容）。
        error_scale=1.0,
        # 误差作为普通目标项（和 area/power 同评估）：error_as_metric=True 时 get_objective
        # 用 error_weight*med/error_scale 线性计入目标（像 area_weight*area/area_scale），
        # 不再用 med_budget 铰链 max(0,med-budget)；此模式下 med_budget/med_violation_weight 忽略。
        error_as_metric=False,
        error_weight=0.0,
        # 类型头初始化偏向 exact(index0)：>0 时给 type_head bias[0] 设此正偏置 → 初始 P(exact)
        # ≈exp(b)/(exp(b)+N-1)（如 4.0→~0.9）。让策略从"近全 exact"起步、按需加近似 cell，
        # 避免冷启动 ~85% 节点随机近似导致前期 obj 爆高、PPA 梯度被淹。默认 0=旧行为。
        exact_init_bias=0.0,
        # 方案 B：先采样本设计总共启用多少个近似 cell，再采样具体 slot/cell。
        # 关闭时保持旧行为（每个 slot 独立采 exact/approx）。开启后能稳定覆盖 n_approx=1/2/4
        # 等极稀疏候选，避免低 MRED budget 下从 all-exact 直接跳到十几个 cell。
        approx_cardinality_sampler=False,
        approx_cardinality_choices=None,
        approx_cardinality_init_logits=None,
        # 保底候选：每轮额外评估一个同 routing、全 exact cell 的设计，只参与 best/日志，
        # 不进 PPO loss。用于避免类型采样把 found_best 拖到比纯截断同 routing 更差。
        inject_exact_candidate=False,
        # error_scale 跨-k 归一模式（仅 error_as_metric 用，解决"不同 k 的 MED 差几个数量级、
        # 固定 error_scale 没法用一个 error_weight 通吃"）：
        #   "fixed"  = 用传入的 error_scale 常数（旧行为）；
        #   "pow2k"  = error_scale=2^(trunc_cols-1)（闭式，med/scale 跨 k ~3.7×）；
        #   "sqrt2k" = error_scale=√k·2^(k-1)（闭式，更平，med/scale 跨 k ~1.3×；floor∝std(Δ)）；
        #   "floor"  = error_scale=截断 MED floor self._trunc_med（精确、各 k 归一后 floor 处=1）。
        # pow2k/floor 下 med/error_scale≈O(1) 对所有 k → 单一 error_weight 跨 k 行为一致。
        error_scale_mode="fixed",
        # 可微误差 surrogate（D2 开关）
        use_error_loss=False,
        error_loss_weight=0.0,
        bias_loss_weight=0.0,
        # ④ 尾部/WCE 约束（默认关）：wce_bound = Σ maxe·2^col（误差可加上界，LSB），
        # 控制最坏情况误差、治重尾(RMSE/MAE≫1)。budget/weight 同 med 口径（LSB, /error_scale）。
        # wce_budget=None 或两个 weight=0 时该项不生效（回归字节级一致）。
        wce_budget=None,
        wce_violation_weight=0.0,
        wce_loss_weight=0.0,
        # 误差闸门来源（codex 审过）：
        #   "analytic"  = 解析 proxy（三角不等式估计，实测系统性低估真实 MED 0–30%；默认，向后兼容）
        #   "verilator" = 每个候选用 verilator MC（circular-wrap 真实 MED）测 med/bias 当软罚闸门。
        # 只影响 get_objective 的离散打分（reward 闸门）；可微 error_loss 仍用解析（不可微分 verilator）。
        # WCE 始终用解析上界（MC 尾部不收敛、随 N 单调增长，不可信）。
        error_gate="analytic",
        error_gate_vectors=16_000_000,
        # Phase C ①：低列截断 + 学习校正（默认关）。trunc_cols=k → 最低 k 列的 PP 用常数
        # （校正常数 C=round(E[Δ])，拆成低列槽位的常数 1 位）驱动而非 a&b；压缩树/布线不变，
        # DC 常数传播删低列逻辑＝截断 PPA 收益。误差项 −E[Δ]+C 进 _analytic_error。
        trunc_cols=0,
        trunc_correct="bias",
        # P0(codex)：delay 作为约束而非奖励项。开启后 delay≤delay_target_ns 不奖励也不罚，
        # delay>target 才按 delay_violation_weight 罚 → 优化预算全给 area/power（释放 slack）。
        # 默认关=旧线性奖励 delay_weight·delay。
        delay_as_constraint=False,
        delay_target_ns=None,   # None → 用 fixed_target_delay（DC 时钟周期）当阈值
        # advantage 归一（点1）：A=-(obj-mean)/(std+eps)。默认 False = 旧行为 A=-obj。
        normalize_advantage=False,
        # Exact 4:2 compressor architecture primitive. Default off for byte-level
        # compatibility with existing FA/HA-only runs.
        use_ct42=False,
        # ===== 外环 cell 搜索（Appr_Comp/OUTER_CELL_SEARCH.md，默认关=行为不变）=====
        # cell 类型进外环状态（进化：解析提议变异 + 闭式可行性过滤 + resample-K），
        # 内环只采布线；每轮全部样本共用同一 cell 配置 → PPO 信用分配只含布线。
        outer_cell_search=False,
        outer_p_struct=0.4,          # 变异算子骰子：结构 4/6 动作
        outer_p_cell=0.4,            # add/remove/swap 一个 cell（解析提议）
        outer_p_resample=0.2,        # resample-K 大步（清空重摆 K 个）
        outer_proposal_retries=50,   # 可行性过滤重试上限（闭式，微秒级）
        outer_med_slack_scale=1.0,   # MRED 模式 MED 等效 slack 比例（<1 更保守）
        outer_w_area=1.0,            # 提议打分：面积节省项权重
        outer_w_err=1.0,             # 提议打分：误差代价项权重
        # 外环实测误差预筛门（默认关=行为不变）：外环 cell 配置在进 DC 前，先用
        # sample-0 布线发射的 RTL 做 verilator MC 实测（秒级 vs DC ~200s/样本）；
        # 超预算 → 贪心摘掉解析误差贡献 wae·2^col 最大的 cell（每步误差严格降）
        # 重发射重测，修复步数耗尽仍超则清空 cells 保底（floor 配置必可行）。
        # 治 MRED 模式闭式 slack 一阶近似失准：07-09 rerun k02-k12 有 25-64% 的
        # episode 整集实测超 budget（k06 达 77/120），8 次 DC 全浪费。
        outer_errgate=False,
        outer_errgate_vectors=2_000_000,   # 预筛 MC 向量数（门控只看均值型 med/mred，2M 足够）
        outer_errgate_max_repairs=6,       # 贪心摘 cell 步数上限；超限清空 cells
        # 外环 cell 维度求解器（默认 None=进化变异不变）。"greedy" = 每 episode 在
        # sample-0 布线上用张量化仿真器实测 Δmred 打分做 lazy-greedy + 升级扫描解 cell
        # 包（Appr_Comp/cellsolver），替代进化 cell 变异；结构搜索仍归外环 struct 变异。
        # 多架构真实 DC+XA 验证（OUTER_CELL_SEARCH.md §3.2.4）：深截断 k12/14 greedy 面积
        # 稳赢 GA、功耗打平。求解在 self.device(GPU)上,~5% episode 开销（DC 主导）。
        outer_cell_solver=None,
        outer_solver_vectors=16_000_000,   # 求解器 MC 向量池（MRED 重尾需 16M 校准）
        outer_solver_cache=None,           # 向量池缓存目录（跨 episode 复用,None=build_dir下）
        # 求解余量：解到 budget×margin。greedy 的 cell 选择利用 sample-0 布线特有的
        # 误差抵消,包在其余布线上系统性偏高(+3~5%,07-10 首集 3 个 k 全部 7/9 越线),
        # 贴线填充(≈99%)必然大面积报废;0.9 → 其余样本落线内。
        outer_solver_margin=0.9,
        # ===== M2（PARETO_ARITH_PLAN.md §7.2，默认关=行为不变）=====
        # 批量 ZERO 算子：zero-col 把最低未清列整列填 ZERO（= 分数截断一步）、
        # unzero-col 反向。解析模型对边界列 ZERO 失真（实测 bias 是解析 3.7×）→
        # 该算子跳过闭式预算过滤，可行性交给 TT oracle / errgate / v5 档案准入。
        outer_zero_ops=False,
        outer_p_zero=0.15,           # 变异骰子里 zero 算子的权重
        # TT oracle：sample-0 布线上用 cellsolver 张量化仿真器实测 cell 配置 mred
        # （与 16M verilator 闸门同流逐位一致，秒级），替代 verilator 预筛门；
        # 超上限按解析贡献降序二分前缀摘除。v5 上限 = 档案 mred 上限（超伪预算
        # 只是落松箱，不摘）；预算模式上限 = mred_budget。
        outer_tt_oracle=False,
        # ===== V6（V6_SEARCH_PLAN.md，默认关=行为不变）=====
        # R3 档案内杂交：同箱同 k 取第二亲本，cell 配置逐槽位均匀重组；
        # 闭式预算修复（含零 cell 的子代跳过闭式，交 TT oracle/档案，与 M2 同语义）。
        outer_crossover=False,
        outer_p_crossover=0.2,       # 静态骰子下 crossover 臂权重（bandit 开启时不用）
        # R2 bandit 自适应骰子：按 (箱,臂) Thompson 采样替代静态概率；
        # 观测=该臂 episode 是否 ≥1 入档，滑动窗口抗非平稳，每臂保底 floor。
        outer_bandit=False,
        outer_bandit_window=12,
        outer_bandit_floor=0.05,
        # R1 单集多配置：被选臂多掷 G 次产出去重后的 G 个 cell 配置，num_samples
        # 路布线按组均分（同 DC 预算下假设吞吐 ×G；r2 账本：keep 臂 33% 预算
        # 20% 产出 → 布线过采样、假设欠采样）。=1 为旧行为；仅 v5+非求解器生效。
        outer_multi_config=1,
        **kwargs,
    ):
        # ── v5 精简版守卫：下列遗留功能已从本文件剪除（完整实现在 trainer/arith_das.py）。
        # 显式开启一律 fail-fast 报错——静默忽略会悄悄改变实验语义。
        _removed = {
            "use_delay_loss": use_delay_loss,    # 可微延迟 surrogate（所有 config 均关）
            "use_error_loss": use_error_loss,    # 可微误差 surrogate（mred 模式下 train_dc 强制关）
            "use_power_proxy": use_power_proxy,  # 功耗代理模型（train_dc 固定 EDA 直出）
            "gomil_path": gomil_path,            # GOMIL 温启动（无 config 使用）
            "area_budget": area_budget,          # 面积预算约束模式（train_dc 固定 None）
        }
        _on = [k for k, v in _removed.items() if v]
        if power_source == "proxy":
            _on.append("power_source='proxy'")
        if synth != "dc":
            _on.append(f"synth={synth!r}（仅支持 'dc'）")
        if fixed_target_delay is None:
            _on.append("fixed_target_delay=None（v5 需固定单一 DC 周期）")
        if _on:
            raise ValueError(
                "arith_das_v5 是 v5 精简版，不支持: " + ", ".join(_on)
                + "；如需这些功能请用 trainer.arith_das.CompressorRouting"
            )

        self.bit_width = bit_width
        self.encode_type = encode_type
        self.ct_arch = ct_arch
        self.use_ct42 = bool(use_ct42)
        self.num_node_types = 5 if self.use_ct42 else 4

        self.lse_gamma_val = lse_gamma_val
        self.num_episodes = num_episodes
        self.log_dir = log_dir
        self.build_dir = build_dir
        self.device = device
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.front_dump_freq = front_dump_freq
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.n_processing = n_processing

        self.delay_weight = delay_weight
        self.area_weight = area_weight
        self.power_weight = power_weight
        self.delay_scale = delay_scale
        self.area_scale = area_scale
        self.power_scale = power_scale

        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.reference_point = reference_point
        self.use_delay_loss = use_delay_loss
        self.use_rule_loss = use_rule_loss
        self.use_disc_loss = use_disc_loss
        self.delay_loss_weight = delay_loss_weight
        self.rule_loss_weight = rule_loss_weight
        self.disc_loss_weight = disc_loss_weight
        self.pareto_target = pareto_target

        self.use_ppo_loss = use_ppo_loss
        self.ppo_loss_weight = ppo_loss_weight

        self.pool_size = pool_size
        self.rule_loss_weight_incr = rule_loss_wight_incr
        self.disc_loss_weight_incr = disc_loss_weight_incr

        self.gomil_path = gomil_path
        if isinstance(init_pool_best_info, str):
            init_pool_best_info = [init_pool_best_info]
        self.init_pool_best_info = init_pool_best_info
        self.synth = synth
        self.kwargs = kwargs
        self.fixed_target_delay = fixed_target_delay
        self.area_budget = area_budget
        self.area_violation_weight = area_violation_weight
        self.delay_violation_weight = delay_violation_weight
        # 功耗源：v5 只支持 EDA 直出（DC report_power）。proxy 路径已整体剪除（见守卫）。
        if power_source is None:
            power_source = "eda"
        if power_source != "eda":
            raise ValueError(
                f"Invalid power_source={power_source!r}; arith_das_v5 仅支持 'eda'"
            )
        self.power_source = power_source

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.tb_logger = SummaryWriter(self.log_dir)
        else:
            self.tb_logger = None

        self.gnn_a = None
        self.gnn_b = None
        self.gnn_c = None

        self.gcn_kwargs = copy.deepcopy(gcn_kwargs)
        expected_input_dim = 3 + self.num_node_types
        if int(self.gcn_kwargs.get("input_dim", expected_input_dim)) != expected_input_dim:
            logging.info(
                "[ct42] overriding gcn input_dim %s -> %s",
                self.gcn_kwargs.get("input_dim"),
                expected_input_dim,
            )
            self.gcn_kwargs["input_dim"] = expected_input_dim
        self.gcn = MultiChannelResGCN(**self.gcn_kwargs)
        self.gcn.to(device)

        # ===== Phase B：近似类型搜索状态 =====
        self.use_approx_types = use_approx_types
        self.approx42_library_path = approx42_library_path
        self.approx42_rtl_path = approx42_rtl_path
        self.approx42_max_types = approx42_max_types
        self.approx_max_col = approx_max_col
        self.approx_col_window = approx_col_window
        self.med_budget = med_budget
        self.med_violation_weight = med_violation_weight
        self.error_gate = error_gate
        self.error_gate_vectors = int(error_gate_vectors)
        self.bias_weight = bias_weight
        self.error_scale = error_scale
        self.error_as_metric = bool(error_as_metric)
        self.error_weight = error_weight
        self.inject_exact_candidate = bool(inject_exact_candidate)
        self.approx_cardinality_sampler = bool(approx_cardinality_sampler)
        if approx_cardinality_choices is None:
            approx_cardinality_choices = [0, 1, 2, 4, 8, 16]
        self.approx_cardinality_choices = [
            int(x) for x in approx_cardinality_choices
        ]
        if sorted(set(self.approx_cardinality_choices)) != self.approx_cardinality_choices:
            raise ValueError(
                "approx_cardinality_choices must be sorted unique non-negative ints"
            )
        if self.approx_cardinality_choices[0] != 0:
            raise ValueError("approx_cardinality_choices must start with 0")
        self.approx_cardinality_logits = None
        self.error_scale_mode = error_scale_mode
        self.use_error_loss = use_error_loss
        self.error_loss_weight = error_loss_weight
        self.bias_loss_weight = bias_loss_weight
        # ④ 尾部/WCE 约束
        self.wce_budget = wce_budget
        self.wce_violation_weight = wce_violation_weight
        self.wce_loss_weight = wce_loss_weight
        # Phase C ①：截断（_setup_truncation 在 _start_reset 拿到 initial_pp 后填充）
        self.trunc_cols = int(trunc_cols or 0)
        self.trunc_correct = trunc_correct
        self._trunc_bits = {}      # {col: 该列常数 1 的个数}
        self._trunc_const = 0      # 实际注入的校正常数 C
        self._trunc_delta = 0.0    # E[Δ]（截断期望丢失值）
        self._trunc_wce = 0.0      # 截断最坏情况误差 max(C, Δmax−C)
        self._trunc_med = 0.0      # E[|C−Δ|]（截断残差 MED，P=1/4 一阶估计）
        self._trunc_model_mred = None  # C* 处解析模型 MRED（外环 MRED-slack 过滤用）
        # 外环 cell 搜索
        self.outer_cell_search = bool(outer_cell_search)
        self.outer_p_struct = float(outer_p_struct)
        self.outer_p_cell = float(outer_p_cell)
        self.outer_p_resample = float(outer_p_resample)
        self.outer_proposal_retries = int(outer_proposal_retries)
        self.outer_med_slack_scale = float(outer_med_slack_scale)
        self.outer_w_area = float(outer_w_area)
        self.outer_w_err = float(outer_w_err)
        self.outer_errgate = bool(outer_errgate)
        self.outer_errgate_vectors = int(outer_errgate_vectors)
        self.outer_errgate_max_repairs = int(outer_errgate_max_repairs)
        self.outer_cell_solver = outer_cell_solver  # None | "greedy"
        self.outer_solver_vectors = int(outer_solver_vectors)
        self.outer_solver_cache = outer_solver_cache
        self.outer_solver_margin = float(outer_solver_margin)
        self.outer_zero_ops = bool(outer_zero_ops)
        self.outer_p_zero = float(outer_p_zero)
        self.outer_tt_oracle = bool(outer_tt_oracle)
        self.outer_crossover = bool(outer_crossover)
        self.outer_p_crossover = float(outer_p_crossover)
        self.outer_bandit = bool(outer_bandit)
        self.outer_bandit_window = int(outer_bandit_window)
        self.outer_bandit_floor = float(outer_bandit_floor)
        self.outer_multi_config = max(1, int(outer_multi_config))
        self._v5_bandit = None       # enable_pareto_v5 里按 outer_bandit 实例化
        self._outer_last_op = None   # 本 episode 的骰子臂（bandit 归因用；种子集=None）
        self._episode_cell_configs = None   # V6-R1：本集 G 个 cell 配置（组0=主掷）
        self._episode_ct_groups = None      # 逐组 slot→node 类型映射（reset 重建）
        if self.outer_zero_ops or self.outer_tt_oracle:
            logging.info("[outer] M2: zero_ops=%s p_zero=%.2f tt_oracle=%s",
                         self.outer_zero_ops, self.outer_p_zero,
                         self.outer_tt_oracle)
        if self.outer_crossover or self.outer_bandit or self.outer_multi_config > 1:
            logging.info("[outer] V6: crossover=%s p_crossover=%.2f bandit=%s(w=%d floor=%.2f) "
                         "multi_config=%d",
                         self.outer_crossover, self.outer_p_crossover,
                         self.outer_bandit, self.outer_bandit_window,
                         self.outer_bandit_floor, self.outer_multi_config)
        self._cell_solver_pool = None   # 惰性加载的 (a,b) 向量池,跨 episode 复用
        self._episode_cell_types = {}
        if self.outer_cell_solver and not self.outer_cell_search:
            logging.warning("[outer] outer_cell_solver 需 outer_cell_search=True，已忽略")
            self.outer_cell_solver = None
        if self.outer_cell_search and not use_approx_types:
            logging.warning(
                "[outer] outer_cell_search=True 但 use_approx_types=False：无类型表，"
                "cell/resample 算子退化为纯结构变异"
            )
        # P0：delay 约束化
        self.delay_as_constraint = bool(delay_as_constraint)
        self.delay_target_ns = delay_target_ns
        self.normalize_advantage = normalize_advantage
        self.type_table_32 = None
        self.type_table_22 = None
        self.type_table_42 = None
        self.type_head_32 = None
        self.type_head_22 = None
        self.type_head_42 = None
        self.approx_module_src_by_name = {}
        self._node_emb = None  # get_Z_mat 设置：逐节点嵌入（类型头 + ppo 复用）
        if self.use_approx_types:
            self._load_approx_types(approx_lib_path, approx_library_path)
            self.type_head_32 = nn.Linear(
                self.gcn.embedding_dim, len(self.type_table_32)
            ).to(device)
            self.type_head_22 = nn.Linear(
                self.gcn.embedding_dim, len(self.type_table_22)
            ).to(device)
            if self.use_ct42:
                self.type_head_42 = nn.Linear(
                    self.gcn.embedding_dim, len(self.type_table_42)
                ).to(device)
            if self.approx_cardinality_sampler:
                if approx_cardinality_init_logits is None:
                    approx_cardinality_init_logits = [0.0] * len(
                        self.approx_cardinality_choices
                    )
                if len(approx_cardinality_init_logits) != len(
                    self.approx_cardinality_choices
                ):
                    raise ValueError(
                        "approx_cardinality_init_logits length must match "
                        "approx_cardinality_choices"
                    )
                self.approx_cardinality_logits = nn.Parameter(
                    torch.tensor(
                        [float(x) for x in approx_cardinality_init_logits],
                        device=device,
                        dtype=torch.float32,
                    )
                )
            # 类型头初始化偏向 exact(index0)：给 bias[0] 设正偏置，使初始策略≈"近全 exact"、
            # 按需再加近似 cell（而非随机≈均匀→~85% 节点一上来就近似）。默认 0=旧行为。
            if exact_init_bias:
                import math as _m
                with torch.no_grad():
                    self.type_head_32.bias[0] = float(exact_init_bias)
                    self.type_head_22.bias[0] = float(exact_init_bias)
                    if self.type_head_42 is not None:
                        self.type_head_42.bias[0] = float(exact_init_bias)
                eb = _m.exp(float(exact_init_bias))
                p42 = (
                    eb / (eb + len(self.type_table_42) - 1)
                    if self.use_ct42 and self.type_table_42 else None
                )
                if p42 is None:
                    logging.info(
                        "[approx] type_head exact-init bias=%.2f -> 初始 P(exact)≈%.2f(T32)/%.2f(T22)",
                        exact_init_bias, eb / (eb + len(self.type_table_32) - 1),
                        eb / (eb + len(self.type_table_22) - 1),
                    )
                else:
                    logging.info(
                        "[approx] type_head exact-init bias=%.2f -> 初始 P(exact)≈%.2f(T32)/%.2f(T22)/%.2f(T42)",
                        exact_init_bias, eb / (eb + len(self.type_table_32) - 1),
                        eb / (eb + len(self.type_table_22) - 1), p42,
                    )
            # （原版此处还构建 _bias/_wae/_maxe 逐类型张量缓存，仅供已剪除的
            #   可微误差 surrogate get_error_loss 使用。）
            t42_msg = f" T42={len(self.type_table_42)}" if self.use_ct42 else ""
            logging.info(
                "[approx] type heads on: T32=%d T22=%d%s, max_col=%d, col_window=%s, "
                "med_budget(LSB)=%s, use_error_loss=%s, wce_budget(LSB)=%s",
                len(self.type_table_32), len(self.type_table_22), t42_msg,
                self.approx_max_col, self.approx_col_window, self.med_budget,
                self.use_error_loss, self.wce_budget,
            )
            if self.approx_cardinality_sampler:
                logging.info(
                    "[approx] cardinality sampler on: choices=%s init_logits=%s",
                    self.approx_cardinality_choices,
                    [float(x) for x in self.approx_cardinality_logits.detach().cpu()],
                )

        opt_params = list(self.gcn.parameters())
        if self.use_approx_types:
            opt_params += list(self.type_head_32.parameters())
            opt_params += list(self.type_head_22.parameters())
            if self.type_head_42 is not None:
                opt_params += list(self.type_head_42.parameters())
            if self.approx_cardinality_logits is not None:
                opt_params.append(self.approx_cardinality_logits)
        self._opt_params = opt_params   # 梯度裁剪要盖全（含类型头/cardinality logits）
        self.optim: optim.Optimizer = getattr(optim, optim_name)(
            opt_params, **optim_kwargs
        )
        self.scheduler: optim.lr_scheduler.LRScheduler = getattr(
            optim.lr_scheduler, scheduler_name
        )(self.optim, **scheduler_kwargs)

        self.init_policy_from = init_policy_from
        if init_policy_from:
            self._load_policy(init_policy_from)

        self.comp_graph: CompressorGraph = None
        self.state: Dict[str, np.ndarray] = None
        self.assignment = None

        self.found_best_info = {
            "objective": float("inf"),
            "simulated_result": None,
            "connection": None,
            "assignment": None,
            "ct": None,
            "cell_types": None,  # Phase B：最优设计的每槽 cell 类型，导出时复原近似 cell
            "cell_type_info": None,
        }

        self.total_epoch_num = 0

        self.initial_pp: np.ndarray = None

        if pool_size > 0:
            self.pool = BoundedParetoPool(pool_size)
        else:
            self.pool = None

        self._start_reset()

    # ==== 近似类型搜索的公共常量（Phase B 方法体见 cell_types.py）====
    _REPO_ROOT = os.path.dirname(  # core.py 在包内，比原单文件多一层目录
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # M2 锚点口径：exact cell 的在环境面积（T28 12T40P140：FA1D0=2.856、HA1D0=2.184、
    # CT42≈2×FA=5.712）。菜单 meta 的 standalone SOP 锚点虚高 ~4×（FA=10.92），会把
    # "比在环境 exact 更贵"的 cell 也标成省面积。cellsolver 侧 07-11 已切此口径
    # （cash-in 从 24-35% 修到 60-158%），提议打分对齐。
    _EXACT_AREA_INCTX = {0: 2.856, 1: 2.184, 4: 5.712}

    def _resolve_path(self, p):
        """相对路径按仓库根解析（pipeline 会 chdir 到 output 目录）。"""
        return p if os.path.isabs(p) else os.path.join(self._REPO_ROOT, p)
