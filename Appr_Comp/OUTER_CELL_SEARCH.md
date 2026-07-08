# 外环 cell 搜索设计（解析提议 + 可行性过滤 + resample-K）

> 2026-07-07。把近似 cell 类型决策从内环（每 routing 样本采样 + PPO）移到外环（进化状态 +
> 解析提议变异 + 闭式可行性过滤），内环只做布线。总开关 `outer_cell_search`，默认 false =
> 行为字节级不变（项目惯例）。

## 0. 动机（证据见 2026-07-04 paired comparison 与 CT42 裁决）

1. **信用混淆**：内环 8 样本同时变布线和类型，PPO 分不清功劳；cell 效应（±1–3%）被布线/
   结构效应（±5–10%）淹没。且类型采样本就条件在结构图 embedding、不依赖具体布线——
   "类型适应布线"的理论好处不存在（PHASE_B §9.4 自己声明独立采样）。
2. **好类型配置保不住**：pool 只存 ct 数量，episode 中发现的好 cell 摆放不进池、每轮重采。
3. **误差不可行样本浪费 EDA**：类型在 DC 之后才知道超 budget（over_budget=8/8 的整轮浪费）；
   而类型固定后解析误差闭式、微秒级——过滤应该发生在花钱之前。

架构对应因果结构：**误差档位由结构决定（k、counts、cells），同误差 PPA 由布线决定**。
外环定误差，内环纯 PPA。

## 1. 状态表示

`state["cells"] = [[s, c, t, idx, k], ...]`
（slot 坐标 = assignment 顶点身份 (stage, col, type, idx)，跨结构变异稳定；k = 该压缩器
类型表 T32/T22/T42 的索引，恒 ≥1——exact 不入表。JSON/pickle 安全。）

- episode 开始建好 `CompressorGraph` 后经 `indice_map[(s,c,t,idx)] → node_idx` 转成
  内部 `{node_idx: (t,k)}`，下游发射/objective/best/export 全部复用现有路径（零改动）。
- 结构变异后 slot 可能消失：按「新 assignment 中 (s,c,t,idx) 仍存在」过滤，掉落的 cell
  丢弃并记日志。
- pool 状态、`_start_reset` 初始态、gomil 态都带 `cells`（初始 []）。

## 2. 闭式误差核算（单一事实源）

新 helper `_error_totals_from_cols(items)`，items = [(col, t, k)]：
```
med  = MED_trunc + Σ wae(t,k)·2^col          （MED 保守上界，LSB）
bias = (−E[Δ]+C) + Σ bias(t,k)·2^col          （带符号，供提议配平）
wce  = WCE_trunc + Σ maxe(t,k)·2^col          （精确可加上界）
```
`_analytic_error`（node_idx 口径）改为映射 col 后调它——过滤器与 reward 口径强一致。

## 3. 可行性过滤 `_cells_budget_ok(cells)`

按设置了哪个预算逐项检查，全过才接受：

| 模式 | 判据 |
|---|---|
| `med_budget` 设置 | `med ≤ med_budget` |
| `wce_budget` 设置 | `wce ≤ wce_budget` |
| `error_metric=mred` 且 `mred_budget` 设置且 `trunc_cols>0` | `Σ wae·2^col ≤ slack`，其中 `slack = MED_trunc·(mred_budget/MRED_floor − 1)·outer_med_slack_scale`；`MRED_floor` = `_setup_truncation` 里 C* 处的解析模型 MRED（新增缓存 `_trunc_model_mred`） |
| `error_metric=mred` 且 `trunc_cols=0` | 无闭式 MRED 下界可用 → 只查 WCE（若设），其余放行交给 verilator 闸门（诚实声明的空洞） |
| `error_as_metric`（无预算） | 不过滤（误差是软目标） |

**假设声明**：MRED-slack 规则假设「cell 误差对 MRED 的推动 ≈ 它对 MED 的相对推动」，
是一阶近似（cell 落在高列时对小积的 1/p 加权可能超线性）；`outer_med_slack_scale`
（默认 1.0）留作保守化旋钮，实测越界率高就调小。

## 4. 解析提议（变异算子）

外环 reset 变异先掷算子骰子（概率 `outer_p_struct / outer_p_cell / outer_p_resample`，
默认 0.4/0.4/0.2）：

**struct**：现有 4/6 动作不变；变异后修剪失效 slot。

**cell**（子骰子 add 0.5 / remove 0.25 / swap 0.25）：
- **add**：候选 = 空闲合法 slot（列 ∈ [trunc_cols, upper)，表长>1）× 该型非 exact cell。
  - 符号偏好：残差 `bias` < −0.5 LSB → 优先 P 组；> +0.5 → 优先 N 组；否则不限。
  - 打分（softmax 采样，非 argmax——保留随机性）：
    `logit = outer_w_area·area_save_frac − outer_w_err·(wae·2^col)/max(slack, ε)`
    （area_save_frac = (exact_area−cell_area)/exact_area，表里没有 area 字段记 0——
    当前 T42 库未表征即如此，等 char_driver42 跑完自动获益。）
  - 每个提议过 §3 过滤，不过换下一个，`outer_proposal_retries`（默认 50）耗尽则本次
    cell 维度不变（结构照常评估，episode 不作废）。
- **remove**：现有 cells 中均匀删一个（永远可行）。
- **swap**：均匀选一个已放 cell，删掉后对该 slot 按 add 流程重提议。

**resample-K（大步）**：K 从 `approx_cardinality_choices` 均匀采样（mask 到 ≤ 空闲 slot 数；
不可行的大 K 会被过滤器天然修剪，无需先验倾斜）；清空后**串行贪心加 K 个**——每加一个都
重算残差 bias 和 slack 再提议下一个（串行 = 正负抵消天然内建）；中途无可行候选则提前停
（K_actual < K，记日志）。这就是 cardinality sampler 的三层结构搬到外环、
把「学习的 logits」换成「均匀先验 + 解析打分」。

## 5. 内环改动（`outer_cell_search=true` 时）

- `get_samples`：跳过 `sample_cell_types`；全部样本共用
  `cell_map = _cell_map_from_types(self._episode_cell_types)`；`cell_type_info={"mode":"outer"}`；
  type_log_prob 不存在。`inject_exact_candidate` 照旧（同布线全 exact）→ **每轮自动 paired**。
- `get_ppo_loss` / `_cell_type_log_prob`：mode=="outer" 返回 None → ratio 只含布线。
- `run_episode` 池更新：获胜样本若是 all-exact baseline，则以 `cells=[]` 入池；
  变异配置本身也照常入池（两个变体都保留）。
- `approx_cardinality_logits` 不再进 optimizer 梯度路径（外环模式下不采样）；
  类型头仍创建（供日后"学习提议"升级），本版不更新。

## 6. 新配置键（全部默认关/保守）

```yaml
outer_cell_search: false
outer_p_struct: 0.4
outer_p_cell: 0.4
outer_p_resample: 0.2
outer_proposal_retries: 50
outer_med_slack_scale: 1.0
outer_w_area: 1.0
outer_w_err: 1.0
```
`scripts/train_dc.py` 加 `--outer_cell_search` CLI 开关。

## 7. 回归保证

- 开关关：不触任何新分支，reset/get_samples/PPO 与现行为逐字节一致（冒烟：同 seed 两跑
  emit RTL diff 为空）。
- 开关开：闭式核算与 `_analytic_error` 同源（单元测试对拍）；提议产物全部过滤器；
  8-bit 位精确恒等式（approx_mul 口径）不受影响（cell 语义未变，只是决策位置变了）。

## 8. 后续升级位（本版不做）

- 类型头作为学习提议（成败统计 + 交叉熵），算子选择 bandit；
- T42 库 DC 表征后 area 项自动生效；
- pool 存 connection（方案 E 完全体）；no-op 结构动作。

## 9.1 验证结果与 v1.1 修正（2026-07-08）

两组训练对照（均 seed42/td1.5/verilator 闸门，XA reeval 口径）：

**budget 模式**（`outputs/2026-07-07_22_med_outer_v1_np5`，ep120/s8/np5，对齐 6 月 Round 2 的
10 个 (k,med_budget) 点）：
- **预算合规 10/10 vs June 6/10**（June k9 超 4.3×、k10_b1536 超 3.2×、k12_b20480/k13 边缘超）
  ——可行性过滤器完全达成设计目标，且把前沿延伸进 June 超调够不到的低误差区
  （MED 709/1197 两个新点）；中段多处支配（10839/660µm² vs June 21216/678；19605/626 vs 42106/629）。
- 代价：同名义 budget 下 power 平均 +4~5%。cells 活跃（每设计 2–13 个）。

**error_as_metric 模式**（`outputs/2026-07-08_05_med_outer_v2align_np5`，ep60/s2/np5，对齐
2026-06-27_error_obj_v2 橙线 k02..k20）：
- 6/10 点收敛到与橙线完全相同的纯截断设计（MED 逐位相同），但 area 平均 +0.6%、
  **power 平均 +16%（k20 +43%）全面劣化**。
- **根因（v1.0 设计缺陷）**：算子骰子把结构变异强度砍到 40%（ep60 仅 ~24 次结构变异 vs
  基线 60 次），结构/布线搜索被饿着；v1 ep120×0.4≈48 次故只差 4~5%，与「power 差距 ∝
  结构变异缺口」一致。
- **v1.1 修正（已实施）**：结构变异每轮必做（与基线同强度），cell 层（闭式免费）作为叠加：
  keep/单 op/resample-K 按 outer_p_* 比例。日志 `op=struct+{keep,cell,resample}`。

**v1.1 重跑裁决**（`outputs/2026-07-08_10_med_outer_v2align_v11_np5`，同参数）：
- **power 差距 +16% → +5.5%（均值）**；v1.0 最差的纯截断诊断点全部翻盘：
  k16 +24.2%→**−2.4%**、k18 +23.4%→**−2.7%**（反超橙线）；k20 +43%→+16% 残留。
- **area 全面追平反超**：10 点均值 −0.6%（8/10 点 ≤ 橙线，k14 −3.1%）。
- MED：7/10 点与橙线收敛到相同设计（floor 逐位同）；k10/k14 用 3/16 个 cell 换误差。
- 残余 +5.5% 集中在 k02–k10（+4~9%）与 k20（+16%，绝对差仅 0.023mW，小设计放大了百分比）
  ——量级已落入既往观测的单 seed 功耗噪声带（±9%）边缘，要进一步归因需多 seed。
- **结论：v1.1 修复有效。budget 模式明确胜出（合规 10/10 + 前沿延伸 + 面积支配），
  error_as_metric 模式面积微优、功耗均值差 5.5%（≈噪声带）——外环架构可作为 cell 搜索
  默认；后续微调方向：pool 双变体稀释 exploitation 的影响、多 seed 复验。**

## 9. 实现状态（2026-07-07，已全部落地并冒烟）

代码：[trainer/arith_das.py](../trainer/arith_das.py)（`_error_totals_from_cols` 单一事实源、
`_outer_med_slack`/`_cells_budget_ok`/`_enumerate_type_slots`/`_cells_prune_stale`/
`_propose_cell_add|remove|swap`/`_op_resample_k`/`_outer_mutate`、reset/get_samples/
`_cell_type_log_prob`/run_episode 池更新/`_start_reset` 的 outer 分支）+
[scripts/train_dc.py](../scripts/train_dc.py) `--outer_cell_search`。

已验证（arith_das env）：
1. `_analytic_error` 重构与旧公式随机配置逐位一致（默认关回归）；默认关时 state 无
   cells 键、reset 老路径不变。
2. MRED-slack 闭式正确（k8@1.3e-5 → slack 410.6 LSB）且真实 train_dc 顺序
   （构造后赋 error_metric → 重算截断 → C* 缓存）下严格 binding：resample-K 贪心填到
   408 LSB 即停。
3. 符号偏好生效：截断残差 −282 时 100/100 提议 P 组，变异把净偏置从 −282 推到 −90。
4. 200 步混合变异全程可行、无 stale cell、结构全合法；resample-K ×50 无重复 slot。
5. 完整构造器 + `_start_reset` + reset：cells 正确映射 node_idx 并生成 cell_map。

跑法：`scripts/train_dc.py --config configs/config_groups/mul_16_approx_error_obj.yaml
--outer_cell_search --trunc_cols k --error_metric mred --mred_budget ... `（其余与
Scheme A 对齐即可与 `2026-07-05_02` 直接对比）。
