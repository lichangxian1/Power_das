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

## 3.1 实测误差预筛门 `outer_errgate`（2026-07-09，默认关）

上面的一阶近似在 MRED 模式下**实测严重失准**：07-09 rerun（`2026-07-09_06_mred_outer_rerun_np5`）
k02–k12 有 25–64% 的 episode 整集实测超 budget（k06 77/120；99 个带 cell 集中 77 个超标，
n_cells=0 集零超标），闭式过滤 0 次拒绝——每个超标集 8 次 DC（~200s/样本）全浪费，
且把外环教育成"少放 cell"。

预筛门（`--outer_errgate`）：`get_samples` 里 sample-0 RTL 发射后、进 DC 前，先用
verilator MC（默认 2M 向量，秒级；门控只看均值型 med/mred，无需 16M）实测本集共用的
cell 配置：

- 判据与 `get_objective` 同口径（MRED 模式比 mred，MED 模式比 med，WCE 不进门）；
- 超预算 → `_outer_drop_worst_cell` 贪心摘掉解析贡献 `wae·2^col` 最大的 cell（每步误差
  严格降、尽量保留摆放）→ 更新 `state["cells"]`/映射 → 重发射 sample-0 → 复测；
- 修复步数耗尽（默认 6）仍超 → 清空 cells 保底（floor 配置必可行）；
- verilator 探测失败 → 放行（与 error_gate 回退策略一致，不因门故障丢整集）。

注意：探测用 sample-0 的布线，其余样本布线不同、实测误差略有差异（旧日志有少数 5/9、3/9
的部分超标集），门是强筛不是保证。每次探测顺带日志实测/解析比值，为后续修正 slack
一阶近似积累标定数据。代码：`_outer_gate_active`/`_gate_budget_exceeded`/
`_outer_drop_worst_cell`/`_refresh_episode_cell_types`/`_outer_errgate_screen`；
冒烟：`scripts/smoke_outer_errgate.py`（假测量对拍控制流，A 摘除最坏/B 耗尽清空/
C 失败放行/D 开关口径，全过）。

## 3.2 cell 维度求解器路线图（2026-07-10 定稿，Claude×Codex 两轮收敛）

> 把 cell 选择从"外环变异算子随机走"升级为"给定布线下的直接求解"。每步带判定门,
> 过线才做下一步;期望前沿增益为个位数,主要价值是**决定性**(最强求解器仍 ~0 →
> cell 问题永久关闭)+ 把 ~40% 外环变异预算还给结构。

**执行顺序**:

1. **errgate 复验**:纯截断基线用 `scripts/regen_trunc_mred_baseline.py` 在同口径下重生成
   (同 DC/XA 流程、同 verilator 16M 口径),然后带 `--outer_errgate` 重跑 k02–k12。
   **状态(07-10)**:基线已就绪(`outputs/2026-07-09_mred_trunc_baseline`,10/10 XA 成功);
   errgate 重跑 = `outputs/2026-07-09_21_mred_warm240eg_np4`(warm240,10 个 k,07-09 21:47
   起跑,ep~50/240)。门实测生效:k06 超预算集 77/120 → 1/48、k12 → 0/50,零 verilator 回退。
2. **Q1 配对回归**(与 1 并行,零 EDA 成本):回归"实测配对 ΔP"对"Σ standalone 预测 Δp"。
   判据:Spearman ≥ 0.8 且符号一致率 ≥ 85%(只在 |ΔP| 超 XA 噪声地板的 pair 上算)→
   功耗预测器允许进 solver loss;0.6–0.8 → 仅离散化后 tie-breaking;<0.6 → 弃用,
   PPA 项保持纯面积。
   **裁决(07-10,`pwrpred/q1_paired_regression.py`)**:同布线 DC 配对 Spearman 0.667
   (tie-break 档);设计级同截断层 0.189/符号 43%(FAIL)。**预测器不进 solver loss,
   PPA 项定为纯面积**;单 cell 边际 ΔP 低于 DC 噪声地板且符号会翻(routing 盲区主导),
   详见 pwrpred/FEASIBILITY.md 第三部分。
3. **停止判据**(修正版,原"功耗<3%关闭"在单 seed ±9% 噪声带下不可判定):
   **主判定轴 = 面积**,cell 相对纯截断面积增益 <1.5% → 关闭 cell 线,外环退回纯结构;
   功耗侧跨 k 聚合:6 个 k 点同 seed 配对,≥5 点方向为正且配对均值 >2% 才算有油水。
4. **贪心/背包 v1**(仅当 3 过线):无超参、可解释,直接量化 MRED 非线性/误差抵消的价值。
5. **diffam 式梯度 solver**(仅当 4 显示非线性余量仍在):**验收基线 = ④贪心+多随机重启**,
   打不赢就停在 ④(Codex 风险提示:STE/Gumbel 在深层硬布尔图上的梯度可能只是弱启发)。

**solver 技术规格**(④⑤共用):

- **挂载点**:与 §3.1 errgate 同钩子——sample-0 布线固定,solver 在其上解 cell 包,
  更新 `state["cells"]`,重发射,过 errgate,进 DC;**每 episode 解一次**,其余样本共用
  (外环模式布线策略 cell-blind,无先后耦合;解决"top routing 候选"的鸡生蛋问题)。
- **形态**:每 slot 一个 logits 向量,候选集 = 合法 (t,k);**精确真值表张量化前向**
  (不复用 diffam 的 C42Feature/C42Behavior MLP——8-bit 原型历史包袱,TT 本可精确表示);
  hard 前向 + STE/Gumbel-ST 反向;loss = simulated MRED hinge + 精确 cell 面积和(v1)。
- **C\***:不进梯度路径;离散化后按现有闭式重算,再过 errgate。
- **口径对齐**(硬约束):solver 内 MC 逐条复刻 `mul_err_wrap.cpp`——31-bit masked golden、
  circular-wrap 到 [−2³⁰,2³⁰)、golden≠0 才计 MRED 分母;同一固定随机流前缀抽样,保证
  256k/1M/16M 之间 common random numbers;离散化终验仍走 verilator 16M。

**定位**:solver 只接管 cell 维度;结构搜索继续归 v1.1 变异 + 内环 PPO,不动。

### 3.2.1 ③④原型落地与 k12 裁决(2026-07-10,`Appr_Comp/cellsolver/`)

用户裁定跳过门控直接做架构尝试("直接结果说话")。三件基建 + 一个裁决:

1. **精确 TT 张量化仿真器**(`sim.py`):树连线复刻 emit_assignment 语义、pp 接线直接
   解析 emit_pp_encoder 文本(含截断常数,AND/Booth 双支持)、xorshift128+ 随机流逐位
   复刻 harness。**对拍 verilator 2M/16M:Δmed=0.00e+00(整数级一致),Δmred~5e-9**。
   GPU 上 2M 约 1-2s(vs verilator 编译+运行 ~4s)。
2. **分层 MRED 估计器**(关键发现):MRED 被极少数小乘积样本主导(k12 floor 在 200k
   均匀前缀下测 3.2e-3、16M 下 1.9e-4,差 16×)——均匀小批量 MC 梯度/gate 全是噪声。
   分层:S12(0<g<2²²,16M 流中 ~134k 个)全量精确 + S3 固定子样本加权 → 确定性、
   ≈16M 口径、秒级。**估计值与 verilator 16M 四位有效数字一致**。
3. **③贪心(lazy greedy 实测打分 + 升级扫描)与④梯度(logits+STE+对偶上升)**:
   同一估计器、同一菜单(T32=10/T22=5/T42=16)、同一修复兜底。

**k12@2.8e-4 裁决**(07-09_06 rerun 最优结构/布线上求解,verilator 16M 终验):

| 配置 | n_cells | mred(16M) | 利用率 | Σcell面积节省(µm², standalone 口径) |
|---|---|---|---|---|
| 纯截断 floor | 0 | 1.909e-4 | 68.2% | 0 |
| GA(外环 v1.1) | 7 | 2.785e-4 | 99.5% | 24.86 |
| **③贪心+升级** | **49** | **2.788e-4** | **99.6%** | **189.67(=7.6× GA)** |
| ④梯度(独立) | 3 | 2.375e-4 | 84.8% | 12.26 |
| ④梯度(贪心温启动精调) | 49 | 2.788e-4 | 99.6% | 189.67(=贪心解,无增益) |

- **④验收失败,③胜出**:独立梯度被面积项拖进不可行区、对偶上升拉不回(50 slot 的
  重尾比值信用分配,STE 梯度弱启发——Codex 的 R3 风险兑现);贪心温启动精调也
  不能改进贪心解。**收益来源不是梯度,是"实测 Δmred 预言机"替换解析 wae 上界**:
  实测下大量 cell 在 MRED 轴近乎免费,解析保守性才是 GA 只敢放 7 个 cell 的根因。
- **待决**:189.67µm² 是 standalone cell 面积口径,**尚未 DC/XA 验证**(greedy pack
  RTL 已备好可直接送 DC);贪心+升级全程 ~15min GPU/零 EDA,入环需减开销
  (S3 子样本减半、增量重评估、只对池赢家求解)。若 DC 证实,07-09"cell 增量小"
  的裁决要改写为"解析保守性人为压制了 cell 收益"。

### 3.2.2 全 k 扫描(2026-07-10,`batch_solve.py`,verilator 16M 终验)

7 个有效 k 点(k02–k14)在各自 GA 最优结构/布线上同预算重解,资格带沿用 GA 当时的
`approx_max_col=16`(公平对齐);k16–k20 资格带 [k,16) 为空 → 三方皆纯截断(配置限制,
非 cell 无潜力,GA 当时 n_approx 也为 0)。仿真器 vs verilator 16M 全 k Δmed=0。

| k | 预算(MRED) | GA cell/省µm²/利用率 | 贪心 cell/省µm²/利用率 | 倍数 | ④>③ | GA_MED | 贪心_MED |
|---|---|---|---|---|---|---|---|
| 2 | 1e-7 | 1 / 6.7 / 90% | 8 / 17.6 / 100% | 2.6× | 否 | 2.1 | 3.9 |
| 4 | 4e-7 | 1 / 4.2 / 92% | 9 / 17.5 / 100% | 4.2× | 否 | 11 | 16 |
| 6 | 2.5e-6 | 2 / 10.8 / 98% | 19 / 27.7 / 100% | 2.6× | 否 | 61 | 179 |
| 8 | 1.3e-5 | 2 / 17.0 / 93% | 32 / 82.7 / 99% | 4.9× | 否 | 331 | 1334 |
| 10 | 6e-5 | 4 / 20.2 / 93% | 44 / 96.4 / 99% | 4.8× | 否 | 1831 | 6598 |
| 12 | 2.8e-4 | 7 / 24.9 / 99% | 49 / 189.7 / 100% | 7.6× | 否 | 7442 | 21757 |
| 14 | 1e-3 | 7 / 29.7 / 97% | 27 / 165.1 / 100%* | 5.6× | 否 | 37999 | 62133 |

**④独立梯度补测**(`grad_sweep.py`,不给贪心温启动、自己解 300 步,sim 16M=verilator):

| k | GA 省 | ④独立梯度 cell/省/利用 | ③贪心省 | 梯度/贪心 | 梯度 vs GA |
|---|---|---|---|---|---|
| 2 | 6.7 | 2 / 1.0 / 70% | 17.6 | 6% | **输 GA** |
| 4 | 4.2 | 1 / 0.5 / 91% | 17.5 | 3% | **输 GA** |
| 6 | 10.8 | 2 / 14.4 / 90% | 27.7 | 52% | 赢 GA |
| 8 | 17.0 | 6 / 17.3 / 98% | 82.7 | 21% | 平 GA |
| 10 | 20.2 | 5 / 24.0 / 50% | 96.4 | 25% | 赢 GA |
| 12 | 24.9 | 3 / 12.3 / 85% | 189.7 | 6% | **输 GA** |
| 14 | 29.7 | 2 / 11.1 / 96% | 165.1 | 7% | **输 GA** |

独立梯度均值仅贪心的 **~17%**,且 **4/7 点连 GA 都输**(梯度反而更差)。梯度不是"比贪心
略弱",而是**根本性不适配该问题**:重尾 MRED 比值损失下 STE 梯度弱、面积项主导→少 cell。

**结论**:
1. **贪心在 7/7 点碾压 GA**,倍数 2.6–7.6×(均值 ~4.6×),且预算利用率从 GA 的 90–99%
   推到 99–100%。规律稳定,非单点偶然。
2. **④梯度在 7/7 点都没超过③贪心**(grad_beats_greedy 全 False)。diffam 式梯度方案
   在该问题上确定性出局,cell 求解器形态定为贪心+升级扫描。
3. **收益根因复证**:GA 只放 1–7 个 cell,贪心放 8–49 个——差距全来自"实测 Δmred 打分"
   vs"解析 wae 上界"。解析保守性系统性压制 cell 采用,是 07-09"cell 增量小"的真因。

**必须标注的边界**:
- **贪心 MED 爆炸(每点 1.6–4× GA)**:本实验只对 MRED 设预算,贪心把误差堆到大乘积上
  (MRED 便宜、MED 贵)钻空子;GA 的 MED 反而常低于纯截断 floor。**面积收益乐观,
  真实应用若在乎 MED/WCE 需改用多指标预算重解**。
- **未过 DC/XA**:µm² 是 standalone cell 面积求和,非综合后真实 PPA;49 个近似 cell
  会否伤时序未知。RTL 已备(scratchpad `cellsolver_batch/k*/final_greedy/`)。
- k14 利用率 100.06%(*轻微越界 6e-4,修复阈值需收紧);高 k(≥16)dead-gate 需
  `approx_max_col=30`+window 才能测 cell 潜力(另一实验)。

> ⚠️ **上表倍数/结论已被 §3.2.3 真实 DC+XA 推翻——standalone 代理系统性高估,勿引用。**

### 3.2.3 真实 DC+XA 复核(2026-07-10,`analyze_dcxa.py`,EDA 机 202.120.39.27)

21 设计(k02–k14 × exact/GA/greedy)整网表送 DC 综合 + XA 功耗。**这是决定性反转:
standalone 代理只有 18–62% 兑现,greedy 招牌优势基本是加法代理的幻觉。**

| k | GA 真实省µm²/Δpow | greedy 真实省µm²/Δpow | greedy代理省→真实(兑现率) | 面积赢家 | 功耗赢家 |
|---|---|---|---|---|---|
| 2 | 8.2 / **−0.4%** | 7.7 / −3.9% | 17.6→7.7 (44%) | GA(0.9×) | greedy |
| 4 | 28.6 / **−4.6%** | 10.9 / **+2.4%** | 17.5→10.9 (62%) | **GA(2.6×)** | GA |
| 6 | 19.8 / **−6.4%** | 16.6 / −3.1% | 27.7→16.6 (60%) | GA(1.2×) | GA |
| 8 | 20.8 / **−2.2%** | 15.1 / **+0.4%** | 82.7→15.1 (18%) | GA(1.4×) | GA |
| 10 | 21.8 / **−3.8%** | 17.6 / **+3.6%** | 96.4→17.6 (18%) | GA(1.2×) | GA |
| 12 | 25.5 / −9.2% | **49.4** / −5.7% | 189.7→49.4 (26%) | **greedy(1.9×)** | GA |
| 14 | 22.0 / −5.4% | **35.8** / +1.2% | 165.1→35.8 (22%) | **greedy(1.6×)** | GA |

**真实裁决(推翻 §3.2.2)**:
1. **"贪心碾压 GA"是代理幻觉**。代理 2.6–7.6× → 真实:greedy 面积只在 k12/k14(深截断)
   赢,k02–k10 五点 **GA 反而省更多真实面积**;k12 招牌从 7.6× 缩到 1.9×,189µm²→49µm²。
2. **功耗上 GA 7/7 完胜**:GA 每点都降功耗(−0.4~−9.2%),greedy 有 3 点(k04/k08/k10)
   **反而升功耗**——MED 爆炸的代价在开关活动上真实兑现了。GA 的少而均衡 cell 更"干净"。
3. **时序无灾**:49 cell 的 k12 delay −1.43ns,全 21 设计时序都过,这一条担忧排除。
4. **原"cell 在 MRED 轴增量小"(07-09)经真实 PPA 复核成立**;greedy 未推翻它。加法代理
   的高估正是当初 GA 解析滤波"保守"的物理理由——DC 边界优化本就吃掉了大部分单 cell 增益。
5. **唯一残余正结论**:深截断(k12/k14)激进堆 cell 能换到真实面积(−5.7~−6.8% vs exact,
   胜 GA),但要付功耗代价(劣于 GA)——是一个 area/power 取舍旋钮,不是免费午餐。

**方法学收获**:standalone 单 cell DC 面积**相加**是系统性高估的代理(兑现率 18–62%),
不能作为 solver 目标;真实面积必须整网表 DC。这与 §3.2 Q1 裁决(功耗预测器 in-design
边际不可信)同源——cell 级量在设计级都被 DC 全局优化/布线重塑。

### 3.2.4 多架构公平复核(2026-07-10,`analyze_arch.py`,17 独立架构真实 DC+XA)

§3.2.3 把 greedy 放在 GA 自己进化出的最优结构上,对 GA 略偏袒。此处从 rerun 的
save_iter 检查点取 **多个不同结构**(k02–k14,iter39/59/99,greedy 在 gpu2 求解),
在**同一 slot 菜单**上 GA 包 vs greedy 包公平对打,合并 §3.2.3 的 7 finals,exact 面积
去重后 **17 个独立架构**,全部整网表 DC+XA(51 设计,21/21+30/30 成功)。

**结果(比 §3.2.3 更细,推翻"GA 全面更好"的过强表述)**:

| 维度 | GA 赢 | greedy 赢 | 规律 |
|---|---|---|---|
| 真实面积(DC,确定) | 8/17 | 9/17 | **强 k 依赖**:k02–k06 GA 赢(≈3/3);k12/k14 greedy 赢(**4/4**);k08/k10 交叉区(结构而定) |
| 真实功耗(XA,±9%噪声) | 12/17 | 5/17 | GA 占优但非全胜;greedy 有 **6/17 反升功耗** |

按 k 的面积赢家:k02 greedy 0/2、k04 0/1、k06 1/3、k08 2/4、k10 2/3、k12 **2/2**、k14 **2/2**。

**裁决(定稿,取代 §3.2.3 第 1–2 条的过强措辞)**:
1. **没有谁全面更好——是分区互补**。**深截断(k12/k14)greedy 面积稳赢(4/4 独立结构),
   浅截断(k02–k06)GA 赢**,交叉区在 k08–k10(结构而定)。§3.1 机制假设"深截断使设计
   小而集中→堆 cell 划算"被多结构证实。
2. **功耗 GA 仍整体占优(12/17)但非 7/7**:greedy 的 MED 爆炸功耗代价是**结构相关**的,
   不是普适(6/17 反升,但也有 k12_i39 greedy −13.5% 完胜 GA −6.3%)。小于 ~2% 的
   功耗差落在单 seed ±9% 噪声带内,不可判读;可判读的大差两个方向都有。
3. **实用结论**:若目标是**深截断区的面积**,greedy(实测打分)是更强工具;若要**低截断
   或功耗稳定**,GA(少而均衡 cell)更优。二者应按 k 分区选用,而非二选一。
4. 高 k 独立结构少(k12/k14 各仅 2 个,训练早收敛),面积趋势清晰但功耗统计力弱,
   结论 1(面积 k 依赖)比结论 2(功耗)证据更硬。

### 3.2.5 greedy 求解器训练接入与上线(2026-07-10)

**接入**(`--outer_cell_solver greedy`,默认 None=行为不变):
- 新 trainer 参数 `outer_cell_solver/outer_solver_vectors(16M)/outer_solver_cache`;
  `_cell_solver_active()` 门 = outer_cell_search + greedy + MRED 预算模式;
- `_outer_greedy_solve()`:get_samples 里 sample-0 发射后、DC 前(与 errgate 同钩子、
  solver 分支优先),在 sample-0 布线上建 TreeSim+GradientCellSolver → greedy_add →
  cfg 经 vertex_list 反查写回 `state["cells"]` → 重发射;异常/无 slot → 空 cells 放行;
- `reset()`:solver 模式下 `_outer_mutate` 后清空 cells(结构变异保留,cell 维度全权
  交求解器);每 episode 解一次,episode 内全样本共用;
- 冒烟 `scripts/smoke_outer_greedy.py`(A 开关口径/B 端到端状态同步+坐标反查+RTL/
  C 默认关回归)本地+远端全过。**坑**:远端 CPU 跑 torch 必须 OMP_NUM_THREADS=4~8,
  否则线程超订抖动假死(0% CPU,老 pwrpred 坑复发)。

**上线过程踩坑与修复(07-10 晚,三连击)**:
1. **PATH 毒点复发**:非交互 ssh 启动不带 vtool PATH → verilator 静默回退解析
   (`[errgate] ... No such file or directory: 'verilator'`),整批作废级事故;
   launch 脚本 env 加 `PATH=~/anaconda3/envs/vtool/bin:$PATH` 修复。
2. **跨布线越线**:密集包(54-65 cell)误差抵消强依赖 sample-0 布线,其余布线系统性
   偏高 10%+,margin=0.9 仍 3 个 k 首集全部 7/9 样本越线报废——margin 猜不动。
3. **鲁棒修复(终版)**:get_samples 先采完整集 8 条布线 → sample-0 解包(budget×0.9)
   → 张量化 sim 对**全部布线**复测(与 verilator 闸门同 16M 同流逐位一致 → sim 合规
   = 门必过)→ 任一布线超全额预算摘 wae·2^col 最大 cell 至全体合规 → 再发射。
   `_outer_greedy_solve_robust`;**over_budget=0 是构造保证**。
   代价(重要观察):全布线合规把包从 54-65 cell 削到 **11-16 cell**(摘 39-47)——
   离线单布线实验的密集包优势,在"整集共用包"约束下大幅缩水;剩余包仍 ~2× GA 规模。

**上线 run(终版)**:`outputs/2026-07-10_22_mred_greedy_np4`(远端 22:55 北京时间起),
k12/k14/k16,**逐字镜像 warm240eg 配方**(同 config max_col=30/window6/num_epochs=1、
同 seed42、同 warm 池 07-09_06、ep240/s8/np4)+ `--outer_cell_solver greedy
--outer_solver_margin 0.9`;**对照 = warm240eg 终局(已完成 ep240)**。
k12/k14→cuda:0,k16→cuda:2。首集:三 k 全部 **over_budget=0/9**、verilator 零失败;
robust 包 k12 15cell(worst_util 99.9%)/k14 11(85.8%)/k16 16(99.7%)。
注意:同机 epo4/noinh 系列(max_col=16、num_epochs=4)口径不同,不作对照。

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
outer_errgate: false            # §3.1 实测误差预筛门
outer_errgate_vectors: 2000000
outer_errgate_max_repairs: 6
```
`scripts/train_dc.py` 加 `--outer_cell_search` / `--outer_errgate`
（及 `--outer_errgate_vectors/--outer_errgate_max_repairs`）CLI 开关。

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
