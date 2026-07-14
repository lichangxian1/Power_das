# v5 单个 Episode 的完整生命周期（逐行对齐代码）

本文把 v5 训练中**一个 episode 从进入到退出的每一步**拆到函数级，所有行号对齐
当前 `trainer/arith_das.py`。宏观架构（两段进程、三口径、三维度协同）见
`V5_TRAINING_FLOW.md`；本文只往深处放大 §4"episode 生命周期"这一节。
术语首次出现时附中文解释。

对应运行中的战役 `2026-07-13_v5r2_np2`（r2 = 修复 r1 十项漏洞后冷启动的第二轮）：
`samples=32`、`bin_cap=8`、`num_epochs=1`——每集 1 个 PPO（Proximal Policy
Optimization，近端策略优化，一种约束"每步别改太猛"的主流强化学习算法）epoch
（一轮完整的梯度更新）、`save_freq=20`、菜单 = 统一 substd（sub-standard，自制
近似压缩器单元库，22/32/42 三族的面积功耗全面优于标准单元实现）、cell（近似
压缩器单元：真值表不完全正确、但面积功耗更省的压缩器替身）上限 64、近似列窗口 6。

---

## 0. 调用链总览

```
run_experiment()                            arith_das.py:2464
 └─ for episode_idx in range(480):
     ├─ run_episode(episode_idx)            arith_das.py:4036
     │   ├─ A. _v5_begin_episode()          :4288   本集编排（种子/轮询、伪预算）
     │   ├─ B. reset()                      :4487   取亲代 + 变异
     │   │    └─ _outer_mutate()            :1697   结构必做 + cell 骰子
     │   ├─ C. get_samples()                :3412   GCN 采样→RTL→oracle→DC+verilator
     │   │    └─ _outer_tt_oracle_screen()  :1874   sample-0 误差预筛/修剪
     │   ├─ D. update_found_best_info()     :3908
     │   │    └─ _v5_admit_samples()        :4345   支配准入（v5 的心脏）
     │   ├─ E. PPO 更新（run_episode 内联）  :4089–4128
     │   └─ 记录/调度器步进                  :4132–4134
     ├─ 每 5 集  _dump_front_snapshot()             轻量前沿快照（front_hist/）
     └─ 每 20 集 save_experiment()          :2352   权重 + front.json + front_state.json
```

一个 episode 的墙钟时间 ≈ 3.5–4 分钟，几乎全部花在 Step C 的 32 路并行
DC（Synopsys Design Compiler，商业逻辑综合工具：把 Verilog 编译成门级电路并
报面积/功耗）综合上——单个 DC ~200s；verilator（开源 Verilog 仿真器，用来测
真实误差）~3s，藏在 DC 时间里。

---

## 1. Step A — `_v5_begin_episode`：本集编排（:4288）

进 episode 第一件事是决定**本集要打哪个误差箱**、用什么状态起手。

### 1.1 种子集模式（种子队列非空时）

队列里存着本段的截断深度 k（最低 k 列部分积作废、用校正常数补偿，误差换面积的
最粗旋钮）列表；seg_lo/seg_hi（低/高误差段的两个并行训练进程）各自为
seg_lo: k2–k14、seg_hi: k12–k24。逻辑：

```
k = 队列[0]                    # 只 peek 不 pop（见下）
激活 k 的截断档                 # _activate_trunc_profile(k)，见 §2.2
if k 的模型误差 floor > 档案 mred 上限:
    出队跳过（这个 k 对本段没意义）
if 该 k 已连续失败 ≥3 次:
    出队放弃，log ERROR
state_override = 标准 Dadda 树（教科书压缩树构造法）+ 截断 k + 零 cell   # _v5_dadda_state(k) :4277
本集 bin = k 的误差 floor 落进的箱
```

（floor = 该 k 纯截断能达到的误差下限；MRED = Mean Relative Error Distance，
平均相对误差，本文说"误差"默认指它；档案上限 = 本段负责的 MRED 区间右端点。）

**peek 不 pop 是个防丢种子的设计**（r2 双评审修复 #3）：种子只有在 Step D
"至少 1 个样本成功入档"之后才真正出队（:4379–4384）。如果这一集 DC/verilator
全批失败，下一集会自动重试同一个 k，最多 3 次。否则一次集群抖动就会让某个
基线点永远缺席。

### 1.2 进化集模式（队列已空）

```
本集 bin = episode_idx % 箱数        # 箱轮询
```

轮询保证 14（或 10）个箱**每个都定期被当作目标**，低误差难啃的箱不会被
放弃——这是"整条前沿都有人管"的机制保证。

### 1.3 伪预算

```
mred_budget = mred_scale = 本集箱的上沿
```

v5 里标量目标不决定存活，但 PPO 仍需要一个标量奖励，其中误差项是铰链罚
`max(0, mred − budget)/scale`（铰链 = 低于门槛不罚、超出线性罚）。把 budget
和 scale 都设成箱上沿，罚项就变成无量纲的 `mred/上沿 − 1`，**任何箱里超标
10% 罚得一样重**——不用逐箱调参。

---

## 2. Step B — `reset`：取亲代 + 变异（:4487）

### 2.1 亲代来源（三层回退）

```
1. 种子覆写非空 → 用它（Dadda(k)，原样评估，不变异）
2. 否则从本集 bin 均匀取一个档案条目（箱空 → 最近非空箱；sample_parent）
3. 档案全空 → 回退旧初始池（里面有兜底 Dadda）
```

取出的亲代是 `payload`（档案条目随身携带的完整设计信息包，导出复现用的一切都
在里面）中 `ct` 字段的深拷贝：`{ct32, ct22, ct42, cells, k}` ——
压缩树三张矩阵（每列放几个 3:2 / 2:2 / 4:2 压缩器）+ 近似 cell 摆放表 + 截断深度。

### 2.2 k 线程化：`_activate_trunc_profile(k)`（:4223）

k 是**设计的属性**而非全局常量。激活某个 k 意味着切换四样东西：截断列的常数
部分积、校正常数 C\*、近似 cell 资格列窗、误差 floor。每个 k 的这套参数做一次
确定性蒙特卡洛（秒级）后**缓存**，之后零成本切换。k 不改压缩树结构——截断列
的压缩器照常实例化，靠 DC 综合的常数传播自动扫成零面积。

### 2.3 变异：`_outer_mutate`（:1697，仅进化集）

**第一层：结构变异，每集必做。** 从合法动作掩码（mask，标记当前哪些动作合法
的 0/1 向量）里随机选一个动作执行 `transition`：某列 +/− 一个 3:2 或 2:2
压缩器、FA↔HA（全加器↔半加器）互换、CT42（4:2 压缩器，一口气收 4 个同权输入）
升/降级。
之后 `_cells_prune_stale` 把因结构变化而失效的 cell 摆放清掉。
（为什么必做：v1.0 把结构变异也放进骰子，结构搜索强度掉到 40%，实测同误差
下功耗系统性劣化 ~16%——结构被饿着了。）

**第二层：cell 叠加层，四选一骰子**（概率 = `outer_p_struct/cell/resample/zero`
归一化）：

| 算子 | 动作 | 可行性检查 |
|---|---|---|
| `keep` | cell 维度不动 | — |
| `cell` | 50% 加一个 / 25% 删一个 / 25% 换一个 | 解析预算过滤 `_cells_budget_ok`，提议失败重试 `outer_proposal_retries` 次 |
| `resample` | 清空重摆 K 个（K 从 {0,1,2,4,8,16,32,64} 基数集采） | 同上 |
| `zero` | `zero-col`：最低未清列**整列**填恒零 cell（= 分数截断一步）；无列可清则 `unzero-col` 反向撤一列 | **跳过闭式过滤**（见下） |

**zero 算子为什么豁免预算过滤**：解析误差模型对边界列 ZERO 的偏置估计失真
3.7 倍，一阶过滤会在提议阶段就把密集 ZERO 包误杀——这正是旧 GA 最多放 ~7 个
cell、而离线贪心同预算能放 25–77 个的机制根源。zero 的可行性改由 Step C 的
TT oracle（实测）和 Step D 的支配准入（生死）裁决。

### 2.4 重建 episode 上下文

用变异后的 `state` 重建 `CompressorTree`、压缩器分配、`CompressorGraph`
（GCN——Graph Convolutional Network，图卷积神经网络，本项目策略网络的主体——
的输入图），并把 `cells` 的槽位坐标映射成本集图的节点号
（`_refresh_episode_cell_types`，正常应 0 条未映射）。

---

## 3. Step C — `get_samples`：采样、发射、预筛、测量（:3412）

整段在 `torch.no_grad()` 下执行（采样不需要梯度）。

### 3.1 GCN 前向 + 32 个布线样本

`get_Z_mat()` 把压缩树图过一遍 GCN，输出每个(级, 列)切片的连接 logits
（logits = 未归一的打分，softmax 后是概率）。然后循环 32 次
`sample_from_logits`：对每个压缩器的每个输入端口按概率采一个连接来源，
边采边把已占用的来源 mask 掉（保证合法布线），累计 `overall_log_prob`
（该样本被采出的对数概率，PPO 要用）。

**同一集 32 个样本共享同一个变异后结构和同一套 cell 配置，只有布线不同。**
外环模式下 cell 配置不进 log_prob——PPO 的信用分配**只含布线**这一件事
（:3432–3437）。种子集同理不做类型采样（r2 修复 #2：种子必须是纯截断，
保证基线可复现）。

### 3.2 RTL 发射

每个样本 `mul.emit_verilog(...)` 发射成 RTL 文件 `MUL-{i}.v`（RTL = 寄存器
传输级代码，就是 Verilog 源码），带三样东西：布线
assignment、cell_map（哪个节点用哪个近似 module）、截断信息（挂在 ct 上，
发射器读出后把截断列部分积置常数）。

### 3.3 sample-0 的 TT oracle 预筛（:1874，M2）

只对 sample-0 跑（TT = truth table 真值表；oracle = "神谕"，泛指能快速给出
准确判定的检验器。全部样本共享 cell 配置，测一个就够）：

1. 用 cellsolver 的张量化仿真器 TreeSim 在**真实布线**上实测本集 cell 配置的
   MRED——与 16M verilator 闸门同流逐位一致（oracle 说可行 ⇒ 闸门必过），
   但只需秒级。日志里的 `[tt-oracle] n_cells=.. mred_sim=.. limit=.. (util=..%)`
   就是这一步。
2. **上限 = 档案 mred 上限**（不是本集伪预算！）。超伪预算只是落进更松的箱
   竞争，不算废样本；只有超档案上限（无箱可落）才需要修。
3. 超上限时：按解析贡献 `wae·2^col`（wae = 该 cell 真值表的加权平均误差，
   乘上所在列的位权 2^col 得到它对总误差的贡献）降序排列 cell，
   **二分搜索最小前缀摘除量**
   （摘掉贡献最大的前 m 个），使实测 MRED 回到上限内；二分完仍超（误差
   非单调的罕见情形）则逐个再摘兜底。
4. 修剪结果写回 `state["cells"]` 并**重发射 sample-0 的 RTL**；后续样本自然
   沿用修剪后的配置。
5. oracle 抛异常 → 回退 errgate（verilator 快速门，若开）或直接放行。

### 3.4 all-exact 保底候选

sample-0 之后额外发射一个**同布线、全精确 cell** 的设计（:3493–3516），标记
`baseline_only`：参与 DC 测量和档案准入，**不进 PPO**。作用是防止 cell 采样
把最优点拖得比纯截断还差——纯截断版本永远在场竞争。

### 3.5 派发测量：33 个 RTL → 32 路 worker

`multiprocessing.Pool(32)` 跑 `parallel_simulate_worker`：每个 worker 对一个
RTL **同时**做两件事——送远端 DC 综合（面积/功耗代理，~200s）+ 本地 verilator
16M 向量误差仿真（真实 MRED/MED，~3s）。时钟固定 1.5ns（`fixed_target_delay`），
所以每个样本只有 1 个 target delay，任务数 = 33。

### 3.6 收结果与失败处理

| 失败类型 | 处理 | 后果 |
|---|---|---|
| DC 失败 | 该样本**踢出本批**（:3568–3584） | 不进 PPO、不进档案；日志 `[dc] x/33 已丢弃` |
| verilator 失败 | 保留样本，`measured_error=None` | objective 回退解析误差；**不入档案**（Step D 跳过）；回退率 >50% 时 ERROR 告警（闸门失真） |
| 整批 DC 全失败 | `run_episode` 直接 return（:4044–4048） | 本集不更新策略/档案，但 `scheduler.step()` 照走，保持学习率退火（按预定日程逐步调小学习率）对齐 |

每个存活样本算出标量 `objective`（见 §5.1）后返回。

---

## 4. Step D — `_v5_admit_samples`：支配准入（:4345）

v5 把旧的"标量 rank 挑一个 best"整个换成：**每个有实测 MRED 的样本都去档案
里闯一遍生死**。

对每个样本：

```
mred = measured_error["mred"]；None（verilator 失败）→ 跳过不入箱
payload = {布线, ct(含 k), cell 配置, DC 结果, 实测误差, ...}   # 可直接导出复现
    ├─ payload["ct"]["k"] = 当前 k          # k 随解存档
    └─ all-exact 样本的 cells 清空          # 状态里的 cells 不属于它，作亲代时不应复活
(ok, bin) = archive.add(mred, area, power, payload)     # utils/common.py:174
```

`archive.add` 内部（`ParetoArchive`，utils/common.py:131）：

1. `area≤0 或 power<0` → 拒收（异常测量，负功耗会让 ε 容差变负、重复点无限入档——r2 修复）；
2. mred 落对数箱；超档案上限 → 拒收；
3. 箱内支配判定：`a 支配 c` 当且仅当 a 面积不劣**且**功耗不劣（功耗差在
   `eps_power=1%` 内视为同值，由面积裁决），且至少一维严格更优；
4. 与已有解面积相同、功耗差在 1% 内 → 视为重复拒收；
5. 收入后剔除被它支配的旧解；箱内超 `bin_cap=8` → NSGA-II（一种经典多目标
   遗传算法，这里只借用它的拥挤度公式）拥挤度淘汰（按面积排序，两端极值免死，
   删中间最挤的）。

日志 `[v5] admit 5/33 (no-mred skip 0) -> bins [3] | archive=36 pts/24 bins`
读法：33 个候选里 5 个入档、落在 3 号箱、档案现有 36 条分布在 24 个非空箱。

种子集下若本集 `n_ok > 0`，此刻才把种子 k 正式出队（配合 Step A 的 peek）。

---

## 5. Step E — PPO 更新（run_episode :4089–4128）

### 5.1 标量奖励 `get_objective`（:3646）

```
objective = w_delay·delay/s_delay + w_area·area/s_area + w_power·power/s_power
          + w_viol · max(0, mred − 伪预算) / mred_scale        # 误差铰链
```

本战役权重 `w_delay/w_area/w_power = 2/2/1`，尺度 `s = 1.44 / 800 / 0.0107`
（把三项都压到 O(1) 量级）。误差铰链用 verilator **实测** mred（实测缺失时
该样本不罚误差项）。再次强调角色：这个标量只喂 PPO，**不参与档案生死**。

### 5.2 PPO 损失 `get_ppo_loss`（:3832）

对本批每个非 baseline 样本：

```
A_i = −(obj_i − mean(obj)) / (std(obj) + 1e-8)     # advantage 归一化
ratio_i = exp(new_log_prob_i − old_log_prob_i)      # 新旧策略概率比
loss_i = −min(A_i·ratio_i, A_i·clip(ratio_i, 1±0.2))
L_ppo = mean(loss_i)
```

- **advantage**（优势）= 这个样本比本批平均好多少。取负是因为 objective 越小
  越好。归一化让学习信号只反映"相对好坏"，不被误差绝对量级淹没（历史教训:
  不归一时策略会退化回全精确角落）。
- `new_log_prob` 是**当前参数**下重算的该布线被采出的对数概率——逐切片重放
  采样过程、同步维护 mask（:3857–3887）；外环模式下不含 cell 类型项。
- clip=0.2 是 PPO 的信任域：概率比超出 [0.8, 1.2] 的部分不再提供梯度，防单步
  跑飞。
- 本批只有 <2 个策略样本时跳过更新（归一化后无相对信号，A≡0）。

### 5.3 总损失与优化

```
L = 1.0·L_ppo + w_disc·L_discrete + 1e-2·L_rule + 0.1·L_delay
```

辅助项：`L_discrete`（推 logits 远离均匀分布，权重逐集递增）、`L_rule`
（布线合法性正则）、`L_delay`（延迟代理）。可微误差 surrogate（代理模型：用
可求导的近似公式替代不可导的真实误差，好让梯度能流过去）在外环模式下
**关闭**（cell 不由类型头采样，推了也白推）。

然后 `backward()` → **梯度裁剪覆盖全部被训参数**（GCN + 类型头 + cardinality
logits——控制"总共放几个近似 cell"的可学习分布参数；r2 修复 #7：原来只裁
GCN）→ Adam（自适应学习率的梯度优化器）step。本战役 `num_epochs=1`，即这
一套只跑一遍。最后 `scheduler.step()`（CosineAnnealing：学习率沿余弦曲线从
初值缓慢降到近零的退火日程）。

### 5.4 旧池同步（:4073–4084）

变异配置与 all-exact 配对候选各以其最优 objective 更新旧初始池——这是 v5
档案之外的兜底通道（档案空时 reset 的第三层回退用它），也让好的 cell 摆放
随状态在池里持久化。

---

## 6. Step F — 落盘

| 频率 | 动作 | 内容 |
|---|---|---|
| 每集 | `log_episode`（每 `log_freq` 集） | TensorBoard（训练曲线可视化工具）标量：objective、损失、mred、n_approx、n_over 等 |
| 每 5 集 | `_dump_front_snapshot` | `logs/front_hist/front_epNNNN.json`：档案轻量快照（不含 payload，KB 级，tmp+rename 原子写）。**回放可视化的数据源**（`scripts/gen_v5_front_viewer.py`） |
| 每 20 集 | `save_experiment`（:2352） | `save_iterN/`：GCN + 类型头权重、best_info.json、front.json（同构快照）；外加 log_dir 根下 `front_state.json`——**含 payload 的全量档案滚动存档**（原子写，崩溃后可收割） |
| 训练结束 | `export_front`（:4401） | 每个档案条目导成 `front/k{kk}_bin{NN}_{j}/`（MUL.v + best_info.json），目录名 k 开头兼容 XA（Synopsys 门级功耗仿真器，功耗的最终裁判口径）终审管线的 `k*` glob（文件名通配符匹配） |

注：5 集一档的 `front_hist` 是 2026-07-14 新加（`front_dump_freq=5`，
`--front_dump_freq` 可调），当前跑着的 v5r2 尚未部署，其颗粒度仍是 20 集。

---

## 7. 失败路径速查

| 场景 | 检测点 | 处理 | 会丢东西吗 |
|---|---|---|---|
| 整批 DC 失败 | run_episode :4044 | 跳过本集，LR 照步进 | 本集白跑，无污染 |
| 单样本 DC 失败 | get_samples :3568 | 踢出本批 | 该样本作废 |
| verilator 失败 | get_samples :3563 | objective 回退解析，不入档案 | 不会误入档（防解析低估骗进箱） |
| 种子评估全败 | begin/admit 配合 | peek 不 pop，下集重试 ≤3 次 | 3 败才放弃该基线点 |
| oracle 异常 | tt_oracle :1955 | 回退 errgate 或放行 | 最坏放过超标样本，Step D 兜底 |
| zero 无可操作列 | _outer_mutate :1726 | cell 维度不变，只做结构变异 | 无 |
| cell 提议 N 次不可行 | _outer_mutate :1750 | cell 维度不变 | 无 |
| DC 测出负功耗等 | archive.add :178 | 拒收 | 防档案被污染 |

---

## 8. 小术语表

- **episode（集）**：一次"取亲代→变异→采 32 布线→测量→准入→PPO 更新"的完整循环。
- **压缩树 / Dadda**：乘法器里把部分积逐列压缩的加法器网络；Dadda 是教科书构造法。
- **布线（routing）**：压缩器输入端口接哪些信号的具体连接方案——同一棵树不同布线，PPA 可差很多。
- **截断深度 k**：最低 k 列部分积不算了、用校正常数 C\* 补偿——误差换面积的最粗粒度旋钮。
- **ZERO cell**：输出恒零的"近似压缩器"= 把该位置部分积直接丢掉；整列填 ZERO 等价于把截断边界推进一列（分数截断）。
- **铰链罚（hinge）**：`max(0, x − 门槛)`,门槛内零罚、超出线性罚。
- **log_prob**：策略采出该动作序列的对数概率；PPO 用新旧 log_prob 之差算概率比。
- **advantage（优势）**：样本相对本批平均的好坏程度，决定梯度方向与力度。
- **支配（dominance）**：解 a 在所有目标上不劣于 b 且至少一项严格更优。
- **拥挤度（crowding distance）**：NSGA-II 的密度度量，删解时优先删挤在一起的。
- **MRED**：Mean Relative Error Distance，平均相对误差（|误差/真值| 的均值）；v5 全流程的误差指标，重尾分布，需 16M 向量才收敛。
- **PPO**：Proximal Policy Optimization，近端策略优化——带概率比裁剪的策略梯度强化学习算法，内环学布线用。
- **GCN**：Graph Convolutional Network，图卷积神经网络——把压缩树当图来编码的策略网络。
