# Arith 三阶段多目标搜索架构方案

> 本文描述的是“结构—Cell—连线”逐层收缩搜索空间的完整方案。它既是算法设计说明，
> 也是实现、恢复训练和解释结果时的统一口径；文中的“冻结”表示该阶段不再修改对应
> 基因，而不是把不同候选强行改成相同结构。
>
> 乘法器首先产生按位权排列的部分积（partial products，PP），再由多级压缩器把每列
> 的多行比特压缩成最终两行，最后交给加法器输出结果。这里可优化的对象有三个层次：
> 压缩器数量与布局决定“有哪些节点”，近似 Cell 决定“每个节点实现什么逻辑”，连线
> 决定“哪些信号送入哪些端口”。三者同时变化时，某次 PPA 改善很难归因，也很难设计
> 合法的杂交与缓存键，因此方案把它们依次放到三个阶段。
>
> 三个阶段并不是三次互不相关的搜索。Stage 1 的结构会成为 Stage 2 的 backbone，
> Stage 2 的结构与 Cell 组合会成为 Stage 3 的 baseline；所有阶段共享同一套
> Verilator/DC 评估口径，并在结束时合并到同一个全局 Pareto 档案。

## 1. 目标与确定参数

将当前 Arith 搜索改造成三个阶段：

1. 仿照原版 Arith 搜索 `ct22 / ct32 / ct42 / k`；
2. 使用标准代际 NSGA-II 搜索近似 Cell；
3. 冻结结构和 Cell，使用 PPO 优化连线。

最终目标是在 `delay <= 1.5` 的硬约束下，得到 `area / power / MRED` 的全局 Pareto 前沿。

| 项目 | 配置 |
|---|---:|
| 种群 P | 128 |
| 每代子代 Q | 128 |
| 单批 DC 候选 | 64 |
| Stage 1 完整代数 | 120 |
| Stage 2 完整代数 | 120 |
| Stage 3 PPO episode | 基础预算120（24个精英 × 5次）；当前主训练扩展为360（24 × 15） |
| 前沿快照频率 | 每5代/episode |
| Cell 杂交 | 方案 A：按列块杂交 |

“固定连线”是固定确定性的连线生成策略，而不是让不同结构共用同一份 wire 表。结构改变后，规范路由器为新结构重新生成唯一、合法的连线。

## 2. 核心名词与总体数据流

### 2.1 核心名词

| 名词 | 在本方案中的含义 |
|---|---|
| 部分积（PP） | 乘法器输入按位相乘后得到的比特。相同位权的 PP 落在同一列，列号越高，权重越大。 |
| 压缩树 | 用多级 HA、FA、CT42 把每列的多行 PP 和进位压缩为两行的有向无环图。这里的“树”是行业习惯称呼，实际结构不一定是严格的数学树。 |
| `CT22` / HA | 2:2 压缩器（Half Adder）。接收两个同列输入，产生一个本列 sum 和一个下一列 carry。 |
| `CT32` / FA | 3:2 压缩器（Full Adder）。接收三个同列输入，产生一个本列 sum 和一个下一列 carry。 |
| `CT42` | 4:2 类压缩器。它一次吸收更多同列输入，并向下一列产生两路进位；可减少级数，但面积、功耗和可布线性不一定总是更优。 |
| `k` | 截断边界。低于该边界的低位部分积按当前截断规则删除或常数补偿，因此 `k` 越大通常误差越大、硬件越小。 |
| Cell | 某个压缩器槽位采用的具体门级/RTL 实现。`EXACT` 保持精确真值表，近似 Cell 用可控误差换取 PPA，`ZERO` 是更激进的零化实现。 |
| 槽位（slot） | Cell 所附着的压缩器位置，通常由 `(stage, column, compressor_type, index)` 唯一标识。结构变化后，原槽位可能消失。 |
| 规范连线 | `CanonicalRouter` 按固定规则为给定结构生成的确定性合法连线。它是可重复的基准路由，不代表所有结构共用一张 wire 表。 |
| backbone | Stage 1 选出的结构骨架，只确定 `k` 和压缩器配置；Stage 2 在其上搜索 Cell。 |
| elite | 从某阶段 Pareto 档案中按误差区间和目标角色挑出的代表性候选。Stage 3 的 elite 已同时固定结构和 Cell。 |
| PPA | Power、Performance、Area 的统称。本项目把 delay 作为 performance 指标，并将 `delay <= 1.5 ns` 作为硬约束。 |
| DC | Synopsys Design Compiler。它负责把候选 RTL 综合映射到目标库，并给出 area、power、delay；这里所谓“DC 偏好”是综合映射对某类结构或连线的条件性响应。 |
| MRED | Mean Relative Error Distance，平均相对误差。它衡量近似乘法结果相对精确结果的平均偏差，跨多个数量级时使用对数坐标更合适。 |
| Pareto 支配 | 若候选 A 在 area、power、MRED 上都不差于 B，且至少一项更好，则 A 支配 B。互不支配的候选组成 Pareto 前沿。 |
| 活跃种群 / 外部档案 | 活跃种群只有128个，负责产生下一代；外部档案保存全历史非支配解，不等同于下一代亲代集合。 |
| 完整代 / episode | “完整代”指128个子代全部评估后再做一次环境选择；Stage 3 的一个 episode 指对一个固定 elite 采样并评估一批64条连线。 |
| `P` / `Q` / `R` | `P` 是当前亲代种群，`Q` 是本代新子代，`R=P∪Q` 是 NSGA-II 做环境选择时的临时合并集合。 |

除非特别说明，文中的 area、power、delay 都采用同一目标库和同一 DC 脚本口径，
MRED 采用同一批 Verilator 测试向量。评估口径不同的点不能直接放进同一 Pareto 前沿。

### 2.2 三阶段数据流

```text
Stage 1：结构搜索
结构和k可变；Cell全部EXACT；规范连线
        │
        ▼
选择32个结构 backbone
        │
        ▼
Stage 2：Cell NSGA-II
不执行结构变异；Cell可变；规范连线；120代
        │
        ▼
选择24个结构+Cell精英
        │
        ▼
Stage 3：PPO连线优化
结构和Cell冻结；仅连线可变
        │
        ▼
合并各阶段档案，输出全局Pareto前沿
```

每个阶段独立 checkpoint 和落盘，可单独恢复、复评或跳过。

### 2.3 阶段之间如何交接

阶段交接不是把上一阶段整个档案直接当成下一阶段亲代。Stage 1 先从结构档案中挑出
32个覆盖不同 MRED 区域和 PPA 偏好的 backbone；Stage 2 再围绕每个 backbone
构造4种 Cell 初始化，恢复为128个活跃候选。Stage 2 结束后只挑24个
“结构 + Cell”精英进入 Stage 3，PPO 以 round-robin 顺序逐个访问它们。

这种交接方式兼顾覆盖度和 DC 预算：外部档案可以很大，但真正参与下一阶段训练的
代表点数量固定。每个代表点都保留来源 ID，因此最终可以追溯某个 Stage 3 点来自
哪个 Stage 1 backbone、哪个 Stage 2 Cell 组合以及哪种 PPO 目标角色。

独立 checkpoint 表示 Stage 2 可以直接读取已经完成的 Stage 1 产物，Stage 3 也可以
从已有 PPO 模型、优化器动量、档案和随机数状态继续，而不必重跑前两阶段。阶段目录
仍共享评估缓存；只要设计哈希和评估环境指纹一致，重复候选就不会再次占用 DC。

## 3. 统一候选与缓存

`Candidate` 是遗传、缓存、档案和 checkpoint 共用的最小设计单位。一个 Candidate
必须包含足够信息，能够在不依赖其他候选可变状态的前提下独立生成 RTL、运行
Verilator 并提交 DC；否则并行评估时很容易出现一个候选覆盖另一个候选状态的问题。

简化字段如下：

```python
Candidate:
    candidate_id, parent_ids, stage
    k, ct22[], ct32[], ct42[]
    cell_map
    routing
    area, power, delay, mred
    valid, failure_reason
    rank, crowding_distance
    operator, operator_context
```
字段可按用途理解为：

| 字段组 | 作用 |
|---|---|
| `candidate_id / parent_ids / stage` | 标识候选、记录谱系和产生阶段，用于复现与收益归因。 |
| `k / ct22 / ct32 / ct42` | 结构基因，描述截断边界和各类压缩器配置。 |
| `cell_map` | 槽位到具体 Cell 型号的稀疏映射；未出现的槽位默认 `EXACT`。实现中也可序列化为排序后的 `cells` 列表。 |
| `routing` | 完整合法连线。为空时表示使用规范连线；Stage 3 候选必须保存实际采样连线。 |
| `area / power / delay / mred` | Verilator 和 DC 的测量结果，不能由候选 ID 推断或跨评估口径复用。 |
| `rank / crowding_distance` | NSGA-II 的临时选择属性。rank 越小越好；拥挤距离越大，说明附近候选越稀疏。 |
| `operator / operator_context` | 记录由哪个变异、杂交或 PPO 策略产生，以及 Bandit 上下文，便于审计。 |

`valid=false` 不等于“性能差”。它表示候选无法完成合法构造或可靠评估；
`failure_reason` 必须进一步区分设计错误和 DC 许可证、超时等基础设施错误。


必须提供稳定哈希：

```text
structure_hash = hash(k, ct22, ct32, ct42)
cell_hash      = hash(structure_hash, normalized_cell_map)
routing_hash   = hash(cell_hash, normalized_routing)
```

哈希用于候选去重和复用 DC、Verilator 结果。同一配置不能因 ID 或亲代不同而重复综合。

这里的“稳定哈希”不能直接使用受 Python 进程影响的内置 `hash()`。应先把数组、
Cell 槽位和 routing 边按固定顺序规范化序列化，再计算 SHA-256 等确定性摘要。
临时 ID、亲代、rank、日志字段和测量结果都不参与设计等价性判断。

三个哈希构成逐层依赖：结构改变时，旧 Cell 和旧 routing 都失效；Cell 改变时，
旧 routing 也不能仅凭 wire 编号复用。缓存键还要加入目标库、时钟约束、DC 脚本、
Verilator 向量集和近似 Cell 库版本等环境指纹，防止不同评估口径误命中。

# Stage 1：结构搜索

## 4. 搜索边界

Stage 1 只搜索：

```text
k
ct22[column]
ct32[column]
ct42[column]
```

`ct22[c] / ct32[c] / ct42[c]` 表示第 `c` 个权重列使用多少个对应压缩器；
后续的压缩器分配器会把这些列级计数展开为具体 stage 和槽位。它们描述的是
压缩能力和跨列进位关系，不包含近似 Cell 型号，也不直接指定 wire 连接。

`k` 与压缩器计数共同构成结构基因。增大 `k` 通常删除更多低位逻辑、降低
area/power，但也会提高 MRED；压缩器动作则在相同或相近误差边界下改变树深、
并行度和 DC 映射机会。Stage 1 同时搜索二者，是为了先建立覆盖整个误差范围的结构骨架。

所有 Cell 强制为 `EXACT`，所有候选使用规范连线，以隔离结构的真实性能贡献。规范路由器必须满足：

- 相同结构得到相同连线；
- 排序和 tie-break 固定；
- 随机种子不参与连线选择；
- 生成失败时标记非法，不能静默修改结构。

采用 `P=128, Q=128` 的 Pareto 种群，运行120个完整代，即15,360个子代评估；加上初始种群共需242个64候选批次。

## 5. 结构动作层级

当 variation controller 选择 `structure` 后：

```text
选择动作组
→ 选择当前合法动作类型
→ 从该类型的所有合法列中均匀选择
```

### 5.1 Classic 动作

```text
add_HA
remove_HA
FA_to_HA
HA_to_FA
```

`variation controller` 是变异调度器：它先决定本次修改结构、Cell 还是其他层；
进入结构层后，Stage 1 Bandit 只选择 `classic / ct42_local / ct42_repair / boundary_k`
这一级动作组。动作掩码随后列出当前候选真正可执行的动作和列，避免先选中非法操作再
依靠大量重试碰运气。

`legalize` 指在结构动作后恢复列高、进位和非负计数约束的修复过程。Classic 动作
可能引起后续列级联变化，因此沿用原版 Arith 的合法化；后面的 exact CT42 动作则
刻意设计成列高等价变换，从而避免不必要的全局修复。

复用原版 Arith 的动作语义、动作掩码和级联合法化。

### 5.2 CT42 精确动作

```text
promote_CT42_exact
demote_CT42_exact
relocate_CT42
```

| 类型 | 本列减少 | 下一列产生 |
|---|---:|---:|
| HA | 1 | 1 |
| FA | 2 | 1 |
| CT42 | 3 | 2 |

因此 `FA + HA` 与一个 `CT42` 对列高完全等价。

动作名中的 `exact` 表示“列高变化和跨列输出严格等价，动作本身不需要 repair”，
不是在描述 CT42 的 Cell 型号。Stage 1 所有压缩器都使用功能精确的 `EXACT` Cell，
这是另一层含义。

```text
promote_CT42_exact:
    FA[c] > 0 and HA[c] > 0
    FA[c] -= 1; HA[c] -= 1; CT42[c] += 1

demote_CT42_exact:
    CT42[c] > 0
    FA[c] += 1; HA[c] += 1; CT42[c] -= 1
```

两者不改变任何列的最终高度，不触发全局 legalize。

`relocate_CT42` 原子执行一次源列 demote 和目标列 promote：

```text
源列 c1：CT42 -> FA + HA
目标列 c2：FA + HA -> CT42
```

先均匀选择 `CT42[c1] > 0` 的源列，再均匀选择 `FA[c2] > 0 and HA[c2] > 0` 的目标列。该动作保持 CT42 总数不变，专门优化 CT42 的位权位置。

### 5.3 CT42 带修复强动作

```text
insert_CT42_repair
delete_CT42_repair
```

`insert_CT42_repair` 直接执行 `CT42[c] += 1`；`delete_CT42_repair` 执行 `CT42[c] -= 1`。随后只在 `c...c+W` 的局部窗口内重新求解 FA/HA。

每列必须满足：

```text
remain[c] =
    pp[c]
  + FA[c-1] + HA[c-1] + 2*CT42[c-1]
  - 2*FA[c] - HA[c] - 3*CT42[c]

remain[c] in {1, 2}
FA[c], HA[c], CT42[c] >= 0
```

建议采用 `W=3~6` 的局部动态规划，最小化：

```text
sum(|delta_FA| + |delta_HA|)
+ lambda * changed_column_count
```

`remain[c]` 是处理完第 `c` 列后仍留在本列、将进入最终两行加法器的信号数；
`{1,2}` 表示该列已经被压到不超过两行。`pp[c]` 是截断和校正常数处理后的
初始列高，前一列产生的 carry 会成为本列输入。

这里的局部 DP（dynamic programming）在窗口内枚举可行的 FA/HA 计数组合，并把
窗口入口和出口 carry 当作边界状态。目标函数优先少改压缩器、少动列，因此
`insert/delete_CT42_repair` 是受控的大步探索，而不是借全局重排生成一个完全不同的结构。

约束：

- 最后一列禁止加入 CT42；
- 最多改动6个 FA/HA；
- 窗口末端 carry 必须与窗口外原边界一致；
- 无法局部闭合时动作非法；
- 修复后必须通过 `compressor_assignment_fused()`；
- 不能用全局 legalize 隐藏大范围级联。

### 5.4 k 动作

```text
increase_k
decrease_k
```

`k` 属于结构基因，默认每次改变 `±1`；`±2` 可作为低概率强变异，但不为不同步长增加独立动作类型。

### 5.5 变异强度

```text
80%：执行1个结构动作
15%：连续执行2个结构动作
 5%：连续执行3~4个结构动作
```

每一步后重新计算合法掩码。任一步失败都回滚整个复合变异。不增加 `promote_2_CT42` 等重复动作。

## 6. Stage 1 Thompson Bandit

Bandit 先选择动作组：

```text
classic
ct42_local
ct42_repair
boundary_k
```

Thompson Bandit 是在线的“变异预算分配器”，不负责选择亲代，也不替代
NSGA-II。每个动作组是一条 arm，维护一个 `Beta(alpha, beta)` 成功率后验；
每次从所有当前合法 arm 的后验中各抽一个成功率，选择抽样值最大的动作组。
`Beta(1,1)` 表示开始时对0到1之间的成功率没有偏好。

上下文把 MRED、CT42 密度和 `k` 离散成若干区域。同一个动作在低误差区有效，
并不意味着它在高截断区也应获得相同概率，因此各上下文分别累计成功和失败。
3% 强制探索防止早期少量偶然结果永久压制某条 arm。Bandit 只选动作组；
具体动作类型和列仍由合法掩码过滤后在组内选择。

Bandit 奖励是延迟的：子代完成 Verilator/DC 并参加本代环境选择后，才知道是否
进入 `P_{t+1}`。许可证失败、超时等没有反映算子质量，既不算成功也不算失败。

当前每个 `(context, arm)` 只统计最近128个可归因子代，形成滑动窗口；旧经验会逐渐退出，
使算子概率能够跟随种群所处区域变化。结构合法但性能差且未进入下一代记为失败，
基础设施失败则完全不进入这个窗口。

## 7. Stage 1 输出

从外部结构 Pareto 档案选择32个 backbone：划分8个 `log10(MRED)` 区间，每区选择 area 最小、power 最小、局部 knee、结构新颖度最高各一个。不足时按拥挤距离补齐。

`log10(MRED)` 分区按误差数量级而不是线性距离切分，避免高误差区域占满名额。
“局部 knee”是该区间内同时兼顾 area、power 和 MRED、靠近归一化理想点的折中解；
“结构新颖度”优先选择压缩器分布或结构哈希与已有代表不同的候选。32个名额来自
`8个区间 × 4种角色`，若去重或空区导致不足，再用拥挤距离大的点补齐覆盖。

# Stage 2：近似 Cell NSGA-II

## 8. 冻结边界和初始化

Stage 2 不再执行结构动作。种群可包含多个 backbone，但每个子代的 `k / ct22 / ct32 / ct42` 必须从一个亲代完整继承，不能逐列拼接。

这里的 Cell 不是任意标准单元，而是 HA、FA 或 CT42 压缩器实例可选的
精确/近似 RTL 实现。一个槽位由 `(stage, column, compressor_type, local_index)`
标识；Cell 只能放入接口和压缩器类型兼容的槽位，`ZERO` 也必须是库中定义的合法
常量化近似实现，不能等价成随意删线。

冻结结构意味着子代可以选择来自哪个 backbone，但不能把两个 backbone 的
`ct22/ct32/ct42` 按列拼成第三种树。这样 Stage 2 的收益可以归因于 Cell 组合，
槽位映射也始终有一个完整 host 结构作为参照。

32个 backbone 各产生4个种子：

```text
全 EXACT
低近似密度
中等近似密度
分层随机近似
```

共128个。所有随机 Cell 必须与压缩器类型和槽位兼容，并继续使用规范连线。

## 9. 标准代际和预算

NSGA-II 是不依赖固定加权和的多目标遗传算法。繁殖时先做二元锦标赛：
非支配 rank 较小的候选优先；rank 相同时，拥挤距离较大的候选优先，以保留
目标空间中较稀疏的区域。选出的亲代经过杂交和变异，完整产生 `Q_t`。

环境选择发生在整代末尾：把旧种群 `P_t` 和全部子代 `Q_t` 合成 `R_t`，
按非支配层从前到后装入下一代；最后一层放不下时，再按拥挤距离截断。
因此任何新子代都不能在本代尚未完成时立即充当亲代，不会产生批次顺序偏差。

```text
P_t = 128
通过锦标赛产生 Q_t = 128
Q_t 分两批进入DC，每批64
R_t = P_t union Q_t = 256
非支配排序 + 拥挤距离
得到 P_{t+1} = 128
```

运行120个完整代。初始种群也需要评估：

```text
128 + 120 * 128 = 15,488 次评估
2 + 120 * 2 = 242 个64候选批次
```

每代两批必须全部评估后才能统一选择下一代并更新 Bandit。

## 10. 约束支配

`delay <= 1.5 ns` 在这里是硬门槛，而不是与 area、power、MRED 并列的第四个
可自由交换目标。约束支配先比较“能否满足时序”，再比较多目标性能，避免一个
面积很小但严重超时的候选挤掉所有可实现设计。

不可行候选没有被立即全部删除，是因为轻微超时的结构可能通过后续 Cell 变异重新
进入可行区。`timing_violation` 是无量纲的相对超额量：delay 等于1.65 ns时取0.1。
对 MRED 取对数只用于计算多样性距离，不会改变谁支配谁。

1. 可行解支配不可行解；
2. 可行解按 area、power、MRED 比较；
3. 不可行解优先保留约束违反量较小者。

```text
timing_violation = max(0, delay / 1.5 - 1)
```

非支配排序使用原始 MRED；拥挤距离使用 `log10(MRED + eps)`。

## 11. 方案 A：按列块杂交

```text
parent A: AAAAA | AAAAA | AAAAA
parent B: BBBBB | BBBBB | BBBBB
child:    AAAAA | BBBBB | AAAAA
```

图中的 A/B 只表示 Cell 配置来源，不表示把两棵压缩树逐列拼接。先完整选定
一个亲代作为 host，子代的结构和槽位集合完全由 host 决定；donor 只提供被选
列块中的 Cell 型号。类型不兼容或在 host 中不存在的 donor 槽位不能强行映射。

使用 `column-k` 而不是绝对列号，是为了按“距离截断边界多远”对齐两个亲代。
例如两个结构的 `k` 不同，它们的第一个保留列仍可视为同一相对位置。这样交换的
更像是低位近似策略、中位近似策略等功能片段，而不是偶然相同的物理列号。

规则：

1. 子代以50%概率完整继承 A 或 B 的结构；
2. 结构来源亲代为 host，另一个为 donor；
3. 在相对坐标 `column-k` 上选择1到3个连续块；
4. donor Cell 只映射到 host 中存在且类型兼容的槽位；
5. 无法映射的 Cell 丢弃，空缺位置保留 host 或回退 EXACT；
6. 杂交后检查合法性，但不得改变结构。

推荐 `p_crossover=0.9`。

## 12. Cell 变异和 Bandit

方案 A 属于标准繁殖流程，不作为 Bandit 臂。杂交后由 Thompson Bandit 为每个子代选择一个主变异：

| 动作 | 含义 |
|---|---|
| `cell_add` | 把一个合法 EXACT 槽位换成支持的近似 Cell |
| `cell_remove` | 把一个近似 Cell 恢复成 EXACT |
| `cell_swap` | 替换型号，或在兼容槽位间移动近似 Cell |
| `cell_resample` | 对连续2到5列重新采样，并保持相近近似密度 |
| `zero_toggle` | 在合法位置开启 ZERO，或把 ZERO 恢复成 EXACT |

选择 `cell_resample` 时不叠加其他 Cell 变异。

Bandit 配置：

```text
先验：Beta(1,1)
统计窗口：最近128个可归因子代
强制探索：3%
上下文：log10(MRED) 区域
成功：子代进入本代 P_{t+1}
```

按子代更新，不能用“本批是否有任意档案命中”做奖励。Rank-2/3 子代只要通过环境选择进入下一代也算成功。DC 基础设施失败不更新。

## 13. 外部档案和 Stage 2 输出

外部档案与128个活跃种群分离：活跃种群负责繁殖，档案保存全历史非支配解；按 `cell_hash` 去重，可用 epsilon 网格限制规模。

外部档案并不是“所有曾经评估过的子代”。新候选进入后，若它支配旧点，旧点会被
删除，因此档案条数可能增加也可能减少；相同目标坐标还可能在图上完全重合。
`epsilon` 网格把目标空间划成小格，每格只保留代表点，用有限内存近似稠密前沿。

每5代保存的前沿快照记录的是该时刻的累计档案，不是第5代刚产生的128个子代。
HTML 应按真实保存粒度回放，不能用最终档案反推不存在的逐代历史。

Stage 2 结束后按8个 `log10(MRED)` 区间选择24个精英，每区选择 area 最小、power 最小和局部 knee。必须保存每个精英的规范连线评估作为 Stage 3 baseline。

# Stage 3：PPO 连线优化

## 14. 冻结边界、状态与动作

Stage 3 冻结 `k / ct22 / ct32 / ct42 / cell_map`，也就是不再增删压缩器、不再更换 Cell，只允许把合法信号分配到既有压缩器输入端口。因此，此阶段优化的是同一份“结构＋Cell 配置”的物理实现机会，而不是重新做结构搜索。

PPO 把一张完整连线表看成一串受约束的端口分配动作：策略依次选择尚未分配的信号应接到哪个压缩器端口；所有动作完成后，才生成可综合 RTL，并由 Verilator 和 DC 返回 MRED、area、power、delay。DC 不可微，策略不能直接从综合器获得梯度，只能根据整条连线最终得到的标量奖励，提高较优动作序列再次出现的概率。

| 名词 | 在本阶段中的含义 |
|---|---|
| 状态（observation） | 当前压缩图、待分配信号、已有部分连线、端口占用，以及描述结构和 Cell 的特征 |
| 动作（action） | 把当前信号接到一个具体、合法的压缩器输入端口 |
| 动作掩码（action mask） | 在采样前把非法动作概率置零，保证策略只在当前合法集合内选择 |
| 一条 route | 从第一个动作到最后一个动作得到的一张完整 wire 表 |
| episode | 固定一个 Stage 2 精英，用其所属策略采样并评估64条完整 route，然后更新一次策略 |
| baseline | 同一结构、同一 Cell 配置使用规范连线得到的实测结果，用于衡量 PPO 连线带来的相对收益 |

动作掩码需要排除位权不匹配、端口已占用、形成组合环、违反 stage 顺序、CT42 端口不兼容，以及会导致剩余信号无法完成分配的动作。掩码只保证合法性，不替策略决定哪个合法端口更有利于 PPA。

### 14.1 目标状态与当前实现边界

目标状态至少应包含：当前 stage/column、信号来源与位权、到达时间、候选压缩器类型、端口占用、局部扇出、`k`、Cell 型号或嵌入，以及 Cell 的误差、面积、功耗和延迟描述。这样同一个共享策略才有条件区分“拓扑相近但截断边界或近似 Cell 不同”的候选。

当前已运行版本的 GCN 节点特征主要是 `stage / column / local index` 和
`PP / HA / FA / CT42` 类型；具体近似 Cell 只在 RTL 生成时生效，`k` 也没有作为
独立特征输入。因此，当前结果应解释为“共享 GNN 在固定24个精英上的在样连线优化
基线”，可以验证是否存在可利用的路由偏好，但还不能据此声称策略已经学会适用于
任意乘法器结构的普遍 DC 偏好。

## 15. 三个 PPO 策略

PPO（Proximal Policy Optimization）是限制单次策略变化幅度的策略梯度方法。这里不让一个标量奖励同时承担所有 Pareto 方向，而是训练三个独立策略：每个 `log10(MRED)` 区间交接的 area、power、knee 精英分别进入对应策略，所以每套策略覆盖8个精英，而不是三套策略都重复训练全部24个精英。

```text
PPO-Area：  area 0.60, power 0.20, MRED 0.20
PPO-Power： area 0.20, power 0.60, MRED 0.20
PPO-Knee：  area 0.35, power 0.35, MRED 0.30
```

以每个 Stage 2 精英自己的规范连线为 baseline：

```text
delta_A = (A0 - A) / A0
delta_P = (P0 - P) / P0
delta_M = log(MRED0 / max(MRED, eps))

reward = wA*delta_A + wP*delta_P + wM*delta_M
       - timing_penalty - mred_band_penalty
```

`A0 / P0 / MRED0` 是该精英的 baseline，`A / P / MRED` 是当前 PPO route 的结果。前三项为正表示相对 baseline 有改善；`timing_penalty` 惩罚超出延迟约束的实现，`mred_band_penalty` 防止策略为了极端 PPA 偏离该精英原本负责的误差区域。

每个精英内部独立做 advantage normalization。这里的 advantage 是某条 route 相对同批候选的好坏程度；将同一精英的64个奖励中心化并缩放后再更新，可以避免大结构与小结构的绝对 PPA 尺度互相污染。不同精英仍共享其 role 对应的策略参数，因此依旧需要监测结构间的梯度冲突和遗忘。

`clip=0.2` 的本意是用旧策略与新策略的动作概率比值限制更新幅度。但当前实现对每批
样本只立即训练1个 epoch，计算损失时新旧策略几乎相同，clip 实际触发很少；从优化
行为看，它更接近“批内归一化的 on-policy REINFORCE / contextual-bandit policy
gradient”。若后续要严格发挥 PPO 的 clip 机制，应冻结采样时的旧策略数据，同一批
训练2到4个 epoch，并同时记录 KL、entropy 和 clip fraction。

## 16. PPO 预算与 baseline 保护

基础预算采用24个精英、每个5个 episode、每个 episode 64条候选连线：

```text
24 * 5 = 120 个全局 episode
24 * 5 * 64 = 7,680 次候选评估
```

当前主训练在保留前120个 episode 及 PPO checkpoint 的基础上，又为每个精英追加10个 episode。因此完整 Stage 3 计划为：

```text
24 * 15 = 360 个全局 episode
24 * 15 * 64 = 23,040 次候选评估
```

全局 episode 编号每处理一个精英递增一次，不是24个精英各自同时增加一轮。round-robin（轮询）按固定顺序在24个精英之间切换，使每个精英获得接近相同的训练次数，也避免某一结构连续占用全部 DC 预算。

当前基线参数为 `clip=0.2`、`learning rate=1e-4`、每批1个更新 epoch、`max grad norm=0.5`。`max grad norm` 会裁剪异常大的梯度，减少单个 DC 离群结果引发的剧烈参数跳变。

Stage 2 baseline 必须永久保留在最终档案中。PPO 结果只有在统一约束下真正支配 baseline 时，baseline 才能被 Pareto 规则淘汰；如果 PPO 没学到有效连线，最终结果至少不会比 Stage 2 的规范连线退化。

# 工程实现

## 17. 模块划分

下面的目录表示逻辑职责，而不要求每项职责都必须拆成一个 Python 文件：

```text
trainer/arith_three_stage/
    candidate.py
    canonical_router.py
    evaluator.py
    constraints.py
    pareto.py
    archive.py
    checkpoint.py
    architecture_search.py
    structure_actions.py
    structure_repair.py
    cell_nsga2.py
    cell_crossover.py
    cell_mutation.py
    thompson_bandit.py
    routing_env.py
    routing_ppo.py
    runner.py
```

当前实现做了合并：`pareto.py` 同时承担非支配排序和外部档案，`selection.py` 承担
锦标赛、环境选择和分区交接，`cell_ops.py` 承担 Cell 种子、杂交和变异，`bandit.py`
实现两阶段独立的 Contextual Thompson Bandit，`runner.py` 统一负责阶段调度、
checkpoint、前沿快照和 Stage 3 策略更新。目录图描述的是可继续拆分的架构边界，
不应误解成当前磁盘上已经存在全部同名文件。

| 主要模块 | 责任 |
|---|---|
| `candidate.py` | 定义跨三阶段通用的 Candidate、稳定哈希和序列化 |
| `canonical_router.py` | 根据当前结构确定性生成合法规范连线 |
| `evaluator.py` | 把 Candidate 转成 RTL，调用 Verilator/DC，并管理评估缓存 |
| `structure_actions.py` | Stage 1 结构动作、合法化和局部 repair |
| `cell_ops.py` | Stage 2 Cell 种子、方案 A 杂交及五类变异 |
| `selection.py` / `pareto.py` | NSGA-II 排序、环境选择、分区选点和历史非支配档案 |
| `runner.py` | 三阶段状态机、并行评估批次、恢复、快照和最终导出 |

可以复用原版 Arith 的动作掩码、结构合法化、压缩器分配和 RTL 生成，但不能让多个候选继续共享一个可变的全局 `self.state`。每个 Candidate 都应能独立重建完整状态，否则并行 DC 返回顺序不同就可能把一个候选的结构或 Cell 串到另一个候选上。

## 18. 统一评估与失败分类

所有阶段使用同一条评估流水线：先检查结构与槽位合法性，再生成规范或 PPO 连线，随后生成 RTL；Verilator 做功能仿真并统计 MRED，DC 综合得到 area、power、delay，最后将结果写入缓存。统一流水线的意义是让三个阶段的点可以直接进行 Pareto 比较，避免因为脚本、约束或单位不同制造“假改进”。

失败类型至少区分为：

| 类型 | 含义 | 是否归因给搜索算子 |
|---|---|---|
| `design_invalid` | 结构、槽位或连线本身不合法 | 是；说明这次变异没有产生可实现设计 |
| `rtl_generation_failed` | 合法候选无法被当前 RTL 生成器表达 | 通常是实现缺陷，需单独排查 |
| `verilator_failed` | 编译或仿真失败，无法获得可靠 MRED | 先区分设计错误与工具错误 |
| `dc_compile_failed` | DC 无法分析或综合该 RTL | 仅确认是设计原因时才归因 |
| `dc_license_failed` | 许可证暂时不可用 | 否；应重试，不更新 Bandit |
| `timeout` | 在预算时间内未完成 | 需区分设计复杂度超限与基础设施阻塞 |

只有能够归因于候选本身的失败才可以计作遗传算子或 Bandit 的失败。DC 许可证、机器故障、进程被杀等基础设施问题不代表算子质量差；把它们写成失败样本会让 Bandit 错误地降低某类动作的选择概率。

缓存命中必须同时满足完整设计哈希和评估环境指纹一致。环境指纹至少包括位宽、Cell 库版本、RTL 生成规则、DC 约束、工具版本和 MRED 测试口径；否则旧结果不能安全复用。

## 19. 阶段产物、快照与恢复

新建训练目录时，目录名必须包含北京时间的日期和小时，至少采用 `YYYY-MM-DD_HH`，例如 `outputs/2026-07-21_08_arith_three_stage/`。这样同一天启动的多次训练不会覆盖或混淆。

当前实现的核心产物如下：

```text
run_dir/
    three_stage_config.json
    evaluation_cache.sqlite
    stage1/
        population.json
        archive.json
        bandit.json
        checkpoint.pt
        backbones_32.json
        front_hist/front_gen*.json
    stage2/
        population.json
        archive.json
        bandit.json
        checkpoint.pt
        elites_24.json
        front_hist/front_gen*.json
    stage3/
        checkpoint.pt
        front_hist/front_gen*.json
    final/
        pareto.json
        pareto.csv
```

`checkpoint.pt` 与 `front_gen*.json` 的用途不同：

- **checkpoint** 是恢复训练的完整状态。Stage 1/2 保存完整代编号、种群、档案、
  Bandit 后验及 Python/NumPy/PyTorch 随机数状态；Stage 3 还保存全局 schedule
  index、三套 PPO 参数、三个 optimizer 状态、档案和随机数状态。因此从 Stage 3
  checkpoint 续训会保留之前已经学习到的 PPO 权重和 Adam 动量。
- **前沿快照** 是轻量可视化数据，只保存该真实保存时刻的累计前沿。默认每5个完整 generation/episode 产生一个独立文件，HTML 的进度条应只停在这些真实刻度；它不包含种群、Bandit、优化器或随机数状态，不能拿来恢复训练。
- **最终档案** 是运行结束时的经验 Pareto 集，供 CSV 分析和 HTML 可视化使用；当前 HTML 由 `visual` 脚本在运行目录中另外生成，不是训练 checkpoint 的组成部分。

当前实现只在一整代或一个 Stage 3 episode 完成后写 checkpoint，所以恢复点不会落在半个64路 DC 批次中。若以后支持批次中途保存，必须额外记录已完成候选、待提交候选和重试计数，才能避免恢复后重复或跳过任务。

“原子保存”是先把完整内容写入同目录临时文件，确认写完后再用 rename 替换正式文件。即使训练在写盘时中断，也只会保留上一份完整 checkpoint，而不会留下看似存在但内容只有一半的文件。

## 20. 最终档案与验证

最终档案合并 Stage 1 结构解、Stage 2 规范连线 baseline 和 Stage 3 PPO 解，再统一应用延迟约束与 Pareto 筛选。这里的“全局 Pareto”是指三个阶段所有已评估候选合并后的经验非支配前沿，并不表示已经从数学上证明找到了整个乘法器设计空间的全局最优。

每个点至少保留 `source_stage / backbone_id / cell_parent_id / ppo_policy / operator_history`。这些 provenance（来源追踪）字段可以回答一个最终点由哪份结构骨架、哪次 Cell 演化及哪套路由策略产生，也能拆分三个阶段各自带来的收益。

必须覆盖的测试：

- exact promote/demote 前后列高完全一致且互为逆操作；
- relocate 前后 CT42 总数不变；
- repair 后所有列高度为1或2且槽位分配成功；
- NSGA-II 每代严格128个子代，两批完成后才选择；
- Stage 1/2 的120代不把初始评估计作第1代；
- 三个 Stage 每5代/episode 原子保存一份轻量前沿快照；
- Bandit 在环境选择后按子代更新，基础设施失败不更新；
- Stage 3 全程结构哈希与 Cell 哈希不变；
- PPO 失败时 Stage 2 baseline 仍在最终档案；
- 从 checkpoint 恢复后，代号、随机状态、Bandit/PPO 参数和档案连续，不覆盖已有快照。

对准备正式报告的最终少量候选，建议清空结果缓存后独立重复 Verilator 与 DC，并记录约束文件、库版本和随机种子。这样可以排除缓存污染或环境漂移造成的虚假 Pareto 改进。

## 21. 实施顺序

1. Candidate、稳定哈希和统一 Evaluator；
2. 确定性 CanonicalRouter；
3. Classic、exact CT42、relocate 动作；
4. 局部 DP repair 与 CT42 insert/delete；
5. Stage 1 种群、档案和 Bandit；
6. Stage 2 标准代际 NSGA-II；
7. 方案 A、5个 Cell 变异及延迟 Bandit 更新；
8. RoutingEnv 与三个 PPO 策略；
9. 全局 Pareto 合并和 HTML 可视化。

该顺序按依赖关系排列：先保证 Candidate、哈希和评估可复现，再验证结构与 Cell 搜索，最后接入成本最高、信用分配最困难的 PPO。每一步都应先通过小规模合法性、可复现性、缓存和 checkpoint 测试，再投入完整 DC 预算。
