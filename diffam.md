当前 DiffAM 可以分成两个层次理解：

1. `DiffAMStage2Search.propose()`：输入固定结构，输出一批尚未经过 DC 的离散 cell 候选。
2. Stage2 Runner：把这些候选送入 DC/Verilator，最终输出24个 Stage2 elites。

最核心的变化是：

```text
Stage1结构 Candidate
    ↓ 转成固定计算图
每个合法压缩器位置建立 cell 选择 logits
    ↓ STE梯度优化
产生几十个离散 cell 配置
    ↓ 软件 MRED 筛选、去重
输出 Stage2 Candidate
    ↓ DC + Verilator
得到真实 area/power/delay/MRED
```

---

## 一、DiffAM 的直接输入是什么

DiffAM 的直接入口是：

```python
searcher.propose(
    backbones,
    size=128,
    round_index=generation,
    excluded_hashes=seen_hashes,
    warm_starts=population,
)
```

对应 [diffam_search.py](/home/lichangxian/Power_das/trainer/arith_three_stage/diffam_search.py:410)。

它有5类显式输入。

### 1. 固定结构 `backbones`

`backbones` 是 Stage1 输出的 `Candidate` 列表，正常情况下是32个不同结构。

每个 backbone 主要包含：

```python
Candidate(
    k=...,
    ct22=[...],
    ct32=[...],
    ct42=[...],
    cells=[...],
    metadata={
        "mred_band": ...
    }
)
```

字段含义如下：

| 字段 | 作用 | DiffAM 是否允许改变 |
|---|---|---:|
| `k` | 截断参数/截断结构 | 否 |
| `ct22` | 各位置的 2-2 compressor 数量/结构 | 否 |
| `ct32` | 各位置的 3-2 compressor 数量/结构 | 否 |
| `ct42` | 各位置的 4-2 compressor 数量/结构 | 否 |
| `cells` | 各 compressor 使用的近似 cell | 是 |
| `metadata["mred_band"]` | 决定本轮使用哪个 MRED budget | 间接使用 |

`Candidate` 的定义在 [candidate.py](/home/lichangxian/Power_das/trainer/arith_three_stage/candidate.py:17)。

结构由下面这些内容唯一确定：

```python
structure_hash = hash({
    "k": k,
    "ct22": ct22,
    "ct32": ct32,
    "ct42": ct42,
})
```

也就是说，DiffAM 接收的不是“待训练神经网络输入”，而是一批已经确定 compressor tree 结构的乘法器。

### 2. 希望输出的候选数 `size`

当前默认：

```python
size = 128
```

输入有32个 backbone 时，当前使用平衡配额，通常每个结构输出：

```text
128 / 32 = 4个cell配置
```

所以一次 `propose()` 最终返回128个 Candidate。

### 3. 当前轮数 `round_index`

`round_index` 用于：

- 改变随机种子；
- 改变当前 MRED budget；
- 记录候选来自第几轮。

当前 budget 选择为：

```python
budget_index = (
    backbone.mred_band
    + round_index
    + attempt
) % 8
```

因此同一 backbone 在不同轮次会尝试不同误差约束。

### 4. 已经评估过的配置 `excluded_hashes`

这是之前已经送过 DC/Verilator 的 `cell_hash` 集合。

作用是：

> 防止同一个“结构 + cell配置”再次进入昂贵的真实评估。

`cell_hash` 由：

```python
cell_hash = hash({
    "structure": structure_hash,
    "cells": cells
})
```

计算得到。

因此：

- 相同结构、相同 cells：视为重复；
- 相同结构、不同 cells：是不同候选；
- 不同结构、相同 cell 编号：也是不同候选。

### 5. 上一代种群 `warm_starts`

从第二轮开始，Runner 会把当前 population 作为 `warm_starts`：

```python
warm_starts=population
```

但当前 warm start 很弱。DiffAM 只读取其中的 cell 配置，对相应 logits 增加偏置：

```text
某位置上一代选择了 cell 2
    ↓
这一轮初始化时给 cell 2 的 logit 加 2.0
```

它不会继承：

- 上一轮 logits；
- Adam optimizer 状态；
- 上一轮梯度；
- 上一轮 λ；
- area/power/delay/MRED 数值。

也就是说，上一代真实 PPA 不会直接进入 DiffAM 的 loss。只是这些 warm candidates 已经由 Runner 的 Pareto 选择过滤过，因此间接包含了选择压力。

---

## 二、DiffAM 还有哪些隐式输入

除了 `propose()` 参数，DiffAM 还会使用以下全局信息。

### 1. 乘法器实验引擎 `engine`

初始化时传入：

```python
DiffAMStage2Search(engine, config, run_dir)
```

`engine` 提供：

- 乘法器位宽；
- Booth/partial-product 编码方式；
- 截断信息；
- 原始 partial-product 矩阵；
- 2-2、3-2、4-2 cell 菜单；
- 哪些列允许使用近似 cell。

### 2. Cell 库

当前代码从下面的文件读取 cell：

```text
Appr_Comp/library.json
Appr_Comp/library42_native.json
```

每个 cell 主要提供：

- 逻辑真值表/LUT；
- 名义面积；
- cell 名称和编号。

当前没有训练 compressor MLP proxy。前向模拟直接使用库中 cell 的离散真值表，见 [solver.py](/home/lichangxian/Power_das/Appr_Comp/cellsolver/solver.py:30)。

### 3. DiffAM 超参数

默认参数包括：

```text
device             = cuda:0
训练步数            = 40
学习率              = 0.03
温度                = 1.0 → 0.25
初始λ               = 50
λ更新间隔           = 10步
每个温度采样数       = 8
MRED budget数量      = 8
MRED范围             = 1e-7 ～ 0.2
向量池               = 16M
```

定义在 [runner.py](/home/lichangxian/Power_das/trainer/arith_three_stage/runner.py:36)。

---

# 三、拿到输入结构后，DiffAM 做什么

## 步骤1：把 backbone 重建成固定 compressor graph

对每个不同的 `structure_hash`，DiffAM 首先创建并缓存一个 `FrozenDiffAMProblem`。

包含：

```python
FrozenDiffAMProblem(
    backbone,
    graph,
    tree,
    pp_specs,
    solver,
)
```

具体过程是：

### 1. 激活该结构的截断参数

```python
engine._activate_trunc_profile(backbone.k)
```

得到该 `k` 对应的 partial-product 分布和截断位。

### 2. 根据 ct22/ct32/ct42 重建 compressor tree

```python
CompressorTree(
    initial_pp,
    backbone.ct32,
    backbone.ct22,
    backbone.ct42,
)
```

这一步确定：

- 每一级有哪些 compressor；
- 每一列有哪些 compressor；
- compressor 的类型；
- 输入输出之间的拓扑关系。

### 3. 转成图结构

```python
graph = CompressorGraph(...)
```

每个 compressor 在图中有一个 `node id`。一个节点还带有：

```text
stage
column
graph_type
local_index
```

例如：

```text
node 17:
    stage       = 2
    column      = 9
    graph_type  = 0
    local_index = 3
```

### 4. 生成固定 canonical routing

```python
routing = CanonicalRouter().route(graph)
```

当前 DiffAM 不搜索 routing。它只需要一个确定的连接方式，让张量模拟器知道每个 compressor 的输入来自哪里。

### 5. 解析 partial-product encoder

代码会调用实际乘法器生成逻辑：

```python
Mul(...).emit_pp_encoder()
```

然后解析出每个 partial-product bit 的生成规则：

```python
pp_specs
```

因此 DiffAM 软件模拟使用的 partial-product 编码与最终 RTL 生成方式保持一致。

### 6. 创建可微 TreeSim

```python
tree = TreeSim(graph, routing, pp_specs, device)
```

这个对象负责：

- 根据输入 `a, b` 生成 partial-product bits；
- 沿固定 compressor tree 传播；
- 在每个节点应用当前选择的 cell LUT；
- 计算最终乘法器输出。

整个结构构建过程在 [diffam_search.py](/home/lichangxian/Power_das/trainer/arith_three_stage/diffam_search.py:83)。

---

# 四、把 cell 选择变成可训练参数

## 1. 找出合法的 cell slot

并不是所有 compressor 都允许替换成近似 cell。

只有同时满足下面条件的位置才成为可训练 slot：

- 节点是 compressor cell；
- 类型存在对应的近似 cell 库；
- 所在列允许近似；
- 该类型至少有一个近似 cell 候选。

最终得到：

```python
solver.space.slots = [
    (node_id, compressor_type, column),
    ...
]
```

例如：

```text
slot 0 = (node 17, 3-2类型, column 9)
slot 1 = (node 23, 3-2类型, column 10)
slot 2 = (node 41, 2-2类型, column 11)
```

不合法或不允许近似的位置始终使用精确 cell。

## 2. 为每个 slot 建立一行 logits

假设有80个可近似位置，cell 菜单最多有6种，那么 logits 形状大致是：

```text
logits.shape = [80, 6]
```

其中：

```text
logits[i, 0]：第i个位置选择精确cell的分数
logits[i, 1]：选择近似cell 1的分数
logits[i, 2]：选择近似cell 2的分数
...
```

不同 compressor 类型的 cell 数量可能不同，因此还会建立 `mask`，把不存在的选择设成负无穷：

```text
3-2位置：[合法, 合法, 合法, -∞, -∞]
4-2位置：[合法, 合法, 合法, 合法, 合法]
```

这里 `cell_type=0` 始终表示精确 cell。

---

# 五、logits 如何初始化

每次 `_train()` 都会重新创建一个 solver，并重新初始化 logits：

```python
solver.logits.normal_(mean=0, std=0.7)
solver.logits[:, 0] += 0.8
```

即：

- 所有分数先随机初始化；
- 精确 cell 的分数额外加0.8；
- 因此初始状态更偏向精确 cell，但不是强制全精确。

如果有 warm start：

```python
solver.logits[row, warm_cell_type] += 2.0
```

例如上一代某位置选择了近似 cell 3：

```text
随机初始：[0.2, -0.1, 0.3, 0.0]
warm偏置：[0.2, -0.1, 0.3, 2.0]
```

这一轮就更可能从 cell 3 附近开始探索。

---

# 六、每个训练步骤做什么

当前每次训练默认40步。

## 1. 用 STE 得到离散选择

代码是：

```python
w_soft = softmax(logits / tau)

hard = one_hot(argmax(w_soft))

selection = hard + w_soft - w_soft.detach()
```

见 [solver.py](/home/lichangxian/Power_das/Appr_Comp/cellsolver/solver.py:197)。

它的效果是：

### 前向数值

因为：

```python
w_soft - w_soft.detach() = 0
```

所以：

```text
selection前向值 = hard
```

每个位置真正使用一个离散 cell，不是多个 cell 的混合输出。

### 反向梯度

`hard` 不可导，`detach()` 部分也没有梯度，因此：

```text
d selection / d logits
=
d w_soft / d logits
```

所以梯度会像 softmax 一样传回 logits。

简化来看：

```text
前向：
选择 cell 2，实际运行 cell 2 的 LUT

反向：
假装这个选择来自 softmax，
计算“提高/降低各个 cell 概率”会怎样改变损失
```

## 2. 使用离散 cell LUT 运行前向模拟

对于每个 slot，`selection` 前向是 one-hot，因此 TreeSim 选择一个具体 cell 的真值表：

```text
partial-product bits
    ↓
node 0：精确3-2 LUT
    ↓
node 1：近似3-2 cell 2 LUT
    ↓
node 2：近似4-2 cell 1 LUT
    ↓
最终乘法结果
```

它不会在前向时调用 DC，也不使用 compressor MLP proxy。

## 3. 估算可微 MRED

DiffAM 使用固定的16M输入池，但每个梯度步骤不会完整计算全部16M。

向量被分成：

```text
S12：小乘积样本，0 < golden < 2^22
S3 ：其余较大乘积样本
```

每步使用：

- 全部 S12；
- 一段随机 S3 样本；
- 对 S3 加权，使估计量接近完整输入分布。

得到：

\[
\widehat{\mathrm{MRED}}
=
\frac{1}{N}
\sum_i
\frac{|y_i-y_i^\star|}{y_i^\star}
\]

这里：

- \(y_i\)：当前离散 cell 配置的乘法结果；
- \(y_i^\star\)：精确乘法结果。

## 4. 计算名义面积项

每个 cell 在库里有一个名义面积。

面积项是：

\[
A_{\text{fraction}}
=
\frac{
\sum_{\text{slot}} A(\text{selected cell})
}{
\sum_{\text{slot}} A(\text{exact cell})
}
\]

注意，这只是 cell 面积之和的比例，不是：

- DC 综合总面积；
- 关键路径延迟；
- XA 功耗；
- 常数传播后的实际面积。

## 5. 计算损失

当前损失为：

\[
L =
A_{\text{fraction}}
+
\lambda
\operatorname{ReLU}
\left(
\frac{\widehat{\mathrm{MRED}}}{B}-1
\right)
\]

其中 \(B\) 是当前 MRED budget。

有两种情况。

### MRED 没超过 budget

\[
\widehat{\mathrm{MRED}}\le B
\]

误差惩罚为0，此时主要推动面积下降。

### MRED 超过 budget

\[
\widehat{\mathrm{MRED}}>B
\]

产生误差惩罚，推动 logits 改为误差更低的 cell。

每10步还会对当前硬配置运行一次 `gate_mred()`，然后更新 λ：

```python
lambda = max(
    5,
    lambda + lambda_step * (hard_mred / budget - 1)
)
```

如果一直超预算，λ 会增大，后续训练更重视 MRED；如果低于预算，λ 可以下降。

## 6. 反向传播并更新 logits

```python
loss.backward()
optimizer.step()
```

真正被训练的只有：

```text
solver.logits
```

不会训练：

- cell LUT；
- compressor tree；
- partial-product encoder；
- MRED estimator；
- PPA proxy；
- routing。

---

# 七、训练后如何产生离散配置

DiffAM 不是只输出最后一个 argmax 配置，而是收集一批候选。

## 1. 训练轨迹配置

每一步都取得：

```python
config = solver.hard_config()
```

如果本步硬配置和上一次不同，就记录下来：

```text
step 1配置
step 3配置
step 6配置
...
```

这样可以保留优化过程中经过的不同离散点。

## 2. 最终配置

训练40步结束后，再记录最终 argmax 配置：

```python
source = "hard_final"
```

## 3. 从最终 logits 采样

使用4个温度：

```text
0.35
0.60
1.00
1.60
```

每个温度默认采样8次，所以最多产生：

```text
4 × 8 = 32个采样配置
```

低温更偏向当前 argmax，高温更容易产生差异较大的配置。

## 4. 加入全精确配置

筛选前还会额外加入：

```python
config = {}
source = "exact"
```

空字典表示所有合法位置都使用 `cell_type=0`，也就是精确 cell。

它只是一个基准候选，并不被假设为 MRED 最低。

---

# 八、内部 cell 配置是什么格式

训练内部使用：

```python
CellConfig = {
    node_id: (compressor_type, cell_type)
}
```

例如：

```python
{
    17: (0, 2),
    23: (0, 1),
    41: (1, 3),
}
```

表示：

```text
node 17使用类型0菜单中的近似cell 2
node 23使用类型0菜单中的近似cell 1
node 41使用类型1菜单中的近似cell 3
其他所有节点使用精确cell
```

精确 cell 不会写入字典。因此：

```python
{}
```

代表全精确配置。

---

# 九、几十个配置如何筛选成4个

假设某个 backbone 的 quota 是4。

## 1. 配置级去重

将配置转成排序后的 key：

```python
(
    (node, compressor_type, cell_type),
    ...
)
```

完全相同的配置只保留第一次出现的版本。

因此，轨迹配置、最终配置和采样配置即使来源不同，只要 cell 选择一样，就只保留一个。

## 2. 转成临时 Candidate，进行全局去重

每个配置被转成 Candidate，并计算：

```python
cell_hash = hash(structure_hash, cells)
```

如果这个 `cell_hash` 已经在 `excluded_hashes` 中，就直接排除。

## 3. 便宜的 `gate_screen`

对未评估候选先使用约24K小乘积样本做快速 MRED 排序：

```python
proxy_screen = gate_screen(config)
```

为了减小快速口径和较完整口径之间的系统偏差，还计算全精确配置的 offset：

```python
offset =
    exact_gate_mred
    - exact_gate_screen

corrected_screen =
    candidate_gate_screen
    + offset
```

## 4. 保留短名单

如果最终需要4个，该阶段最多保留：

```text
max(2 × 4, 4) = 8个
```

Screen 排序逻辑是：

- 估计满足 `budget × 1.10`：优先面积节省大；
- 估计不满足：优先 MRED 更小。

## 5. 对短名单运行 `gate_mred`

`gate_mred()` 使用：

- 全部小乘积层；
- 固定的大乘积子集；
- 分层权重。

它比 `gate_screen()` 更贵、更准确，但仍是软件估计，不是 Verilator 的最终16M实测。

## 6. 先选质量最好，再选择差异大的配置

最终排序首先考虑：

- 满足 budget 的配置优先；
- 满足时，面积节省更大优先；
- 不满足时，MRED 更小优先。

先选择排名第一的配置。之后为了避免4个配置过于相似，会选择与已选配置 Hamming distance 更大的配置。

这里的配置距离就是：

```text
有多少个node的cell选择不一样
```

筛选代码位于 [diffam_search.py](/home/lichangxian/Power_das/trainer/arith_three_stage/diffam_search.py:324)。

---

# 十、DiffAM 的直接输出是什么

`propose()` 的直接输出是：

```python
List[Candidate]
```

默认长度128。

每个 Candidate 类似：

```python
Candidate(
    k=12,                         # 不变
    ct22=[...],                   # 不变
    ct32=[...],                   # 不变
    ct42=[...],                   # 不变
    cells=[
        [stage, column, graph_type, local_index, cell_type],
        ...
    ],
    routing=None,
    stage=2,
    operator="diffam_ste",
    area=None,
    power=None,
    delay=None,
    mred=None,
    metadata={
        "method": "diffam",
        "source": "trajectory" | "hard_final"
                  | "logit_sample" | "exact",
        "target_mred": ...,
        "proxy_screen_mred": ...,
        "proxy_mred": ...,
        "proxy_area_saving": ...,
        "seed": ...,
        "diffam_round": ...,
        "diffam_attempt": ...,
        "generation": ...
    }
)
```

这里最重要的是：

> DiffAM 输出时，真实 area、power、delay、MRED 都还是 `None`。

因为 DiffAM 输出的只是“建议送去真实评估的离散设计”。

### `cells` 的编码

输出中每个近似 cell 是：

```text
[
    stage,
    column,
    graph_type,
    local_index,
    cell_type
]
```

例如：

```python
[
    [2, 9, 0, 3, 2],
    [3, 11, 1, 0, 1],
]
```

表示两个 compressor 使用了近似 cell。没有出现在 `cells` 里的 compressor 默认使用精确 cell。

转换代码位于 [diffam_search.py](/home/lichangxian/Power_das/trainer/arith_three_stage/diffam_search.py:151)。

转换后还会验证：

```python
candidate.structure_hash == backbone.structure_hash
```

因此能够保证 DiffAM 没有修改 Stage1 结构。

---

# 十一、DiffAM 输出之后，Runner 做什么

DiffAM 返回128个 Candidate 后，Runner 才开始真实评估：

```python
self._evaluate_with_retries(offspring)
```

这一阶段会进行：

- RTL生成；
- DC综合；
- 面积提取；
- 延迟提取；
- 功耗评估；
- Verilator 16M MRED 验证。

评估后，每个 Candidate 才变成：

```python
area = 实际综合面积
power = 实际功耗
delay = 实际延迟
mred = Verilator 16M MRED
```

然后 Runner 把：

```text
上一代128个
+
DiffAM新生成128个
```

放在一起，使用 NSGA-II 环境选择保留128个：

```python
population = environmental_select(
    population + offspring,
    128,
    delay_limit=1.5,
)
```

同时更新外部 Pareto archive。

循环完成后，Runner 在不同 MRED band 中选择：

```text
area代表点
power代表点
knee代表点
```

8个 band × 3种角色，最终通常输出24个 elites：

```text
stage2/elites_24.json
```

Runner 的完整调用过程在 [runner.py](/home/lichangxian/Power_das/trainer/arith_three_stage/runner.py:550)。

---

## 最准确的一句话总结

当前 DiffAM 的输入是：

> 固定的 Stage1 compressor tree、合法 cell 库、MRED budget、随机向量和可选的上一代 cell 配置。

它做的事情是：

> 把每个合法位置的 cell 选择参数化成 logits，通过“离散前向、STE反向”最小化“名义面积 + 超出 MRED budget 的惩罚”，然后从训练轨迹和 logits 中生成大量离散配置，再经过软件 MRED、面积和多样性筛选。

它的直接输出是：

> 一批结构完全不变、只有 `cells` 不同、尚未经过真实 PPA 评估的 Stage2 Candidate。

随后 DC/Verilator 和 NSGA-II 才把它们变成真正的 Stage2 Pareto 结果。