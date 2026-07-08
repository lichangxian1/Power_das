# 近似压缩器 cell 功耗预测模型可行性验证（2026-07-08）

**问题**：能否训一个预测模型，输入近似压缩器的 RTL / ABC 综合网表，输出与 DC 功耗数值一致或排序一致（梯度一致）的功耗值，从而替代/加速 DC 表征？

**结论：功耗维度完全可行，且比预想的更简单——不需要网表，真值表解析特征 + 线性/GBDT 模型即达 EDA 噪声地板。**

## 数据

现成监督数据，零新增 EDA 成本：
- `library42_native.json`：1999 个原生 4:2 cell，DC `compile -map_effort medium` 标签（area/dyn_mw/tmax），输入活动 p_one=0.25
- `library.json`：3882 个 3:2/2:2 cell（dyn_w/area/tmax），static_prob=0.25/toggle=0.125

## 特征

1. **TT 解析特征**（零成本，纯真值表闭式计算）：每输出在 p=0.25 偏置测度下的信号概率 p1、随机向量 toggle=2p1(1-p1)、**每输入布尔影响度** I_i(f)=P(f(X)≠f(X⊕e_i))、总影响度（平均敏感度）、minterm 数、与 exact 的加权 L1 / 汉明距离
2. **yosys+ABC 网表特征**：`proc;opt;techmap;abc -g AND,...,XNOR` 后按门型计数（1999 cell 单机一次通刷 <5min）

## 结果（5-fold CV，GBDT = HistGradientBoostingRegressor）

### 功耗 dyn（主指标）

| 数据集 | 特征 | Spearman | Pearson | MAPE | pairwise 排序准确率 |
|---|---|---|---|---|---|
| comp42n (1999) | TT 解析 | **0.9954** | 0.9927 | **1.21%** | 0.972 |
| comp42n family 留出（train rand → test structured, OOD） | TT 解析 | **0.9882** | 0.9698 | 4.33% | 0.960 |
| comp32 (3722) | TT 解析 | **0.9990** | 0.9990 | 1.26% | 0.988 |
| comp42n | **Ridge 线性**（TT 解析） | 0.9913 | — | **0.91%** | — |
| comp42n | yosys 网表 only | 0.2250 | 0.33 | 12.6% | 0.571 |

**对照标尺**：12 个双档复检 cell 上 DC medium vs ultra 的功耗 Spearman=1.000、比值 1.000±0.008 —— 预测器 MAPE ~1% 已贴近 DC 自身跨口径波动；功耗标签跨综合力度稳定（翻车的 5/17 是 area/tmax 闸门，不是功耗）。

**机理**：动态功耗 ≈ 开关活动性的光滑（近线性）函数。permutation importance 前两名 `sum_infl_tot`（0.85）、`carry_infl_tot`（0.44）—— 即输出布尔总敏感度。这也解释了为什么 yosys generic 门数没用：功耗由活动率驱动，不由门数驱动（generic 网表也不反映 T28 mapping）。

### 面积 / 时序（次要）

- area：comp32 Spearman 0.88 / MAPE 5%；comp42n 0.65–0.70（TT+yosys 联合最好）。可用于粗筛，不能替代 DC 精排。
- tmax：0.40–0.70，最难（mapping 离散性），且本身 medium/ultra 排序就不一致 → 仍需终口径复检。

## 使用建议

1. **外环 cell 搜索 / 库扩展的即时打分**：`dyn ≈ w·[TT解析特征]` 线性公式可以直接嵌进 `outer_cell_search` 的解析提议打分（当前只有 area/wae），让候选生成在 2^48 全空间即时评估功耗，DC 只复检幸存者。
2. **预筛选留 margin**：top-5% 重叠率 0.84–0.96，预筛时取预测 top-4K 送 DC 精检 top-1K 这种比例安全。
3. **范围边界**：本验证是 **cell 级**（≤4 输入）。整乘法器级功耗预测是另一个问题（图结构 + XA 标签，outputs/*/summary 有数据但未验证）；且 cell 级标签是 DC report_power 代理口径，选型排序自洽即可，绝对值对齐 XA 需另标定。

## 第二部分：完整 16-bit 乘法器级、对齐 XA（2026-07-08 追加）

**问题**：预测完整近似乘法器的 XA SAIF 实测功耗（@1.5ns），替代/前置 XA 复检。

**数据**：扫描全部 `outputs/*/reeval_xa*.csv` + `ppa_xa.csv`，取 success 且本地有 MUL.v 的设计：**121 个设计 / 13 个 run**（0.162–0.803 mW）。booth 与 06-27~07-02 几个 run 的 MUL.v 未同步到本地，未入库。数据集 `design_xa.csv`（`build_design_dataset.py`）。

**特征**：
- S1 纯结构（RTL 免 EDA）：FA/HA/CT42/comp32/22/42n 实例计数、库查表 dyn/area 求和、**活跃 pp 位数 n_pp_active**（截断深度信号，`assign pp_*=1'b0` 不算）、常量 0/1 计数、wire/assign 数
- S2 = S1 + DC 副产品（area_dc、delay）——对应「DC 已跑、XA 未跑」场景

**结果（leave-one-run-out，13 折跨 run 留出，`train_eval_design.py`）**：

| 方案 | 全局 Spearman | MAPE | run 内 Spearman（mean / min） |
|---|---|---|---|
| 裸基线 area_dc | 0.963 | — | 0.835 / 0.18 |
| S1 纯结构, Ridge | 0.942 | 7.7% | 0.801 / 0.39 |
| **S2 +DC area/delay, Ridge** | **0.967** | **5.8%** | **0.921 / 0.73** |

要点：
- **可行，但比 cell 级低一档**。S2 Ridge 是可用工位：DC 跑完即可预测 XA ±6%、run 内排序 0.92，XA 只需复检帕累托候选。相对裸 area_dc 的增量主要在 run 内排序（0.835→0.921）和绝对值对齐。
- 纯结构 S1 之所以能到 0.94：截断主导 regime 下功耗 ≈ f(活跃 pp 位数, 树规模)。**第一版特征忘了活跃 pp 位（截断位的 assign 仍存在只是赋 1'b0），S1 全局 Spearman 只有 0.23**——教训：设计级最重要的单一特征是有效开关规模。
- 已知盲区：数据 121 个且多为 k-sweep 截断设计，cell 密集/booth 设计欠覆盖；布线/glitch 效应（paired 实验显示 routing 贡献可达功耗差 8–34%）当前特征完全看不到，这是 run 内 min 0.73 的主要残差来源。

**升级路径**（按投入排序）：
1. 零成本上线：S2 Ridge 作 XA 前置过滤器（每次 DC 后即时预测，XA 复检预测 top 候选）。
2. 数据自然增长：reeval 时顺带覆盖各 iter 检查点（netlist 现成，只花 XA），n 可从 121 → 千级。
3. 结构升级：GNN over CompressorGraph + **解析活动传播**特征（从 pp 的 p1=0.25 逐 cell 用真值表传播信号概率/toggle——cell 级实验已证明敏感度≈功耗），目标 run 内 >0.95，届时可作训练内环 objective，直接替换 DC report_power 的 20× 代理，消除「objective 降但 XA 升」的错位。

## 复现

```bash
cd Appr_Comp/pwrpred
python extract_tt_features.py            # -> tt42.csv
yosys -q -p 'read_verilog ../rtl/comp42n_lib.v; proc; opt; techmap; opt; \
  abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT; opt_clean; tee -q -o yosys_stat42.txt stat'
OMP_NUM_THREADS=4 python train_eval.py   # 注意：不限线程会 OpenMP 抖动跑不完
python extract_tt_features32.py && OMP_NUM_THREADS=4 python train_eval32.py
```

（脚本内 BASE 路径指向当时的 scratchpad，迁移使用时改为本目录。）
