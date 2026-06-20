# 阶段 C：进一步提升近似乘法器 PPA 优化能力的方案

> 背景：Phase B（[PHASE_B_DESIGN.md](PHASE_B_DESIGN.md)）已把「压缩器类型搜索 + 闭式误差 reward」接进 ARITH-DAS 并跑通；
> 2026-06-20 的 DC-in-loop 实验（`outputs/2026-06-20_dc_train/`）证明了两件事，也暴露了三个结构性瓶颈。
> 本文是把这些瓶颈拆成可落地动作的方案，按 PPA 杠杆排序。**Tier 1 = 拓宽被近似的对象；Tier 2 = 磨锋利优化器。**

---

## 0. 最新结果说明了什么（2026-06-20 DC-in-loop，apples-to-apples @2ns，DC area + XA power）

把 4 个 DC 点对 EvoApprox 前沿插值对比（`outputs/2026-06-20_dc_train/dc_trained_eval.csv`）：

| 设计 | MAE | Power vs 前沿 | Area vs 前沿 | BIAS/MAE |
|---|--:|--:|--:|--:|
| dc_mb4096 | 3.2e4 | **−5.7%** | +26% | 0.56 |
| dc_mb16384 | 3.5e4 | **−13.9%** | +11% | 0.66 |
| dc_mb65536 | 6.1e4 | **−23.9%** | +4% | 0.47 |
| dc_mb262144 | 7.8e4 | **−28.5%** | **−5%** | 0.10 |

**确定的结论：**
1. **Power 轴已压过 EvoApprox 前沿（6–28%）**，且 DC-in-loop 明显优于 ABC-in-loop（换 in-loop PPA 源走对了）。
2. **Area 是弱轴**：基本贴前沿（+11%~−5%），原因是当前唯一的近似动作（低列 cell 替换）只省几个门的边角料。
3. **Bias 抵消是真正差异化**：所有项目点系统性落在 `|BIAS|=MAE` 对角线下方（卖点图 `dcvs_bias_vs_mae.png`）。

**暴露的瓶颈：**
- **B1 近似动作空间太窄**：只在一棵本质精确的 Dadda 树上、对低列换 cell。→ area 压不下、够不到低功耗区。
- **B2 重尾/WCE 完全没进 reward**：每个设计 `RMSE/MAE≈100、ER≈99.9%、WCE≈2³¹`。reward 只约束均值（med/bias）。→ 设计不可用，且尾巴堵死了高精度前沿。
- **B3 误差代理有偏 + 前沿稀疏 + 定权标量化**：解析误差用 `P(1)=1/4`（只第一级 PP 严格成立）；只扫了 4 个 budget；固定权重而非真 Pareto。

---

## TIER 1 — 拓宽「被近似的对象」（拿 area 和低功耗区）

### ① 低列截断 + 学习校正 + 抵消（核心新颖机制）

**机制三件套：**

1. **整列截断**：对最低 k 列根本不生成 PP、不建压缩器（PP 生成器跳过 `a_i & b_j where i+j < k`）。阶跃式 area/power 下降——EvoApprox 的低功耗点全靠它。
2. **截断引入确定性负偏置，量级闭式可算**：
   ```
   Δ(k) = Σ_{col<k} 2^col · E[s_col]     # s_col=该列真实位和；丢掉的永远是正值
   → 截断本身 bias = −Δ(k)
   ```
3. **用「学习校正常数 C + 保留列的正偏置 cell」去 null 掉 −Δ**：让 RL 协同优化 (k, C, 每 cell 符号)，使
   ```
   total_bias = −Δ(k) + C + Σ_node bias·2^col  ≈ 0
   ```
   这正是现有 `_analytic_error` 的 `abs_bias` 已在压的量——只要把 `−Δ(k)` 和 `+C` 加进那个求和，
   `bias_weight` 项会自动把净偏置压到 0。**一句话：截断负责 PPA，学习抵消负责把精度赚回来。**

**落地点（按改动量）：**

| 改动 | 位置 | 说明 |
|---|---|---|
| 加截断动作 | 外层 `transition/get_action_mask`（定 `ct32/ct22` 数量处） | 给 `col<approx_max_col` 加「截断/保留」决策或「截断深度 k」categorical，与 `sample_cell_types` 并列 |
| PP 生成跳过截断列 | `utils/mul.py` PP 初始化 / `compressor_tree.initial_pp` | `i+j<k` 的 PP 不生成（真正的工程，area 收益在此） |
| 注入校正常数 C | netlist 末尾常数加法 | C 可固定 round(Δ) 或离散档位可学 |
| reward 加 −Δ(k)+C | `_analytic_error` 的 `bias_total` | Δ(k)、C 闭式 O(1) |

**验证**：解析 bias 必须与 `Appr_Comp/approx_mul.py` 的 8-bit 穷举实测吻合（同 Phase A 验恒等式口径）。

#### ① 机制已 EDA-free 穷举验证（2026-06-20，`approx_mul.py --trunc k --correct bias`）

把截断+校正接进 bit-exact 参考器并 **8-bit 穷举（65536 输入对）** 验证。新增恒等式
`out − a*b == −Δ(a,b) + Σ_cells e·2^col`，**全场景 65536/65536 通过**；`--trunc 0` 与原 Phase A 输出逐字一致（回归）。

k=4（E[Δ]=12.25 → C=round=12，正偏置 cell）穷举结果：

| 场景 | 恒等式 | ER | MED | WCE | bias |
|---|:--:|--:|--:|--:|--:|
| 截断 only (C=0) | OK | 0.81 | 12.25 | 49 | **−12.25** |
| 截断 + 常数校正 (C=12) | OK | 0.91 | 7.85 | 37 | **−0.25** |
| 截断 + cell抵消 (C=0, 正cell) | OK | 0.90 | 25.4 | 130 | **+8.50** |
| 截断 + 常数 + cell | OK | 0.97 | 30.2 | 127 | +20.5 |

**证明的结论**：
1. 截断 = **确定性负偏置 −E[Δ]**（穷举精确吻合）。
2. **常数 C=round(E[Δ]) 把 bias 归零**（−12.25→−0.25），MED 同时减半——这是主抵消杠杆。
3. **正偏置 cell 是第二杠杆**（−12.25→+8.50，部分抵消 + 省面积）；**符号选错则更糟**（负 cell：−12.25→−50）→ 必须 RL 选符号，正是 bias reward 的活。
4. **WCE 随 k 增大**（k=4→37，k=6→241）→ 正是 ④ WCE 项要压的尾巴。**①②④ 必须合用**：截断吃 PPA、常数+cell 归零偏置、WCE 项控尾。

#### ① 剩余工程：接进搜索（Layer 2，需本机 openroad 冒烟）

参考器证明了数学，但**搜索 loop 的 RTL 发射尚未截断**（本机 yosys+openroad 可验证，故低风险但工作量实）。精确改点：

| 改动 | 位置 | 说明 |
|---|---|---|
| `self.initial_pp[:k]=0` 后再建 `CompressorGraph` | `CompressorRouting.__init__/reset` | 低列 height=0 → 无 PP 节点/压缩器；需查 `get_action_mask/transition/to_graph` 容忍零高列 |
| `emit_pp_encoder` 尊重截断 | `utils/mul.py:131`（**现在硬重算满 pp，忽略截断**） | 跳过 col<k 的 `wire/assign`（否则零高列 emit 出非法 `wire[-1:0]`） |
| `out[<k]` 与常数注入 | 末级 prefix adder / 输出装配（`routed_wire_list` → out） | 截断列无 routed 线 → `out[<k]` 接 0；C 拆成 col≥k 的常数 `1'b1` PP 位注入树 |
| reward 加 `−Δ(k)+C` | `_analytic_error`（`bias_total/wce_total`） | 与参考器同公式；gated by `trunc_cols`（默认 0=回归） |
| `trunc_cols` 当扫描超参（v1）/ 策略头（v2） | 配置 / 新动作头 | v1 先像 med_budget 一样扫 k；v2 再加可学截断深度头（仿类型头） |

**注意**：reward 项与 RTL 截断**必须同时落地**（否则搜索奖励了不存在的截断）。故 Layer 2 一次性做完 + 本机 openroad 冒烟（emit→yosys 综合通过→`approx_mul.py` 对导出 RTL 穷举校验 ER/MED/WCE）再开。

### ② 近似部分积（截断的兄弟，次优先）

把某列两行 PP 的相加用 **OR 压缩**（省一个压缩器），或丢偶数行 PP 位。机制/误差建模与 ① 同构，作为 ① 跑通后的增量。

### ③ 回收 delay 富余（几乎免费，零机制改动）

4 个 DC 设计关键路径 1.31–1.40ns，但 target 2ns → 白扔 ~0.6ns slack。
- **收紧 `--target_delay`（如 1.5ns）重训**：DC 拿 slack 去 downsize → 直接换更小 area/power。
- **把 delay 画进对比图**：项目大概率在 delay 轴也压 EvoApprox（exact dc_dw=1.98ns）。

---

## TIER 2 — 磨锋利优化器（同样搜索成本，更好的解）

### ④ 尾部/WCE 约束（治「不可用」+ 解锁低 MAE）✅ 已实现（默认关）

**问题**：`_analytic_error` 只惩罚均值 → RL 乐于让个别近似 cell 落到高权重路径，单点误差灾难性（WCE≈2³¹）。
尾巴同时撑大有效误差，低 budget 够不到真低 MAE。

**改法（已落地，约束式 + 可微，均默认关）**：解析 WCE 可加上界
```
wce_bound = Σ_node 2^col · maxe(cell)     # 库表每 cell 的 maxe 字段已有
```
- 约束式（`get_objective`）：`err_term += wce_violation_weight·max(0, wce_bound − wce_budget)/error_scale`
- 可微（`get_error_loss`）：`+= wce_loss_weight·relu(Σ (p·maxe)·2^col − wce_budget)/error_scale`

**配置键**（全默认关 → 回归字节级一致）：
```yaml
wce_budget: null          # LSB；设了才罚
wce_violation_weight: 0.0 # 约束式权重（配 /error_scale，建议起步 1.0）
wce_loss_weight: 0.0      # 可微 surrogate 权重（建议起步 1.0）
```
**现成 config**：`configs/config_groups/mul_16_and_approx_wce.yaml`（在 p2p1 上加 wce 项，起步 budget=2^20）。

**验证**：`approx_mul.py` 穷举出的真实 WCE/RMSE 随 `wce_budget` 单调下降；目标把 RMSE/MAE 从 ~100 压到个位数。

### ⑤ 用真实逐列翻转概率替换 P(1)=1/4（消除代理偏差）

**问题**：`_analytic_error` 的 `bias/wae` 是 P=1/4 假设算的，只第一级 PP 严格成立 → 优化器在有偏代理上找解。
**基础设施已就绪**：`scripts/enrich_eda_observables.py` 已从 SAIF 抽 `node_toggles`。
**做法**：(1) 对精确 16-bit 乘法器跑一次 SAIF 拿每节点真实 P(1)；(2) 用真实 P 对 cell 真值表（`cand_32/22.json`）
重算期望 bias/wae，替换库里 1/4 版；(3) in-loop power 同理换成真实 toggle 的 SAIF 功耗（现训练用 DC 0.5 翻转率，比 XA 高 ~20×）。
**验证**：解析 MAE/BIAS 对 Verilator 4M MC 实测从明显偏差收到几个 %。

### ⑥ 铺密前沿 + 收敛 + area 一等目标

- **med_budget 往两头扫**：现 4 个点挤在 MAE 3e4–1.3e5；往低（256/1024/4096）填高精度段，往高（5e5/1e6/4e6）填低功耗段（配 ① 才能真触达 0.2mW 区）。
- **DC-in-loop 跑到收敛**：现 200ep，ABC 那批 1000ep；短跑本身压住了 PPA。
- **area 一等目标 / 真 Pareto**：把 `reference_point/pareto_target`（现为 `["delay","area"]`，**无 error**）扩成 (error, area, power) 三目标，别再固定权重标量化。

---

## 落地顺序（每步独立可验证、独立出图）

1. **④ 尾部约束**（最小，先让设计可用，RMSE/MAE 从 ~100 降下来）← **本轮实现**
2. **⑤ 真实概率**（接已有 SAIF infra，消除代理偏差）
3. **① 截断+校正+抵消**（最大工程 = 最大 PPA 收益 + 核心新颖性）
4. **⑥ 扫 budget + 收敛 + 三目标 Pareto**（铺满前沿，出最终对比图）
5. **③ 收紧 target_delay**（零机制改动，顺手回收 slack）

---

## 实现状态

| 项 | 状态 |
|---|---|
| ④ WCE 约束式（`get_objective`）+ 可微（`get_error_loss`）+ 配置键 + maxe 张量 | ✅ 2026-06-20，默认关＝字节级一致 |
| ① 机制：截断+校正+抵消，`approx_mul.py --trunc/--correct` 8-bit 穷举验证（恒等式 65536/65536，回归一致） | ✅ 2026-06-20 |
| ① Layer 2：搜索 loop RTL 发射截断 + reward `−Δ+C` + `trunc_cols` 超参（需本机 openroad 冒烟） | ⏭ 下一轮（spec 见上） |
| ②③⑤⑥ | ⏭ 待办（顺序见上） |
