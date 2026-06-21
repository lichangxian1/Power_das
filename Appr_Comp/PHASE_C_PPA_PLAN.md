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

#### ① Layer 2：接进搜索 ✅ 已实现（2026-06-20，比原 spec 更省）

**关键简化（实现时发现）**：不必物理删列、不必动 `CompressorGraph`/动作掩码/输出装配。
压缩树只是把驱动 `pp_*` 的东西求和——所以**截断 = 把低列 PP 线用常数（校正常数 C 的位）驱动而非 `a&b`**；
树/布线/动作空间**全不变**，DC `compile_ultra` 常数传播自动删掉低列死逻辑（=截断 PPA 收益）。
`out = a*b − Δ_actual + C` 位精确，零图改动、零输出装配改动。

| 改动 | 位置 | 说明 |
|---|---|---|
| 低列 PP 用常数位驱动（前 m_c 个 `1'b1` 表示 C，其余 `1'b0`） | `utils/mul.py` `emit_pp_encoder`（AND 分支） | 读 `self.ct.trunc_cols/trunc_bits`；col≥k 仍 `a&b`。trunc=0 逐字回归 |
| `_setup_truncation`：算 E[Δ]、C=round(E[Δ])、贪心拆成低列常数位、Δmax、WCE_trunc | `CompressorRouting` | `_start_reset` 拿到 `initial_pp` 后调一次 |
| reward 加 `−E[Δ]+C`（净偏置→cell 抵消残差）+ `WCE_trunc`（尾部上界） | `_analytic_error`（`bias_total/wce_total`） | gated by `trunc_cols`（默认 0=回归） |
| 截断列强制 exact（cell 会被 DC 删，别浪费） | `_masked_type_logits`（`col < trunc_cols`） | 近似 cell 落在 `[trunc_cols, approx_max_col)` |
| 把 `trunc_cols/trunc_bits` 挂到 `ct`（自动随 deepcopy/pickle 进 worker） | `get_samples` / `export_best_candidate` | 无需改 multiprocessing 签名 |
| `trunc_cols`/`trunc_correct` 当超参（v1，像 med_budget 扫 k） | `configs/.../mul_16_and_approx_trunc.yaml` | v2 再加可学截断深度头（仿类型头） |

**验证（全过）**：
- **位精确**：构造同款常数驱动喂进项目自带树模型，`out==a*b−Δ+C` 全 65536 输入通过；`WCE_trunc` 与参考器一致（k=4→37）。
- **常数可表示**：C=round(E[Δ]) 贪心拆进低列槽位，Σ m_c·2^c==C。
- **本机 openroad 实测**：16-bit @2ns，**trunc=8 area 3752→3391（−9.6%）**、delay/power 同降 → 证明 DC 确实删低列（纯截断、无 cell）。
- **回归**：trunc=0 发射零常数 PP、`_analytic_error`/掩码不变 → 字节级一致。

**跑法**：`scripts/train_dc.py --config configs/config_groups/mul_16_and_approx_trunc.yaml ...`（同 WCE sweep）。
建议扫 `trunc_cols∈{4,8,12}` × `wce_budget`，铺 PPA–误差前沿；导出最优解用 `approx_mul.py` 穷举校验真实 ER/MED/WCE。

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
| ④ 修 dtype bug：`maxe` JSON 里是 int → `p(Float)@maxes(Long)` 崩；强制 `float32` | ✅ 2026-06-20（WCE sweep 已可跑） |
| ① 机制：截断+校正+抵消，`approx_mul.py --trunc/--correct` 8-bit 穷举验证（恒等式 65536/65536） | ✅ 2026-06-20 |
| ① Layer 2：搜索 loop 常数注入截断 + reward `−Δ+C`/WCE_trunc + `trunc_cols` 超参；本机 openroad 实测 area −9.6% | ✅ 2026-06-20，默认关＝字节级一致 |
| ②③⑤⑥ | ⏭ 待办（顺序见上） |

---

## 2026-06-21 过夜 DC 8 点扫描结果（codex 审过 + 修过 3 处误差核算后）

**配置**：16×16，DC-in-loop，180 ep，8 samples，64 并发 DC。两轴：k-深度（k=0/4/8/12/16 @med65536）+ 误差预算（k8 @med=16384/65536/262144/1048576）。
**真实误差**：verilator 2M MC（`scripts/plot_sweep.py` + `/tmp/mcsim/tb.cpp`），out 与 golden 都按 31 位、差值 wrap 到最小幅度 ⇒ 纯近似误差（与顶位约定无关），用 WCE_trunc 自校验通过（k16 实测 max|e|=574849 < 解析上界 737281，bias≈−75≈0 证明截断校正常数有效）。

| run | k | obj | area | pwr(mW) | #近似cell | MED | NMED | bias |
|---|---|---|---|---|---|---|---|---|
| k0 | 0 | 5.386 | 895.4 | 12.37 | 26 | 9705 | 2.26e-6 | −9600 |
| k4 | 4 | 5.358 | 893.6 | 11.81 | 17 | 17351 | 4.04e-6 | −17349 |
| k8 | 8 | 5.083 | 821.4 | 11.60 | 9 | 6514 | 1.52e-6 | +3039 |
| **k12** | 12 | **4.605** | **729.3** | **9.40** | **1** | **4578** | **1.07e-6** | −2050 |
| k16 | 16 | 10.608 ⚠崩 | 556.1 | 7.36 | 0 | 76271 | 1.78e-5 | −75 |

**核心结论**
1. **k=12 全面最优**：面积 −18%、功耗 −24%（vs k0），且 NMED **也最低**。最优截断深度 ∈ (12,16]。
2. **k=16 过度截断崩溃**：obj 10.6（惩罚主导），NMED ×10。
3. **截断 > 近似 cell 作为误差杆**：k 增大时 RL 用的近似 cell 数 26→17→9→1，PPA 与精度同步改善——截断给的是确定性、可校正、有界的误差/单位面积，cell 给的是带偏置的散布。
4. **误差预算在固定 k 下是钝刀**：k8 实测 MED 不随 med_budget 单调（16384→11848 但 65536→6514）⇒ in-loop 解析误差代理与真实误差失准 ⇒ 对应 ⑤（用 SAIF/实测列概率重标定）。
5. **偏置部分可校正但非免费**：全局常数（均值）校正只削 MED ~40–50%，对部分设计反增（最小化 MED 的是中位数非均值）⇒ 可加"输出中位数常数校正"廉价层，但散布是地板。

**图**：`outputs/2026-06-21_dc_sweep/fig_k_depth.png`、`fig_error_budget.png`。
**下一步**：k=13/14 探针定最优深度；可学 k（policy head）；⑤ 重标定误差代理；输出中位数常数校正。
