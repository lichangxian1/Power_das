# 近似压缩器库 → ARITH-DAS / 乘法器优化 接入计划

> 目标：系统生成近似 **3:2 / 2:2 压缩器库**，按误差方向/大小与 area/power/delay 做 Pareto 筛选，
> 分别保留**正误差组**与**负误差组**，接入 ARITH-DAS 乘法器搜索，让不同方向误差在整电路里有机会互相抵消。
>
> 本文是对初版大纲的精修版，已结合本仓库实际代码结构。涉及的关键代码位置在文中以链接给出。

---

## 0. 关键背景（来自本仓库代码，决定计划怎么落地）

- 压缩器在框架里**不是真值表**，而是按名字实例化的模块：
  [`../utils/compressor_tree.py`](../utils/compressor_tree.py) 的 `declare_fa` / `declare_ha`（约 L1237–L1254）
  硬编码生成 `FA (...)` / `HA (...)`；模块体定义在 [`../utils/template.py`](../utils/template.py)（`FA_verilog_src` / `HA_verilog_src`，约 L135），
  由 [`../utils/mul.py`](../utils/mul.py) 追加到 netlist 末尾（约 L270）。
- `CompressorTree` 只存 `ct32` / `ct22`（每列**数量**），**没有"类型"维度**。
  搜索器 `CompressorGraph` / `CompressorRouting`（[`../trainer/arith_das.py`](../trainer/arith_das.py)，约 L205 起）选数量与布线，不选类型。
- 奖励通道已存在：`delay_weight / area_weight / power_weight` + 对应 `*_scale`（[`../trainer/arith_das.py`](../trainer/arith_das.py) 约 L449），
  还有 `area_budget` / `*_violation_weight`。**加一个 error 通道是顺理成章的扩展。**
- 真实 PPA 走远端沙盒 **DC + VCS + PrimeTime**：[`../run_power_sweep.py`](../run_power_sweep.py)（`evaluate_single_routing`，约 L202）、
  [`../sandbox_base/scripts/dc_run.tcl`](../sandbox_base/scripts/dc_run.tcl)；本地另有 yosys + abc + openroad 轻量流程
  （[`../utils/mul.py`](../utils/mul.py) `simulate_worker`，约 L333）。per-cell toggle / SAIF 管线由 `DUMP_SAIF=1` 开启。
- 标准单元库：t28（`library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib`）。

---

## 1. 四个最重要的修正（优先理解）

### 1.1 不要枚举 / 综合 65536 个 3:2，先纯 Python 算误差再剪枝，综合量降到 < 100
65536 = 所有 `4^8` 映射，但绝大多数不是有意义的压缩器。加上**单元误差有界** `|e(x)| ≤ 1`
（近似压缩器标准约束，否则误差在乘法器里爆掉）后：

以下计数已由 [`enumerate_compressors.py`](enumerate_compressors.py) 实际枚举验证（默认 `P(1)=1/4, |e|≤1`）：

| 类型 | 全部 `|e|≤1` 候选 | **置换等价类（要综合的代表，PPA 各异）** | 完全对称（只依赖 popcount） |
|------|------|------|------|
| 3:2  | 2916 | **660** | 36 |
| 2:2  | 54   | **36**  | 18 |

- **全部候选**：分析阶段全算误差，纯 Python，免费。
- **置换等价类**：综合后 area/power 对输入置换不变 → 每类只需综合一个代表。**这才是真正交给 DC 的数量。**
- **完全对称**：输出只依赖 popcount 的子集；**注意它会排除掉最有用的非对称近似**
  （如 `sum=a^b` 丢掉 cin 依赖、把 FA 当 HA 用）——所以**不要只取对称子集**。

> **结论：整库（含 exact）一次性综合约 700 个代表（3:2 的 660 + 2:2 的 36），远小于 65536。**
> 全是几个门的小单元，DC 编译各几秒；一次错峰批量即可全表征，缓存成 JSON，搜索阶段只读表、永不重综合。
> `80 路并行 / 切波` 不再需要；也化解了"远端共享、错峰才满速"的约束。
> 若想再压：综合前按「误差 Pareto」（每组 P/N/Z 在 `(|bias|, nmed)` 前沿）预筛到几十个再交 DC。

**误差计算阶段不要强加对称**（cin 概率与 a/b 不同，见 1.2）。

可选放宽：若要探索更大单元误差，放宽到 `|e|≤2`，计数随每 pattern 选择数增长，但抵消逻辑与组合复杂度都偏向小误差，**默认 `|e|≤1`**。

### 1.2 输出误差 = Σ(单元误差 × 列权重)，精确恒等；但每个单元的期望误差要用真实逐位概率
压缩器只是把"该列加权值"重新打包：近似单元把加权值改变 `e_local · 2^col`，末级 CPA 精确。所以

```
out = exact_product + Σ_{近似单元}  e_local(x) · 2^(col)
```

**精确成立**（与传播 / 顺序无关，是加权值守恒）。推论：

- 平均 / 偏置误差对单元误差是**精确可加的线性函数** → reward 的误差项可闭式、便宜、可微松弛
  （`Σ_type softmax(type) · e_type · 2^col`），不必每步仿真。
- 但 `E[e_local]` 取决于**该单元输入位的真实分布**。`P(1)=1/4` 只对 **AND 第一级 PP** 严格成立
  （`a_i & b_j`，独立均匀 → 1/4）；越往深层 sum 趋于 0.5、carry 偏低，且 cin 来自上一列进位，概率 ≠ a/b。
  **别用单一 (3/4,1/4) 给所有位置。**
- 两套概率：
  - (a) i.i.d. (3/4,1/4) 解析模型 → 库初筛 / Pareto 排序；
  - (b) 用现成 **SAIF / per-cell toggle 管线**（`DUMP_SAIF=1`）从真实 exact 乘法器测每个压缩器输入位的信号概率，
    作为放置 / 最终 reward 的 `P(x)`。把库选择接到现有 EDA observable 基础设施上。
- **WCE 不可加**（仅上界 `≤ Σ 2^col · max|e|`），最终 WCE 必须从仿真取。

### 1.3 现有流水线会把近似乘法器判为"逻辑错误"丢弃 —— 必须先拆的拦路石
Verilator testbench（[`../utils/template.py`](../utils/template.py) `verilate_main_template`）断言 `out == a*b`；
沙盒侧 `logic_failed` 直接返回 `inf`（[`../send_eda.py`](../send_eda.py)）。近似设计**故意**不等于精确值，会被全部判废。

> **落地前必须加"近似模式"**：testbench 不做 pass/fail，而是累计误差指标；`run_all.sh` / `evaluate_single_routing`
> 不再因不匹配丢弃，而是返回误差指标。**这一步不先做，后面什么都跑不通。**

### 1.4 列权重决定"正负抵消"的真实代数
抵消发生在 `2^col` 加权层：第 0 列一个 `+1` 抵消第 0 列一个 `-1`，但抵消第 3 列一个 `-1` 需要 8 个第 0 列 `+1`。
所以"正负各取几个混着用"要在**加权偏置** `Σ 2^col · E[e]` 上配平，不是数个数。
同时佐证：**低位列做近似最划算**（输出误差权重小）→ "手工低位替换" baseline 是合理起点，搜索也应据此初始化 / 偏置。

---

## 2. 分阶段计划

### 阶段 0 — 误差解析与候选剪枝（纯 Python，秒级，零 EDA 依赖）
- 枚举 3:2 的 2916 个（`|e|≤1`）、2:2 的 36 个；对每个算：
  `signed e(x)`、`weighted_signed_error`(=bias)、`weighted_absolute_error`、`error_rate`、`max_error`、正/负误差概率。
- 误差权重用**逐位概率向量**（可配置），默认 (3/4,1/4)，预留"从 SAIF 读概率"的接口。
- 按输入置换去重 → 3:2 的 660 + 2:2 的 36 个待综合代表（cost 各异）。
- 产物：`Appr_Comp/cand_32.json`、`Appr_Comp/cand_22.json`（含每候选真值表、全部误差指标、`canon_key`、`is_symmetric`、分组）。
- **已实现**：[`enumerate_compressors.py`](enumerate_compressors.py)（纯 Python，已跑通并产出上述 JSON）。

### 阶段 1 — DC 表征（一次错峰批量，约 700 个代表，可缓存）

**前半（已实现）：候选 → Verilog**，[`gen_verilog.py`](gen_verilog.py)
- 每个置换等价类只生成一个代表（规范真值表），端口对齐 FA/HA：
  `module comp32_xxxx (a, b, cin, sum, cout);` / `module comp22_xx (a, cin, sum, cout);`（2:2 两输入是 `a, cin`）。
- 逻辑用真值表 minterm 的 SOP 连续赋值，交给 `compile_ultra` 优化；常数输出退化 `1'b0/1'b1`。
- 产物：`rtl/comp32_lib.v`(660) + `rtl/comp22_lib.v`(36) + `rtl/manifest.json`(696)。
- 已用 Python 把每个模块的 SOP 重新仿真核对真值表：**5424 个 (module,pattern) 全对**；
  exact 代表确为 FA(XOR3/MAJ3) 与 HA(a^cin / a&cin)。

**后半（待运行，需错峰）：批量 DC 表征 —— 单 session 循环，不是 700 次 dc_shell**
- **不要**复用 [`../sandbox_base/scripts/run_all.sh`](../sandbox_base/scripts/run_all.sh)：它是 DC→v2lvs→SPICE→XA/VCS 全栈，
  XA 模拟功耗是给整乘法器用的，表征一个几门的小单元完全没必要、且极慢。
- **方法**：写一个专用 `Appr_Comp/scripts/dc_char.tcl`，在**一个 dc_shell session 内**：
  `analyze` 整个 lib 文件一次 → `foreach` 遍历 696 个 module：`elaborate` 为 top → 套用与
  [`../sandbox_base/scripts/dc_run.tcl`](../sandbox_base/scripts/dc_run.tcl) 相同的 NLDM 约束
  （`set_input_transition 0.02`、`set_load 3.0`）→ `compile_ultra` → 输出一行机读 PPA。
  小单元编译各 1~3s，**整批一个远端 job 约 20~40 分钟**，一次许可证、错峰友好。
- **功耗**：用 DC `report_power` + `set_switching_activity`（输入静态概率取 `P(1)`，与误差模型一致），
  **不走 XA/SPICE**。少数 cell 可后续抽样用 XA 校验。
- **延迟**：逐 arc `report_timing -from <in> -to <out>`，得到 Tas/Tac/Tbs/Tbc/Tcs/Tcc（2:2 为 Tas/Tac/Tcs/Tcc）。
  正好填上 [`../trainer/arith_das.py`](../trainer/arith_das.py) 里目前全是 `None` 的 `UFO_MAC_CONSTANT`。
  近似 cell 若丢掉某输入依赖，对应 arc 不存在 → 记 `nan`（这本身就是有用信息）。
  **单元孤立延迟不代表 in-context 关键路径**，最终延迟仍以整乘法器 STA 为准。
- **驱动**：一个 thin driver（仿照 [`../run_power_sweep.py`](../run_power_sweep.py) 的 ssh/rsync）把
  两个 lib.v + dc_char.tcl 推到远端 sandbox，跑 `dc_shell -f dc_char.tcl`，回收 PPA 报告。**一次性，非 700 任务。**
- 缓存库表 `Appr_Comp/library.json`：每 cell `{name, type, bias, nmed, er, wce, max_e, area, leak, dyn, delay_arcs}`，
  按 `canon_key` 索引。**搜索期只读此表，永不重综合。** 改概率假设只需重算误差（Python），PPA 不变。

### 阶段 2 — Pareto 与分组（已实现：`pareto_select.py`）
- 按 `bias` 符号分 **P / N / Z** 三组（comp32: P=468/N=176/Z=15；comp22: P=26/N=8/Z=1）；
  误差轴用 `wae=E[|e|]`，每组在 (wae, area)/(wae, power)/(wae, delay) 求 Pareto front。
- 选代表时**只保留真正省面积的 cell（area<exact）**，再限 `wae<=cap`(默认0.5)，按 wae 升序均匀取 k(默认3)。
  记录 `bias` 供 +/- 配对。命名 `comp{32,22}_apx_{pos,neg}_{i}` + `comp{32,22}_exact`。
- 已选 **12 个代表** -> `Appr_Comp/selected_compressors.json`：
  comp32 exact + pos1(+.031/省5%) pos2(+.078/省55%) pos3(+.188/省62%) + neg1(-.016/省9%) neg2(-.047/省26%) neg3(-.375/省66%)；
  comp22 exact + pos1(+.188/省46%) pos2(+.250/passthrough) + neg1(-.062/省48%) neg2(-.250)。
- 产物图：`outputs/2026-06-18_appr_comp_pareto/pareto_comp32.png` / `pareto_comp22.png`（4 面板：误差-面积/功耗/延迟 + bias 谱）。
- 直觉印证：真正省面积的是**丢掉输入依赖**的 cell（如 drop-cin `comp32_a994` bias=-0.016 省9%）。
- **待补**：对这 12 个 cell 跑 `char_driver.py --arcs --limit ...` 拿 6-arc 延迟填 `UFO_MAC_CONSTANT`。

### 阶段 3 — 接入 ARITH-DAS（核心工程）
**数据结构改动**：给压缩器加"类型"维度。最小侵入：
- 把 `declare_fa(name, ins, sum, carry, cell="FA")` 参数化，emit / assignment 路径携带 per-slot 的 cell 名；
- 在 netlist 末尾追加被用到的近似模块体（与现在追加 `FA_verilog_src` 同位置，[`../utils/mul.py`](../utils/mul.py) 约 L270）。

**Phase A（手工 baseline，不动搜索）**：按"低于某列阈值 → 用某近似 cell"打标签，先把方式 A 跑通，
拿到**第一个端到端可测的近似乘法器**。

**Phase B（搜索类型）**：在 `CompressorGraph` / 策略 logits 上加**类型头**（每压缩槽从集合 T 选 cell）。
误差项走阶段 0 的闭式 `Σ_type softmax · e_type · 2^col`，便宜且可微。
集合 T 例：
```
T = { exact_32, apx_32_pos_1..3, apx_32_neg_1..3,
      exact_22, apx_22_pos_1..3, apx_22_neg_1..3 }
```

**Reward**：扩展已有通道
```
R = −wA·Â − wD·D̂ − wP·P̂ − λ_abs·NMED − λ_bias·|bias|
```
- 分开 `|bias|` 与 NMED 很关键：抵消的本质是用 +/− 混合把**期望偏置**压到 0，而 NMED / 方差未必降；
  两个一起惩罚才对应"正负抵消"的真实收益。
- 复用已有 `*_scale` 归一化各项；新增 `error_scale`。
- 建议同时支持**约束式**：`min PPA s.t. NMED ≤ budget`（用现成 `area_budget` / `violation_weight` 同款机制），
  更符合"误差可控"。

### 阶段 4 — 最终评估 harness
- 改造 Verilator / VCS testbench：不再 pass/fail，累计 `err = out − a*b`，输出：
  **ER**（≠0 比例）、**MED**（mean|err|）、**NMED**（MED / (2^n−1)²）、**WCE**（max|err|）、**mean signed**（bias）。
- 8-bit 可穷举 2¹⁶ 输入取精确指标；16/32-bit 用 1e6 蒙特卡洛（沿用现成模板规模）。
- 输入分布与误差模型一致（均匀 → PP=1/4），可配置。
- 对比三件套：**exact** / **手工低位替换** / **搜索所得**；报 area·delay·power + ER/MED/NMED/WCE。

---

## 3. 落地顺序（建议）
1. **阶段 0**：枚举 + 误差脚本（纯 Python，最易验证计数与误差恒等式）。
2. **阶段 1**：批量 DC，产出 `library.json`。
3. **阶段 3 拦路石**（1.3）：拆"logic_failed 判废" + 近似模式 testbench。
4. **阶段 3 Phase A**：手工低位替换 baseline → 第一个端到端可测点。
5. **阶段 4**：评估 harness。
6. **阶段 3 Phase B**：类型搜索 + reward 误差项。

---

## 4. 范围与风险清单
- **先只做 `and` 编码**（`configs/config_groups/mul_8_and.yaml` / `mul_16_and.yaml` 已存在）。
  Booth 的 PP 有符号 + 符号扩展常数，(3/4,1/4) 不成立，近似更难，**推后**。
- **拦路石优先级最高**：阶段 3 的 `logic_failed` 判废必须先改，否则全链路返回 `inf`。
- **概率模型**：默认 (3/4,1/4) 仅用于初排；放置 / 最终 reward 用 SAIF 实测逐位概率。
- **WCE**：仅有解析上界，精确值必须仿真。
- **延迟**：单元孤立延迟非关键路径；最终以整乘法器 STA 为准。
- **产物归档**：库 JSON / 代码放 `Appr_Comp/`；图表 / 对比表放 `outputs/YYYY-MM-DD_appr_comp/`（北京时间）。

---

## 5. 任务清单（可勾选）
- [x] 阶段0：3:2 / 2:2 枚举 + `|e|≤1` 剪枝 + 误差指标 → `cand_32.json` / `cand_22.json`（`enumerate_compressors.py`）
- [~] 阶段0：逐位概率接口（`--p-one` 统一值已支持；逐位向量 / 读 SAIF 待补）
- [x] 阶段1：候选 → Verilog（端口对齐 FA/HA，连续赋值）—— `gen_verilog.py`，696 module SOP 全核对通过
- [x] 阶段1：批量 DC 表征 —— `scripts/dc_char.tcl` + `char_driver.py`，**错峰全量跑完，`library.json` 校验通过**
  （696/696，0 error；FA area=10.92/HA=8.74 µm²、tmax≈0.5 ns；唯一 null 是常数 cell comp22_55 的 tmax，语义正确）。
  口径注意：为提速用 `compile`（非 `compile_ultra`），绝对面积比主流程偏小约 30% 但**全库一致、相对排序有效**；
  延迟只存了 `tmax`，6-arc（填 `UFO_MAC_CONSTANT`）留给阶段2 选中的少数 cell 用 `--arcs` 单独补。
  **教训：DC 表征是 contention-bound——9 个并发 dc_run 时 ~37s/cell(~7hr)、空载 ~5-12s/cell；必须错峰且别和自己的 sweep 抢 DC license。**
- [x] 阶段1.5 扩样：`expand_library.py` —— comp22 全 256 空间(160 reps)、comp32 |e|≤2 取 lowest-wae 2000 新 rep。
  增量表征(`char_driver.py --run --incremental`)跑完，`library.json` **=2820 cell(0 error/0 null)**；
  负误差大幅增多(comp32 N 176→828)更利于正负抵消。教训：扩样表征 ~6.4/min 空载、2124 cell ~5.5h，仍 contention/license bound，须先腾空远端。
- [x] 阶段2：P/N/Z 分组 + Pareto front + 代表选取 —— `pareto_select.py`，12 代表 + 2 张图，记录 bias 配对（已在 2820 全库上重跑）
- [x] 阶段2：12 个选中 cell 补 6-arc 延迟（`char_driver.py --arcs --selected`）→ `delay_constants.py`（含 UFO_MAC_CONSTANT 的 FA/HA + APX_COMPRESSOR_DELAY），并 merge 回 `selected_compressors.json`
- [ ] 阶段3：netlist 追加近似模块 + `declare_fa/ha` 加 `cell=` 参数
- [ ] 阶段3：拆 `logic_failed` 判废 + 近似模式 testbench（**拦路石**）
- [ ] 阶段3 Phase A：手工低位替换 baseline
- [ ] 阶段4：误差指标 testbench（ER/MED/NMED/WCE/bias）+ 三件套对比
- [ ] 阶段3 Phase B：类型头 + 闭式误差 reward（`λ_abs·NMED + λ_bias·|bias|` / 约束式）
