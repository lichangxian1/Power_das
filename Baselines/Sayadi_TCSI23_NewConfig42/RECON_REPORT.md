# Sayadi TCSI'23 复现报告（2026-07-12）

> 论文：Sayadi/Timarchi/Sheikh-Akbari, "Two Efficient Approximate Unsigned Multipliers by
> Developing New Configuration for Approximate 4:2 Compressors", IEEE TCAS-I 70(4), 2023.
> 复现目标：16-bit proposed-mul1 / proposed-mul2 的 RTL + 本项目统一口径 PPA。
> 规格细节见 [SPEC.md](SPEC.md)。

## 一句话结论
16-bit 两个设计已复现为可综合 RTL（`rtl/sayadi_mul{1,2}_16.v`，模块接口 `MUL(clk,a,b,out[30:0])`），
RTL==golden verilator 全对（2M+角点 0 mismatch），统一口径 DC+XA 完成：
**mul1: 275 µm² / 0.169 mW / 1.10 ns；mul2: 378 µm² / 0.224 mW / 1.26 ns（均 @1.5ns 约束）**。
误差在 **超低精度档**（16M wrap-MED ≈ 63M ≈ 1.5% 满量程），与本项目前沿（MED ≤ 60k）不在同一误差量级——
是误差轴远端的极端对照点，不构成直接支配关系比较。

## 1. 复现路径与校验链
1. **压缩器库**（Table II/III/IV 全 32 个变体）：与 Fig.4/6/7 卡诺图逐格核对一致；
   verilog 表达式与真值表 `verify.check_exprs` 全枚举通过。
2. **8-bit 结构**（Fig.9/10 逐盒转录 + 归约调度规则逆向）：调度器对 8-bit 输出与图**逐列完全一致**
   （stage1: c5[4] c6[4] c7[4,3] c8[4,4] c9[4,3] c10[4,2] c11[4] c12[4]；stage2 同图）。
3. **8-bit 精度 vs Table VII**（65536 穷举）：

   | | ER | NMED | MRED | MaxED |
   |---|---|---|---|---|
   | mul1 论文 | 99.93% | 0.018 | 0.509 | 7120 |
   | mul1 复现 | **99.93%（精确）** | 0.0184 | 0.513 | 10450 |
   | mul2 论文 | 98.86% | 0.017 | 0.151 | 7148 |
   | mul2 复现 | **98.86%（精确）** | 0.0178 | 0.152 | 9811 |

   ER 两位小数精确吻合 + NMED/MRED <1% ⇒ 功能结构正确。MaxED 是单点指标，
   对"哪个 pp 接哪个端口"的排列敏感（图中红/黑点标注自相矛盾处即此类歧义，见 SPEC §6），
   已扫描 4 个 AC6G 盒全部 24 排列不能复现 7120 ⇒ 论文 16-bit MaxED 依赖未披露的端口细节，作 caveat。
4. **16-bit 布局**：论文未给图，按其 Algorithm 1（逐 (stage,col) 槽位 NMED 贪心，2M 固定向量）生成
   （`assignment16.json`）。**16-bit vs Table VIII**（10M 均匀向量）：

   | | ER | NMED | MRED |
   |---|---|---|---|
   | mul1 论文 | 100% | 0.010 | 0.119 |
   | mul1 复现 | **100%** | 0.0147 | 0.094 |
   | mul2 论文 | 99.98% | 0.009 | 0.066 |
   | mul2 复现 | 100.00% | 0.0148 | 0.078 |

   同数量级、ER 吻合；NMED 偏高 ~1.5×、MRED 一升一降。坐标下降 trial-and-error 精修
   （`refine_assignment.py`，论文 Sec III-C 所述后处理，3 遍收敛）只再降 ~2-3%
   （mul1 0.0147→0.0144 / MRED 0.092；mul2 0.0148→0.0143 / MRED 0.073）⇒
   差距是**结构性**的（论文 16-bit 归约树/分区细节未披露，其文本公式 (3n/2−1) 与
   Algorithm 1 公式 (2(2n−1)/3) 对分区边界也互相矛盾）。交付 RTL 用纯 Algorithm-1 结果
   （忠实于论文方法本身）。
5. **RTL == golden**：verilator，2,000,100 向量（2M 随机 + 100 角点组合）×2 设计，0 mismatch。
   （8-bit 版本另做 65536 穷举全对，作发射器冒烟。）

## 2. 统一口径结果（与所有 Power_das baseline 同流程可比）
verilator 16M circular-wrap MED（未截断全积口径）+ DC area(compile) + XA SAIF power，TSMC28 @1.5ns：

| 设计 | 真实 MED | bias | area µm² | power mW | delay ns |
|---|---|---|---|---|---|
| Sayadi mul1-16 | 63,067,481 | +247,430 | **275.0** | **0.169** | 1.10 |
| Sayadi mul2-16 | 63,763,130 | −650,881 | 377.7 | 0.224 | 1.26 |
| (参考) ELEX24 MUL1-16 | 1,701,427 | | 1006.5 | 1.019 | 1.50 |
| (参考) ELEX24 MUL2-16 | 323,910 | | 591.7 | 0.331 | 1.45 |

- mul1 比 mul2 更小/更省（ACFGI sum≡1 → 常数传播消掉大片逻辑；ACFGII 传变量）。
  注意与论文 8-bit Table VI 的相对次序（mul2 略省）不同——位宽/工艺/综合流程不同所致。
- 两设计误差 ≈ 63M（NMED≈1.5%），比本项目帕累托前沿最大误差点（MED≈60k）大 3 个数量级：
  它们是**面积/功耗极小、误差极大**的端点，可作 PPA-误差图右下角的文献极端参照，
  不进入 MED≤60k 区间的支配性比较。
- MC WCE ≈ 6.7e8 仅记录不采信（口径约定：WCE 蒙特卡洛不收敛）。

## 2.5 接入项目对比图（2026-07-12）
- **整乘法器 vs deepk 扫描**：`outputs/2026-07-11_03_mred_deepk_np4/cmp_{area,power}_deepk.png`
  （生成脚本 `scripts/plot_mred_deepk.py` 已加 SAYADI 常量）。统一口径 16M wrap-MRED：
  mul1=0.3991、mul2=0.3399，落在 k24(0.327)–k26(1.02) 之间；但 area/power 均高于 k22
  （mul1 275µm²/0.169mW vs k22 243.9µm²/0.100mW）⇒ **被 deepk 两条臂（纯截断与 cells arith）
  全面支配**：同误差档 deepk 功耗低 ~2.5×。
- **cell 级对比**：`outputs/2026-07-11_cell_pareto/{cell_ppa_vs_error_pareto,substd_cell_pareto}.png`。
  Sayadi cell 同 dc_char 口径（sp=0.25 tr=0.125, compile_ultra）表征结果（`sayadi_cells.json`）：

  | cell | wae/use (LSB) | bias | area µm² | dyn mW | tmax ns |
  |---|---|---|---|---|---|
  | AC6G（16 变体=2 同构类，PPA 相同） | 0.2148 | −0.0039 | 10.75 | 0.407 | 0.47 |
  | ACFGI（sum=1, carry=x_n） | 0.7656 | +0.50 | 0.34 | ~0 | 0 |
  | ACFGII（sum=x_i, carry=x_j） | 0.5313 | −0.25 | 0.00 | 0 | 0 |

  语义注意（2026-07-12 更正）：本项目 comp42n 是 **4 入 3 出（a,b,c,d→sum,carry,cout，无 cin）**，
  Sayadi cell（4 入 2 出）= cout≡0 的特例，**语义兼容可直接进 T42 库**（初版报告误记为
  "5 入 3 出不可替换"）。定位：AC6G 比本项目 T42 菜单
  （wae 0.004–0.03, 17–23µm²）便宜 ~40% 但误差大 ~10×；ACFGI/II 是"零门"极端点
  （满足 sub-std 硬约束，但 wae 0.53–0.77 + 强 bias，只适合极低精度档）。

## 3. 产物清单
- `rtl/sayadi_mul{1,2}_16.v` — 交付 RTL（`assignment16.json` 布局，已验证/已测量）
- `rtl/sayadi_mul{1,2}_8.v` — 8-bit 校验版（Fig.9/10 布局）
- `sayadi_common.py` / `run_algorithm1.py` / `refine_assignment.py` / `generate_rtl.py` / `verify.py` / `run_ppa.py`
- `assignment16.json`（交付布局）、`assignment16_refined.json`（精修版，仅记录）
- `results_ppa.json`、`greedy16.log`、`refine.log`、`ppa_run.log`
- `paper_sayadi_tcsi23.pdf` + `_fig*.png`（图转录用高清裁剪）

## 4. 复现保真度分级
- **可信**：压缩器库、8-bit 结构与调度规则（图级一致 + ER 精确）、16-bit 生成方法（Algorithm 1 原样）、
  RTL 功能（verilator 全对）、统一口径 PPA。
- **有偏差（已量化）**：16-bit NMED ~1.5×（结构性，论文未披露 16-bit 树）；MaxED（端口排列敏感）。
- 作为 baseline 使用没有问题：对比看的是"该架构在我们口径下的真实 误差/面积/功耗"。
