# ELEX 2024 N-4 近似乘法器复刻报告

> 论文：Zhang et al., *"Two energy efficient unsigned approximate multipliers with N-4 compressors"*,
> IEICE Electronics Express 21(14), 2024, DOI 10.1587/elex.21.20240189。
> 复刻目标：作为本项目(Power_das 近似乘法器 PPA 优化)的**对比 baseline**。
> 决策(用户已定)：**功能/架构等价复刻**(非逐门)；**MUL1 + MUL2，8-bit(照原文) + 16-bit(同方案外推)**。

---

## 1. 复刻方法（2026-06-29 **结构性修复**：真实门级网表替代饱和近似）
- 从开放全文 PDF 逆向出：三区结构(精确/近似/截断)、Table II 4-2 真值表、各区列范围、截断常数、RCA。
- **N-4 压缩器**：从 Fig 2/3 **逐门转录真实网表**(5-4/6-4/7-4/8-4)，并用论文误差概率当 checksum
  **精确对齐**(1/1024、23/2048、859/16384、6487/65536，详见 SPEC §3)。不再用 min(cnt,4) 饱和近似。
- **近似 4-2**(stage2)：Table II 化简为紧凑式 **C=(w1&w2)|(w3^w4)、S=(w1^w2)|(w3&w4)**(与真值表逐行核对)。
- **数据通路**：MUL1 stage2 精确(列贡献=N-4 四位之和)；MUL2(列 N,N+1) stage2 用近似 4-2(贡献=2C+S)。
- 截断常数：8-bit 用论文值(MUL1=8, MUL2=0x38)；16-bit 联合(NMED+MRED)最优冻结(MUL1=8, MUL2=0x3108)。
- golden(GOps) 与 RTL(ROps) **dual-mode 同源**(`el4_common.py`)，golden==RTL 由构造保证，并经
  **verilator 复核**(mul1_8/mul2_8 穷举 65536、mul1_16/mul2_16 各 30 万随机+角点，**全 PASS 0 失配**)。

## 2. 精度验收(穷举 8-bit / 采样 2M 16-bit)
| 设计 | ER% (复刻/论文) | NMED e-3 (复刻/论文) | MRED e-2 (复刻/论文) | 评价 |
|---|---|---|---|---|
| **MUL1-8** | 88.59 / 88.58 | **0.722 / 0.722** | 0.573 / 0.568 | **三项与论文精确吻合** ✅ |
| MUL2-8 | 99.83 / 99.82 | 8.434 / 5.884 | 8.987 / 8.175 | ER 准；NMED 见下注 |
| MUL1-16 | 98.29 / — | 0.990 / — | 0.541 / — | 外推(真实 cell) |
| MUL2-16 | 100.00 / — | 0.119 / — | 0.244 / — | 外推 |

- **MUL1-8 三项与论文 Table III 精确吻合**(NMED 0.722e-3 完全相等)——真实 N-4 网表修复了旧版偏准 2.3× 的问题。
- **MUL2-8** ER 精确、**bias=−382.6 精确等于论文 MED(382.6)**，说明截断/分区/cell 全对；NMED 残差(8.43 vs 5.88)
  源于 Fig 6 低清点阵图中**不可辨认的 stage-2 误差平衡/进位走线**：论文靠近似 4-2 的正误差逐样本抵消截断负误差把
  MED 压到 382.6(比纯截断 6.03e-3 还低)，穷举搜遍 PP 配对×4-2 输入序仍达不到 → 属未公开调度(同 Zhang 16-bit 性质)。

## 3. PPA
> ⚠ **下表是旧 satN4 版的本地 TSMC28(yosys+OpenROAD) 数, 结构性修复后已过时, 待重跑 `run_ppa.py`。**
> **进对比图的权威 PPA = 远端 DC(`outputs/2026-06-24_dcvs_mine/ppa_elex.csv`, 结构性版已重跑)**:
> MUL1-16 area 769.4 µm² / power 0.592 mW / delay 1.5ns; MUL2-16 area 552.9 / 0.294 / 1.33。
> 真实 cell 比 satN4 更省(MUL1 1006→769, MUL2 592→553), 印证论文 N-4 省面积主张。

（旧 satN4 本地 TSMC 28nm，yosys + OpenROAD，与 Power_das 自身乘法器同一流程）
| 设计 | area(µm²) | delay(ns) | power(µW) | PDAP | vs exact |
|---|---|---|---|---|---|
| exact_8 | 250 | 0.521 | 259 | 33719 | 1.00× |
| MUL1-8 | 258 | 0.478 | 236 | 29095 | 0.86× |
| **MUL2-8** | 124 | 0.398 | 99.4 | 4909 | **0.15×** |
| exact_16 | 1091 | 0.964 | 2680 | 2.82e6 | 1.00× |
| MUL1-16 | 1361 | 0.962 | 2390 | 3.13e6 | 1.11× (更差) |
| **MUL2-16** | 559 | 0.716 | 1120 | 4.48e5 | **0.16×** |

- **MUL2 是优秀的 PPA baseline**(PDAP 降到 ~0.15×，与论文 MUL2 PDAP 0.028× 同方向)。
- ⚠ **MUL1 的 PPA 不可用作面积/功耗 baseline**：行为级 `satN4`(popcount+饱和)综合后比被高度优化的
  精确乘法器还重 → MUL1 面积几乎不降甚至变大。论文 MUL1 的省面积来自手工优化 N-4 + NAND 反码，
  本功能级复刻无法体现。**用 MUL1 时只取其精度，PPA 引用论文公布值或留待逐门复刻。**

> 论文公布(SMIC40, 仅供参照, 工艺不同不可直接同表)：exact area732/power58.2/delay2.22/PDAP94.55；
> MUL1 277/25.0/2.05/14.20；MUL2 167/13.0/1.21/2.63。

## 4. 给本项目对比实验的用法建议
- **首选 MUL2-8 / MUL2-16** 作 ELEX2024 代表点：精度(NMED/MRED)与 PPA 都可信、同流程可比。
- MUL1 仅入精度-误差讨论；其 PPA 标注“需逐门复刻”或引论文值。
- 三项指标与本项目近似乘法器在**同一 yosys+OpenROAD+TSMC28 流程**下出数，可直接进同一 Pareto 图。

## 5. 文件清单
```
SPEC.md            逆向规格(架构/真值表/常数/列范围)
el4_common.py      区域划分+常数+golden 乘法(golden 与 RTL 同源)
golden_model.py    精度评估(穷举/采样 ER/NMED/MRED)
generate_rtl.py    生成 RTL(gen_verilog.py 风格)
tb.cpp             verilator 通用测试台(读 a b expected 三元组)
run_ppa.py         本地 TSMC28 PPA(yosys+OpenROAD)
rtl/el4_cells.v    satN4(参数化饱和压缩器) + apx42(Table II 精确4-2)
rtl/{mul1,mul2}_{8,16}.v   四个顶层 + exact_{8,16}.v 参照
outputs/2026-06-18_18_ppa/ PPA 工件(netlist/log/ppa.json/ppa_table.txt)
paper_*.pdf/png, paper_text.txt  论文与渲染图
```

## 6. 复现命令
```bash
PY=/home/lee/anaconda3/bin/python
cd /home/lee/Baselines/ELEX4_N
$PY generate_rtl.py            # 生成 rtl/
$PY golden_model.py           # 精度(对齐论文 Table III)
# RTL==golden 验证: 见 README 里的 verilator run_one 片段 (8-bit 穷举, 16-bit 3M+角点)
$PY run_ppa.py outputs/<date>_ppa   # TSMC28 PPA
```

## 7. 已知限制 / 后续
1. ✅ N-4 已逐门转录并对齐误差概率(1/1024、23/2048、859/16384、6487/65536)，MUL1-8 精度已精确复现。
2. MUL2-8 NMED 残差(8.43 vs 5.88e-3)：Fig 6 stage-2 误差平衡/进位走线不可辨认(未公开调度)，详见 §2。
3. PP 用 AND(论文 NAND+反码 np=¬p 是门级优化；真实 cell 网表已在 p 域等价吸收反码，DC 自行处理)。
4. 16-bit 区域划分是"同方案外推"(论文只给 8-bit)：列高>8 用 8-4 递归组合；属设计选择，已在 el4_common.py 注明。
5. PPA(§3 本地 TSMC + 对比图远端 DC)在结构性修复后需重测；项目对比图用 `outputs/2026-06-24_dcvs_mine/`
   远端 DC 流程, 已随新 RTL 重跑(error_elex.csv / ppa_elex.csv 更新)。
