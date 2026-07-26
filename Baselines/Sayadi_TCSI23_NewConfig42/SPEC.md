# Sayadi TCSI'23 复刻规格 — "Two Efficient Approximate Unsigned Multipliers by Developing New Configuration for Approximate 4:2 Compressors"

> Sayadi, Timarchi, Sheikh-Akbari, *IEEE TCAS-I* **70(4):1649–1659, 2023**. DOI 10.1109/TCSI.2023.3242558.
> 本文件 = 从论文 PDF（`paper_sayadi_tcsi23.pdf`）逆向提取的可复刻规格。
> 目标：16-bit proposed-mul1 / proposed-mul2 RTL，接入本项目统一口径（verilator 16M wrap-MED + DC area + XA power @1.5ns）。

## 1. 三种近似 4:2 压缩器（无 Cin/Cout；carry 权重 = 列权重×2）

### AC6G（6 门，16 变体，Table II）
- sum = x1|x2|x3|x4（全部 16 个变体功能相同，分组只影响门共享）
- carry 按 Table II（已与 Fig.4 AC6G-12 卡诺图逐格核对一致）：
  `AC6G-n: carry = (xa & (xb|xc)) | (xd & xe)`，见 `sayadi_common._AC6G_CARRY`。
- CPD = 2·tOR + tAND。

### ACFGI（0 门，4 变体，Table III）
- sum = 常数 1，carry = x_n（n=1..4）。

### ACFGII（0 门，12 变体，Table IV）
- sum = x_i, carry = x_j：(1,2),(1,3),(1,4),(2,1),(2,3),(2,4),(3,1),(3,2),(3,4),(4,1),(4,2),(4,3)。

## 2. n-bit 乘法器结构（Sec III-C/D）
- 列 1..2N−1（列 c 权重 2^(c−1)），压缩级数 = log2(N)−1（8-bit: 2 级；16-bit: 3 级），末级 2 行 RCA。
- **截断区**：列 1..N/2 完全丢弃（无补偿常数）。
- **中列区**（mul1 用 ACFGI / mul2 用 ACFGII）：列 N/2+1 .. floor(2(2N−1)/3)（8-bit: 5–10；16-bit: 9–20）。
- **高列区**（AC6G）：.. 2N−3（8-bit: 11–13；16-bit: 21–29）。更高列 exact HA/FA。

## 3. 归约调度（从 Fig.9/10 逆向；对 8-bit 与图**逐列完全一致**）
每级、每列（升序）取**最少盒数** b 满足：
```
leftover + b + b_prev_col ≤ H_next      （H_next 序列：16-bit [8,4,2]；8-bit [4,2]）
```
- 盒自顶向下装 4,4,...,余数（余 1 时重排成 3+2，不允许 1 输入盒）。
- 下一级该列输入顺序 = [剩余 pp…, 本列盒 sum S1..Sb, 左邻列盒 carry C1..Cb']（与图中绘制顺序一致）。
- **最后一级中列区把所有 ≥2 输入全部装盒**（ACFGI/II 零门成本，缩短末级加法器）；高列区仍最少盒（HA/FA 有门成本）。
- 不足 4 输入的盒**顶对齐**（x_{k+1..4}=0；Fig.9 col10 stage-1 黑点证实）。
- 高列区盒：4 输入→AC6G（可搜索），3→exact FA，2→exact HA。
- pp 行序：按 b 位下标升序（Fig.8 顶行 = b0·a_*）。

## 4. Algorithm 1（每 (stage, column) 槽位一个变体，NMED 贪心）
- 逐级、逐列（升序），对该槽位所有候选（ACFGI 1-4 / ACFGII 1-12 / AC6G 1-16）全树模拟取 NMED 最小。
- 未指定槽位建模为"理想压缩器"（值保持，超出 2c+s 表示范围的部分记入修正项）。
- 8-bit 用 65536 穷举；16-bit 用 2M 固定种子随机向量（论文用 MATLAB 模拟，向量数未披露；Table VIII 精度用 10M 均匀向量复测）。

## 5. Fig.9 / Fig.10 转录（8-bit 校验用，见 run_algorithm1.PAPER_8）
| slot | mul1 | mul2 |  | slot | mul1 | mul2 |
|---|---|---|---|---|---|---|
| s1 c5 | ACFGI-4 | ACFGII-1 | | s2 c5 | ACFGI-4 | ACFGII-1 |
| s1 c6 | ACFGI-4 | ACFGII-1 | | s2 c6 | ACFGI-4 | ACFGII-1 |
| s1 c7 | ACFGI-4 | ACFGII-1 | | s2 c7 | ACFGI-4 | ACFGII-1 |
| s1 c8 | ACFGI-4 | ACFGII-1 | | s2 c8 | ACFGI-4 | ACFGII-1 |
| s1 c9 | ACFGI-4 | ACFGII-5 | | s2 c9 | ACFGI-4 | ACFGII-1 |
| s1 c10 | ACFGI-2 | ACFGII-11 | | s2 c10 | ACFGI-3 | ACFGII-10 |
| s1 c11 | AC6G-12 | AC6G-12 | | s2 c11 | AC6G-7 | AC6G-7 |
| s1 c12 | AC6G-14 | AC6G-14 | | s2 c13 | AC6G-7 | AC6G-7 |

其余：s2 c12 = HA[S,C]，s2 c14 = HA[pp,pp]，c15 直通；末级 RCA，中列 sum 常 1 → FA1/HA1（综合期常数传播等效）。

## 6. 图中已发现的不一致（不影响功能转录）
- 红/黑点编码 = 端到端是否被使用；但 Fig.9 col10-box2 与 Fig.10 col10-box2 同一槽位的黑点标注互相矛盾（一个 top-aligned 一个 bottom-up），Fig.10 col5 row5 在 stage-1 画红、stage-2 画黑。→ 端口顺序统一取 top-down；该歧义对分布类指标（ER/NMED/MRED）无影响（同列 pp 独立同分布），只影响 MaxED 这类单点指标。

## 7. 校验结果锚点
- **Table VII（8-bit 穷举）**：mul1 ER 99.93% / NMED 0.018 / MRED 0.509 / MaxED 7120；mul2 ER 98.86% / 0.017 / 0.151 / 7148。
  我方 Fig.9/10 布局复现：mul1 **ER 99.93%（精确）**/ 0.0184 / 0.513 / 10450；mul2 **ER 98.86%（精确）**/ 0.0178 / 0.152 / 9811。
  → ER 两位小数精确吻合 + NMED/MRED <1% 偏差 = 功能结构正确；MaxED 偏差来自端口排列敏感性（上节），已作为 caveat 记录。
- **Table VIII（16-bit, 10M 均匀向量）**：mul1 ER 100% / NMED 0.010 / MRED 0.119；mul2 ER 99.98% / 0.009 / 0.066。16-bit 结构论文未给图，由 Algorithm 1 生成（与论文方法一致）。
