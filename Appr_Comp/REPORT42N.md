# 原生 4:2 近似压缩器库（comp42n，2026-07-07）

替代旧 pair32 库（5 输入带 cin、串行拼装、无 PPA 表征）的重设计。端口 `(a,b,c,d,sum,carry,cout)`
与搜索原语 CT42 完全对齐，无 cin 遗留问题。

## 流水线

1. **生成**（`gen_comp42_native.py`）：结构化家族（丢输入/OR合并/cout0/sum简化/零偏置对/exact微扰）
   + 有偏随机填缝 + 旧 pair32 库 cin=0 投影，|e|≤1 为主（少数家族 |e|=2），
   S4 输入置换规范型去重 → **1999 个代表**（P=1175/N=755/Z=69）。
   (carry,cout) 编码贴 CT42_BAL 拆分（h=1 跟随 exact carry 位）。SOP 仿真自校验 16×3 全过。
2. **DC round1**（`char42n_driver.sh`，16 session 并行，`compile -map_effort medium`，t28 同
   dc_char.tcl 口径）：**1999/1999 + 5 锚点全部成功** → `library42_native.json`。
3. **选型**（`build_select42n.py`）：闸门 area<CT42_BAL(17.304) ∧ tmax≤0.557 ∧ maxe≤2 → 池 95/1999
   （随机真值表绝大多数综合不过 exact，印证"结构化打底"决策）；P/N/Z 各组 (wae,area)∪(wae,dyn)
   Pareto + log-wae 均匀取点 → 17 个。
4. **DC round2**（`compile_ultra` + 三输出 arc，主流程口径）复检：5 个在 ultra 下超锚点被剔
   → **最终 12 个代表**（`selected_compressors42_native.json`，含 medium/ultra 双档 PPA）。
5. **端到端冒烟**：trainer loader 加载 13 类型（exact+12）、native4 免 cin 发射、8-bit 带 2 个
   近似 cell 的 MUL verilator 穷举，circular-wrap max|e|=16 ≤ 解析上界 24。

## 最终 12 代表（vs exact CT42_BAL ultra 18.14µm² / tmax 0.56）

| cell | 组 | wae | bias | maxe | area_ultra | ΔA% | tmax |
|---|---|--:|--:|--:|--:|--:|--:|
| comp42n_1d21d225 | N | 0.023 | −0.023 | 1 | 14.62 | −19.4 | 0.54 |
| comp42n_6f1aced0 | N | 0.047 | −0.047 | 1 | 14.62 | −19.4 | 0.52 |
| comp42n_23d463fb | N | 0.109 | −0.102 | 1 | 13.78 | −24.1 | 0.55 |
| comp42n_941c5d96 | N | 0.117 | −0.117 | 1 | 17.14 | −5.6 | 0.57 |
| comp42n_55d1f840 | N | 0.141 | −0.141 | 1 | 14.28 | −21.3 | 0.50 |
| comp42n_494aa88e | P | 0.180 | +0.180 | 1 | 11.26 | −38.0 | 0.52 |
| comp42n_88e68432 | Z | 0.281 | 0.000 | 1 | 13.94 | −23.1 | 0.49 |
| comp42n_3a20fd14 | N | 0.328 | −0.328 | 1 | 17.98 | −0.9 | 0.51 |
| comp42n_537f16a9 | P | 0.438 | +0.438 | 1 | 15.46 | −14.8 | 0.52 |
| comp42n_21b8eab9 | P | 0.438 | +0.125 | 1 | 10.42 | −42.6 | 0.51 |
| comp42n_f871669b | N | 0.531 | −0.250 | 2 | 11.76 | −35.2 | 0.53 |
| comp42n_43873371 | P | 0.578 | +0.125 | 2 | 9.74 | −46.3 | 0.49 |

对比：3:2/2:2 库的近似 cell 单元收益是几 µm² 量级；这里最强 cell 一个省 8.4µm²（−46%），
且全部通过时序闸门（v1 CT42 之死的直接防护）。零偏置 Z cell（88e68432）与 ±bias 谱系齐备，
正负抵消配对空间完整。

## 训练用法

```
scripts/train_dc.py ... --use_ct42 \
  --approx42_library_path Appr_Comp/selected_compressors42_native.json \
  --approx42_rtl_path Appr_Comp/rtl/comp42n_lib.v \
  --approx42_max_types 13
```
（trainer 已适配：`pattern_bits==4` 的 cell 发射时不接 `.cin`；旧 5 端口库仍兼容。）

## 口径与教训

- 误差指标 P(1)=1/4（第一级 PP 口径），LUT 全存库，可按实测概率重算。
- medium 与 ultra 排序不完全一致（5/17 在 ultra 下翻车）——**选型必须用最终流程口径复检**。
- 共享机 16 session 并行可行但接近极限（负载 39→60）；进程存活检查要匹配
  `common_shell_exec`/tcl 名而非 `dc_shell`（本次曾因此误诊"集体阵亡"并多开了 8 个重复 session）。
