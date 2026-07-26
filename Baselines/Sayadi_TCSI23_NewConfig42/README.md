# Sayadi_TCSI23_NewConfig42 — Sayadi et al. TCSI 2023 近似乘法器复刻 (Power_das 对比 baseline)

> "Two Efficient Approximate Unsigned Multipliers by Developing New Configuration for Approximate 4:2 Compressors",
> IEEE TCAS-I 70(4), 2023. DOI 10.1109/TCSI.2023.3242558. 28nm TSMC, proposed-mul1 (ACFGI+AC6G) / proposed-mul2 (ACFGII+AC6G)。

见 [SPEC.md](SPEC.md) 逆向规格、[RECON_REPORT.md](RECON_REPORT.md) 完整复现报告。

## 快速复现
```bash
PY=python3   # 需要 numpy + verilator
$PY run_algorithm1.py check8      # 8-bit 结构校验 vs Table VII（ER 应精确 = 99.93 / 98.86）
$PY run_algorithm1.py greedy16    # Algorithm 1 生成 16-bit 布局 -> assignment16.json
$PY refine_assignment.py          # trial-and-error 精修 -> assignment16_refined.json
$PY generate_rtl.py assignment16_refined.json   # -> rtl/sayadi_mul{1,2}_16.v
$PY verify.py                     # RTL == golden (verilator, 2M+corner)
$PY run_ppa.py                    # 统一口径: verilator 16M wrap-MED + DC area + XA power @1.5ns
```

## 结果（16-bit, 统一口径：verilator 16M wrap-MED + DC area + XA power @1.5ns, TSMC28）

| 设计 | 真实 MED | area µm² | power mW | delay ns |
|---|---|---|---|---|
| mul1-16 (ACFGI+AC6G) | 63,067,481 | 275.0 | 0.169 | 1.10 |
| mul2-16 (ACFGII+AC6G) | 63,763,130 | 377.7 | 0.224 | 1.26 |

校验：8-bit ER 与论文 Table VII **精确吻合**（99.93/98.86），NMED/MRED <1%；
16-bit（Algorithm 1 生成，论文未披露布局）ER 吻合、NMED ~1.5× 偏高（结构性，见 RECON_REPORT §1.4）。
RTL==golden verilator 2M+角点全对。误差档位 ≈ 63M MED（NMED 1.5%）——比本项目前沿（≤60k）
大 3 个数量级，是文献极端低功耗参照点。
