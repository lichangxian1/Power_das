# ELEX4_N — ELEX2024 N-4 近似乘法器复刻 (Power_das 对比 baseline)

见 [RECON_REPORT.md](RECON_REPORT.md) 完整报告，[SPEC.md](SPEC.md) 逆向规格。

## 快速复现
```bash
PY=/home/lee/anaconda3/bin/python
$PY generate_rtl.py                       # 生成 rtl/*.v
$PY golden_model.py                       # 精度 ER/NMED/MRED (对齐论文 Table III)
$PY run_ppa.py outputs/<date>_ppa         # TSMC28 PPA (yosys+OpenROAD)
```

## RTL == golden 一致性验证 (verilator)
```bash
# 先生成验证向量(8-bit 穷举 / 16-bit 3M+角点):
$PY - <<'EOF'
import random; from el4_common import approx_mul, trunc_const
def gen(N,d,fn,n):
    c=trunc_const(N,d); rng=random.Random(12345)
    with open(fn,"w") as f:
        if N<=8:
            for a in range(1<<N):
                for b in range(1<<N): f.write(f"{a} {b} {approx_mul(a,b,N,d,c)}\n")
        else:
            M=(1<<N)-1; P=set((x,y) for x in [0,1,2,M,M-1,1<<(N-1),255,256] for y in [0,1,2,M,M-1,1<<(N-1),255,256])
            while len(P)<n: P.add((rng.randrange(1<<N),rng.randrange(1<<N)))
            for a,b in P: f.write(f"{a} {b} {approx_mul(a,b,N,d,c)}\n")
for N in (8,16):
    for d in ("mul1","mul2"): gen(N,d,f"vec_{d}_{N}.txt",3_000_000)
EOF
# 再逐个 verilate+run:
run_one(){ local t=$1 v=$2; rm -rf obj_$t
  verilator --cc -Wno-WIDTH rtl/el4_cells.v rtl/$t.v --top-module $t --exe tb.cpp --Mdir obj_$t -o sim_$t \
    -CFLAGS "-DDUT_HEADER=V$t.h -DDUT_CLASS=V$t -DVEC_FILE=$v" >/dev/null 2>&1
  make -C obj_$t -f V$t.mk >/dev/null 2>&1; (cd obj_$t && ./sim_$t); }
run_one mul1_8 ../vec_mul1_8.txt; run_one mul2_8 ../vec_mul2_8.txt
run_one mul1_16 ../vec_mul1_16.txt; run_one mul2_16 ../vec_mul2_16.txt
# 期望全部 PASS
```

## 结论一句话
MUL2-8/16 = 可信 baseline(精度贴论文、PPA 同流程可比, PDAP≈0.15×)。
MUL1 精度可用、**PPA 不可用**(行为级 N-4 偏重；要 MUL1 的 PPA 须逐门复刻)。

## 2026-06-24 接入本项目 PPA-vs-error 图（同口径重测）
用与 Power_das 完全一致的口径（verilator circular-wrap 16M MED + DC area + XA power **@1.5ns**）测 16-bit：

| 设计 | 真实 MED | area µm² | power mW | delay |
|---|---|---|---|---|
| MUL1-16 | 1,701,427 | 1006.5 | 1.019 | 1.50 |
| MUL2-16 | 323,910 | 591.7 | 0.331 | 1.45 |

→ 两点都被本项目前沿(MED≤60k)**三轴(误差/面积/功耗)同时严格支配**。叠加图与说明：
`outputs/2026-06-24_dcvs_mine/dcvs_{area,power}_elex.png`（镜像到 `Power_das/outputs/2026-06-24_elex_overlay/`）。
复现脚本：`outputs/2026-06-24_dcvs_mine/replot_with_elex.py`；wrapper(31-bit masked) + DC driver 同目录。
