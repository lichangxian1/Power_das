# Scalable approximate-multiplier families — Power_das baselines

Parametric AxM families (one knob → a whole MED-vs-PPA curve), reproduced in the project 口径
(16-bit `MUL(clk,a,b,out[30:0])` → verilator circular-wrap MED + remote DC area + XA power @1.5ns).
These cover the **high-error / low-PPA region** and add architecture diversity beyond EvoApprox,
truncation, and the compressor designs (Zhang/ELEX).

## Families
| family | ref | knob | mechanism | region |
|---|---|---|---|---|
| **DRUM** | Hashemi, Bahar, Reda, *ICCAD 2015* | `k` = significant bits kept | leading-one dynamic truncation, LSB forced to 1 (**unbiased**), k×k multiply + shift | wide sweep, near-unbiased |
| **Mitchell** | J. N. Mitchell, *IRE Trans. 1962* | `W` = mantissa bits kept | logarithmic: `log2(a·b)≈log2 a+log2 b`, piecewise-linear; **always underestimates** | distinct log-error region |

Both are same-source golden↔RTL (behavioural RTL = the actual DRUM/Mitchell datapath: LOD +
shift + small multiply / fraction adder), verified golden==RTL by verilator (0 mismatches,
400k random+corner). DRUM is near-unbiased (bias/MED≈0); Mitchell has the characteristic
**−100% bias/MED** (a clean illustration of the Σbias·2^col the project's adjustment removes).

## Reproduce
```bash
PY=/home/lee/anaconda3/bin/python3
cd /home/lee/Baselines/scalable_axmul
$PY drum.py                       # DRUM golden self-test
$PY mitchell.py                   # Mitchell golden self-test
$PY run_family.py drum verify     # golden==RTL spot check
$PY run_family.py drum ppa        # emit + MED + remote DC sweep -> drum_results.csv
$PY run_family.py mitchell ppa    # -> mitchell_results.csv
$PY plot_families.py              # overlay both curves onto the project plot
```

## Files
```
drum.py / mitchell.py     golden + behavioural RTL emitter + self-test (SWEEP = knob values)
run_family.py             verify (golden==RTL) + MED + remote DC sweep -> {fam}_results.csv
plot_families.py          overlay DRUM+Mitchell vs ours + baselines -> {area,power}_with_families.png
tb_eq.cpp                 verilator equivalence testbench
```
Shared 口径 harness: `/home/lee/Baselines/common_eval.py`.
