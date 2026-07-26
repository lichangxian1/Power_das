# Zhang TCAS-II'23 — reverse-engineered design spec (reproduction)

> Zhang, Nishizawa, Kimura, "Area Efficient Approximate 4-2 Compressor and Probability-Based
> Error Adjustment for Approximate Multiplier", *IEEE TCAS-II* 70(5):1714–1718, 2023.
> DOI 10.1109/TCSII.2023.3257852. Original PDF in this folder; key pages rendered `_pg{2,3,4}.png`.

## 1. Proposed 4-gate approximate 4-2 compressor (the paper's core cell)
4 same-weight inputs p1..p4 → C (weight +1), S (same weight); value = 2C+S.
```
C = p1 | p2
S = (p3 | p4) & ~(p1 ^ p2)        # 2 NOR + 1 XOR + 1 OR  = 4 gates
```
Error vs popcount nonzero only at 6 of 16 patterns: {0011:−1, 0100:+1, 0111:−1, 1000:+1,
1011:−1, 1111:−1}. With P(pp=1)=1/4 → P(+1)=54/256=0.211, P(−1)=16/256=0.0625 (verified,
matches the paper's analysis exactly). All four −1 cases have p3=p4=1, so the error-correction
**AND gate = p3 & p4** adds +1 exactly on (and only on) the −1 cases.

## 2. Esposito approximate compressor — exact gate netlist (from Table I)
Two same-weight outputs, value = min(popcount, 2); no XOR (the cheap cells):
```
4-2:  w1 = (p3|p4) | (p1&p2)      w2 = (p1|p2) | (p3&p4)     # 6 AND/OR gates
3-2:  w1 = a|b                    w2 = c | (a&b)             # 3 gates
```
Derived from the (w2,w1) columns of Table I and **verified** to reproduce the paper's
P(w=0)=135/256 (4-2) and P(w1=0)=36/64, P(w2=0)=45/64 (3-2). These exact cells (vs a generic
≥2-of-4) are what make the synthesised area faithful — Esposito is genuinely cheaper than the
proposed cell (which carries an XOR), matching the paper's premise.

## 3. Two multiplier variants (unified Dadda + truncation, Fig. 2)
- **Truncation / constant approximation** (from Kumar [8]): drop the low columns, output a fixed
  "average" constant. 8-bit: low 4 cols → `0110` (=6). 16-bit: low 6 cols → `011110` (=30).
- **Proposed** (uniform): every inexact-column compression uses the proposed 4-2 compressor
  (proposed-greedy across all reduction stages); exact FA/HA elsewhere; final CPA.
- **ProposedH** (hybrid, Fig. 2): on the tall inexact columns, **Esposito @ level 1** reshapes the
  partial products (raising P(−1), lowering P(+1)), then **one proposed compressor @ level 2** with
  the **error-correction AND gate** on the top column(s); the lowest inexact column (C1, adjacent to
  truncation) stays proposed-greedy. 8-bit finishes exact at level 3; **16-bit re-applies Esposito
  at level 3** (paper: "Esposito's compressors are used at the first and third levels", "two error
  correcting AND gates in the second level"). Final RCA.

Calibrated column assignment (see `zhang_gen.config`):
| N | trunc / const | inexact cols | Esposito cols (L1[/L3]) | proposed-greedy | AND-gate cols |
|---|---|---|---|---|---|
| 8 | 0–3 / 6 | 4–7 | 5,6,7 | 4 | 7 |
| 16 | 0–5 / 30 | 6–15 | 7–15 | 6 | 14,15 |

## 4. Why this maps to the project's closed form
The probability-based error adjustment = adding a compensation constant equal to the expected
truncation/compressor bias — i.e. the closed-form Σ bias·2^col this project already uses. The
reproduced 16-bit ProposedH carries a large residual negative bias (Esposito level-3 saturation),
a concrete example of the uncompensated Σbias·2^col that the project's method removes.

## 5. Files
```
zhang_common.py   proposed/Esposito cell logic + self-test (matches paper +1/−1 rates)
zhang_core.py     dual-mode reduction (GoldenOps=compute / RtlOps=emit gates) -> golden==RTL
calibrate.py      golden metrics vs paper Table III/IV
zhang_gen.py      structural RTL emitter -> rtl/MUL_{proposed,proposedH}_{8,16}.v (口径 MUL top)
run_zhang_ppa.py  16-bit verilator MED + remote DC PPA @1.5ns -> zhang_error.csv / zhang_ppa.csv
tb_eq.cpp         verilator equivalence testbench (golden vectors vs RTL)
golden_validation.txt   reproduced NMED/MRED vs paper
```

## 6. Reproduce
```bash
PY=/home/lee/anaconda3/bin/python3
cd /home/lee/Baselines/Zhang_TCASII23_ProbAdj
$PY zhang_common.py        # cell self-test
$PY calibrate.py           # 8-bit golden NMED/MRED vs Table III
$PY zhang_gen.py           # emit rtl/
$PY run_zhang_ppa.py ppa   # 16-bit MED + remote DC PPA
```
