# Zhang TCAS-II 2023 — "Area Efficient Approximate 4-2 Compressor and Probability-Based Error Adjustment" — reproduction

> Zhang, Nishizawa, Kimura, *IEEE TCAS-II: Express Briefs* **70(5):1714–1718, 2023**.
> DOI 10.1109/TCSII.2023.3257992 (IEEE Xplore doc 10073562).
> Reproduced as a baseline for the Power_das approximate-multiplier project. The paper's
> **probability-based error adjustment** is conceptually the closed-form Σ bias·2^col this
> project already uses (hence a natural same-family comparison point).

## Status: core design fully recovered (open-access reproduction + verified)
The novel **4-gate 4-2 approximate compressor** was recovered and **verified against the
paper's own probability analysis** (see `reference_wjaets_reproduction.txt`, Table I/II):
- `C = p1 | p2`            (weight +1)
- `S = (p3 | p4) & ~(p1 ^ p2)`   (same weight)
- value = 2C+S vs exact popcount → errors only at {0011:−1, 0100:+1, 0111:−1, 1000:+1, 1011:−1, 1111:−1}
- ⇒ +1 rate = 54/256 = 0.211, −1 rate = 16/256 = 0.0625 — **exactly matches the paper.**

Variants reproduced: **Proposed** (uniform proposed compressor) and **ProposedH** (hybrid:
Esposito compressors @ level 1 + proposed @ level 2 + constant approximation + error-correction AND gate).

## Optional — drop the original PDF here for exact column assignment + error-table validation
```
/home/lee/Baselines/Zhang_TCASII23_ProbAdj/paper_zhang_tcasii23.pdf
```
Used to pin the exact per-column compressor assignment (Fig. 2 dot diagram) and to validate
reproduced ER/NMED/MRED against the paper's tables. Without it I reconstruct from the
published parameters (8-bit: low-4-cols constant `0110`; 16-bit: 6 const cols + 10 approx cols).
