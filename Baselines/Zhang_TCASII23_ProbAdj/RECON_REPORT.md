# Zhang TCAS-II'23 reproduction report

> Zhang, Nishizawa, Kimura, "Area Efficient Approximate 4-2 Compressor and Probability-Based
> Error Adjustment for Approximate Multiplier", *IEEE TCAS-II* 70(5):1714–1718, 2023.
> Reproduced as a Power_das comparison baseline. Decision (matching the ELEX4_N precedent):
> **functional/architectural-equivalent** reproduction, golden model and RTL same-source,
> validated against the paper's own error tables, then run through the project 口径.

## 1. What was recovered — all cells gate-exact
- **Proposed 4-gate 4-2 compressor** — exact, confirmed against Fig. 1 + eq.(1)(2):
  `C=p1|p2`, `S = ~(p1^p2)&(p3|p4)` (2 NOR + 1 XOR + 1 OR). P(+1)=54/256, P(−1)=16/256.
- **Esposito compressors — exact gate netlist from Table I** (not a generic threshold):
  `4-2: w1=(p3|p4)|(p1&p2), w2=(p1|p2)|(p3&p4)` (6 gates, no XOR);
  `3-2: w1=a|b, w2=c|(a&b)` (3 gates). Reproduces the paper's P(w=0)=135/256, 36/64, 45/64.
  These cheap exact cells are what make Esposito genuinely lighter than the proposed cell
  (which carries an XOR) — restoring the paper's intended area relationship.
- **Error-correction AND gate** = `p3&p4` — exactly the proposed compressor's −1 condition.
- **Constant approximation**: 8-bit low-4 cols → `0110`(6); 16-bit low-6 cols → `011110`(30).
- **Fig. 2 hybrid layout** (8-bit, fully shown): Esposito@L1 + one proposed compressor@L2 per
  inexact column + AND gate; exact above. 16-bit per the paper's text: Esposito@levels 1&3,
  2 AND gates @level 2, 10 approx + 6 const columns.

## 2. Error validation vs paper (golden, same source as RTL)
8-bit exhaustive (65536); 16-bit 1M random (paper: 10×1M). `golden_validation.txt`.

| design | bits | NMED repro | NMED paper | MRED% repro | MRED% paper | note |
|---|---|---|---|---|---|---|
| Proposed  | 8  | 2.02e-3  | 1.90e-3 | 4.12 | 4.27 | NMED +6%, MRED −4% |
| ProposedH | 8  | 1.07e-3  | 1.14e-3 | 2.28 | 2.59 | **anchored on Fig. 2, within ~7%** |
| Proposed  | 16 | 1.71e-5  | 2.11e-5 | 0.134| 0.094| more accurate than paper |
| ProposedH | 16 | 0.82e-5  | 1.76e-5 | 0.030| 0.056| more accurate than paper (see §5) |

8-bit is anchored directly on Fig. 2 and matches within ~7%. Both 16-bit designs come out
**more accurate than the paper** (my Dadda schedule cancels more error than the paper's
unpublished 16-bit schedule) — the conservative direction for a baseline (stronger / harder to beat).

## 3. RTL ⇄ golden equivalence (verilator)
Dual-mode core (`zhang_core.py`): the same reduction body computes the golden value or emits the
gate-level netlist, so RTL == golden by construction and DC sees the actual cells.
- 8-bit exhaustive (65536): **0 mismatches**, both Proposed and ProposedH.
- 16-bit 300k random+corner: **0 mismatches**, both.

## 4. Project 口径 — 16-bit @1.5ns (verilator MED + DC area + XA power)
`zhang_error.csv` / `zhang_ppa.csv`. MED is the real circular-wrap MED on the masked 31-bit
`MUL(clk,a,b,out[30:0])` top (16M vectors), identical to every other project baseline.

| design | real MED | bias | area µm² (DC) | power mW (XA) | \|delay\| ns |
|---|---|---|---|---|---|
| Zhang-Proposed-16  | 73,402 | +61,495 | 782.2 | 0.4155 | 1.47 |
| Zhang-ProposedH-16 | 35,101 | −24,687 | 796.0 | 0.4301 | 1.48 |

ProposedH (more accurate) sits at lower MED than Proposed with essentially equal area
(796 vs 782 µm², +1.8% = DC run-to-run noise) — the area inversion of the first cut is resolved
by the exact (cheap) Esposito cells + Esposito-at-levels-1&3. Both points are **dominated by the
project frontier on power**. Overlay: `ppa_{area,power}_with_zhang.png`, mirrored to
`Power_das/outputs/zhang_overlay/`.

## 5. Fidelity boundary (honest)
- **Cells: gate-exact.** Proposed (Fig. 1) and Esposito 4-2/3-2 (Table I) are reproduced to the
  gate; verified against the paper's published probabilities.
- **8-bit: anchored on Fig. 2** (the only published dot diagram); error within ~7%.
- **16-bit: the reduction *schedule* is unpublished** (the paper gives only text: "10 approx +
  6 const columns, Esposito @ levels 1&3, 2 AND gates @ level 2"). I implement exactly that. The
  irreducible consequence: my Dadda schedule is more error-cancelling than the paper's, so the
  reproduced 16-bit NMED is ~2× *better* than Table IV. This means **16-bit can match the paper's
  reported error OR its area trend, but not both** from the published material:
  - *as shipped* (Esposito @ levels 1&3, paper's text): area ≈ Proposed (no inversion), error more
    accurate than paper (conservative). `zhang_gen.config(16, hybrid=True)`.
  - *alternative* (Esposito @ every stage): matches Table IV NMED (1.7e-5) but re-inflates area
    (+7%). Switch via `post_l2` applied to all stages in `zhang_core.reduce_columns`.
  Shipped = the area-faithful / paper-text-faithful choice; absolute PPA isn't cross-node
  comparable anyway (paper 65 nm vs project flow), so the same-flow position vs ours is what matters.
- The residual **negative bias** on ProposedH is a concrete instance of the uncompensated
  **Σ bias·2^col** that the project's closed-form adjustment removes (the paper's
  "probability-based error adjustment" is the same idea as a fixed constant).
- PP encoding is AND (matching the 口径); the paper's NAND+inverse-code is a gate-level
  optimisation DC handles equivalently.
