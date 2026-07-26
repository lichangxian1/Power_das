# Sabetzadeh TCAS-II 2023 — "An Ultra-Efficient Approximate Multiplier With Error Compensation" — reproduction

> Sabetzadeh, Moaiyeri, Ahmadinejad, *IEEE TCAS-II: Express Briefs* **70(2):776–780, 2023**.
> DOI 10.1109/TCSII.2022.3215065 (IEEE Xplore doc 9920015).
> Reproduced as a **strong-SOTA baseline** for the Power_das approximate-multiplier project
> (same 口径: 16-bit `MUL(clk,a,b,out[30:0])` → verilator MED + DC area + XA power @1.5ns).

## ⚠️ ACTION NEEDED — drop the paper PDF here
The exact **Error Compensation Module (ECM)** logic and the LSB-half **constant** value are
not available in any open-access source. To match the design gate-faithfully, place the paper at:

```
/home/lee/Baselines/Sabetzadeh_TCASII23_Ultra/paper_sabetzadeh_tcasii23.pdf
```

Once it's there I will extract: (1) which product columns become the constant + its value,
(2) the ECM inputs/equations and which output bits it corrects, (3) the reported
ER/NMED/MRED/MED and area/power/delay/EDP for the 8-bit design (validation targets).

## Architecture (confirmed from abstract + secondary sources, pending PDF for exact logic)
- 8×8 unsigned. Product P[15:0] split in half.
- **LSB half P[7:0] → a constant** "compensation term" (statistically chosen).
- **MSB half P[15:8] → computed precisely.**
- **ECM**: low-complexity OR-based boundary correction recovering the carry lost from the
  truncated half. EDP −77% vs exact, −54% vs prior approximate; PDAP −67%, MRED ≈1.27% (8-bit).
