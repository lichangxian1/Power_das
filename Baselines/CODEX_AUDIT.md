# Approximate-Multiplier Baseline Audit

Date: 2026-06-29  
Scope: local read/verify only. I did not run any remote SSH/DC/PPA step.

## 1. `scalable_axmul/drum.py`

**Verdict: ✅ CORRECT** — DRUM math and emitted RTL implement the same keep-`k` leading-significant-bit function with LSB forced to 1.

**Evidence**

- Re-derived `drum_operand(x,k)`: find leading one `l`, exact if `l < k`, otherwise shift by `s = l-(k-1)` and use `(x >> s) | 1`; product is `(ma*mb) << (sa+sb)`. This is the DRUM unbiasing/round-to-odd style step and is exact at `k >= 16`.
- `emit(k)` has the same LOD, same shift, same LSB-forcing condition `ta/tb`, a `K x K` multiply, and `out = full[30:0]`.
- Ran requested local self-test:
  - `python3 drum.py`: `k=16 exact: OK`.
  - Representative self-test: `k=8 MED=3787985.4`, `NMED=8.820e-4`, `bias/MED=+0.011`.
- Ran requested Verilator check:
  - `python3 run_family.py drum verify`: `mismatches=0` for the script-selected knobs `k=3,6,9,12,12`, 400000 vectors each.
- Extra all-knob local Verilator check:
  - `k=3..12`: `mismatches=0`, 120000 random+corner vectors per knob.
- CSV consistency:
  - `drum_results.csv` `k=8 MED=3785510.225888`; `MED/2^32 = 8.8138278245e-4`, matching the self-test within MC noise.

**Discrepancies / caveats**

- The family is near-unbiased at coarse knobs, but not uniformly `bias/MED≈0`: in `drum_results.csv`, `k=12` has `bias/MED = 30732.282524 / 237375.859514 ≈ 0.129`. This is small in absolute output scale but not a zero-ratio claim.

## 2. `scalable_axmul/mitchell.py`

**Verdict: ✅ CORRECT** — Mitchell logarithmic approximation and emitted RTL match the golden function.

**Evidence**

- Re-derived the function: for nonzero inputs, `ka,kb` are leading-one positions; fractional parts below the leading one are truncated to `W` bits; `cross = fa*2^kb + fb*2^ka`; `s = ka+kb`; result is `2^s + cross` if `cross < 2^s`, otherwise `2*cross`.
- `emit(W)` implements the same LOD, fractional mask, truncation shifts, `xsum`, `base`, and branch. Zero inputs return zero.
- The model is always underestimating for the sampled self-test; signed bias equals `-MED`.
- Ran requested local self-test:
  - `python3 mitchell.py`: `W=15` pure-Mitchell floor `NMED=9.253e-3`; every printed `bias/MED=-1.000`.
- Ran requested Verilator check:
  - `python3 run_family.py mitchell verify`: `mismatches=0` for script-selected knobs `W=2,5,11,11`, 400000 vectors each.
- Extra all-knob local Verilator check:
  - `W=2,3,4,5,6,8,11`: `mismatches=0`, 120000 random+corner vectors per knob.
- CSV consistency:
  - `mitchell_results.csv` `W=11 MED=40091465.528513`; `MED/2^32 = 9.334521724e-3`, consistent with the self-test `W=11 NMED=9.321e-3` within MC noise.

**Discrepancies / caveats**

- None found beyond the expected Mitchell negative bias/error floor.

## 3. `Zhang_TCASII23_ProbAdj/`

**Verdict: ⚠️ MINOR ISSUE** — current source logic is internally consistent, but checked-in `vec_proposedH_16.txt` is stale, and the 16-bit NMED differs materially from the paper table.

**Evidence**

- Proposed 4-2 cell:
  - `C = p1 | p2`.
  - `S = (p3 | p4) & ~(p1 ^ p2)`.
  - `zhang_common.py` self-test passed; nonzero error map is exactly `{0011:-1, 0100:+1, 0111:-1, 1000:+1, 1011:-1, 1111:-1}`.
  - Probability check passed: `P(+1)=54/256`, `P(-1)=16/256`.
- Esposito cells:
  - 4-2 implementation is `w1=(p3|p4)|(p1&p2)`, `w2=(p1|p2)|(p3&p4)`: 6 AND/OR gates, no XOR.
  - 3-2 implementation is `w1=a|b`, `w2=c|(a&b)`: 3 gates, no XOR.
  - Truth-table value is `min(popcount, 2)` for 3 or 4 inputs, matching the paper text.
- Current golden/RTL dual-mode consistency:
  - `proposed_8`: 65536 checked, `mismatches=0`.
  - `proposed_16`: 300000 checked, `mismatches=0`.
  - `proposedH_8`: 65536 checked, `mismatches=0`.
  - `proposedH_16`: regenerated current-golden vectors, 300000 checked, `mismatches=0`.
- Bias accounting:
  - Instrumented the reduction and verified `constant + truncation + Σ(local compressor bias * 2^col)` reconstructs signed bias.
  - 8-bit Proposed: bias `+74.105469`, reconstructed `+74.105469`.
  - 8-bit ProposedH: bias `-14.132812`, reconstructed `-14.132812`.
  - 16-bit Proposed sample: bias `+61150.477`, reconstructed `+61150.477`.
  - 16-bit ProposedH sample: bias `-24904.820`, reconstructed `-24904.820`.
- Shipped metrics:
  - `zhang_error.csv`: Proposed-16 `MED=73401.794644`, `NMED=1.7090187e-5`; ProposedH-16 `MED=35100.967996`, `NMED=8.1725810e-6`.
  - `golden_validation.txt`: 8-bit Proposed `2.018e-3` vs paper `1.900e-3`; 8-bit ProposedH `1.073e-3` vs paper `1.140e-3`.

**Discrepancies / caveats**

- `vec_proposedH_16.txt` is stale. Using it against current `rtl/MUL_proposedH_16.v` produced `254730` mismatches over 300000 vectors. Current golden generation matches current RTL, so this is an artifact drift, not a RTL/golden drift.
- `calibrate.py` prints a candidate search whose best 8-bit hybrid rows are not the shipped `zhang_gen.config(8, True)` setting. The validation file matches the shipped config, not the top rows printed by `calibrate.py`.
- The 16-bit schedule is unpublished and acknowledged. The current 16-bit results are substantially more accurate than paper Table IV:
  - Proposed `1.707e-5` vs paper `2.110e-5` (about 19% lower NMED).
  - ProposedH `0.817e-5` vs paper `1.760e-5` (about 54% lower NMED).

## 4. `ELEX4_N/`

**Verdict: ❌ BUG** — golden and RTL agree, but the implemented design family does not structurally match the stricter SPEC/paper claims.

**Evidence**

- `golden_model.py` and `generate_rtl.py` share `el4_common.py`; the generated RTL is therefore internally aligned with the golden model.
- Verilator checks passed:
  - `mul1_8`: 65536 exhaustive vectors, `PASS`.
  - `mul2_8`: 65536 exhaustive vectors, `PASS`.
  - `mul1_16`: 300000 random+corner vectors, `PASS`.
  - `mul2_16`: 300000 random+corner vectors, `PASS`.
- `python3 golden_model.py`:
  - MUL1-8: `NMED=0.315e-3`, paper says `0.722e-3`.
  - MUL2-8: `NMED=6.172e-3`, paper says `5.884e-3`.
  - MUL1-16: `MED=1699354.33`, sample `NMED=0.396e-3`.
  - MUL2-16: `MED=323965.82`, sample `NMED=0.075e-3`.
- Project wrapper/PPA output uses the 31-bit masked口径 for 16-bit:
  - `error_elex.csv`: MUL1-16 `MED=1701426.703839`, `NMED=3.9614427e-4`; MUL2-16 `MED=323910.378159`, `NMED=7.5416262e-5`.

**Discrepancies / caveats**

- `SPEC.md` says the exact N-4 gate-level netlists from Fig. 2/3 still need transcription and warns that pure saturation cannot simply replace the 6-4/7-4/8-4 cells. The generator nevertheless uses `satN4` popcount saturation for top-level datapaths.
- The recovered `apx42` Table-II cell exists in `el4_cells.v`, but `mul1_*`/`mul2_*` top modules do not instantiate it; they sum saturated column counts directly.
- Partial products are generated with ANDs, while SPEC describes NAND/inverted partial products as part of the paper’s gate-level optimization.
- MUL1-8 is about 2.3x more accurate than the paper NMED, so this is not a structurally faithful ELEX baseline even though it is a reproducible internal model. This matches `RECON_REPORT.md` caveats, but not the stricter `SPEC.md` implementation target.

## 5. `trunc_dadda_baseline/`

**Verdict: ⚠️ MINOR ISSUE** — truncation plus constant correction is mathematically sound for the measured `k01..k24` set, but generated/measured artifacts are not perfectly aligned.

**Evidence**

- Re-derived `gen_trunc.py`:
  - Initial unsigned partial-product column height is `min(c+1,16,31-c)`.
  - For columns `< k`, expected removed value is `E[Δ] = Σ 0.25 * height[c] * 2^c`.
  - Correction is `C = round(E[Δ])`.
  - `C` is greedily represented as constant `1'b1` bits in truncated columns, within available column height.
- Independent constants:
  - `k=8`: `E=448.25`, `C=448`, bits `{7:3, 6:1}`; RTL shows one `1'b1` in column 6 and three in column 7.
  - `k=16`: `E=245760.25`, `C=245760`, bits `{15:7, 14:1}`.
  - The expected signed truncation bias is `C-E = -0.25` for these cuts; MC CSV bias is close to this, with sampling variation at high error.
- `error_trunc.csv` / `combined_trunc.csv` are internally consistent for `k01..k24`.
  - `k08 MED=200.018611`, `NMED=4.6570462e-8`.
  - `k24 MED=14798167.208210`, `NMED=3.4454668e-3`.
- RTL top signature is the project form `MUL(clk,a,b,out[30:0])`.

**Discrepancies / caveats**

- `gen_trunc.py` emits `k01..k25`, but `error_trunc.csv`, `ppa_trunc.csv`, and `combined_trunc.csv` contain only `k01..k24`.
- `err_logs/raw.txt` stops at `k22`; it is not a complete raw backing log for `error_trunc.csv`.
- This is a single fixed random routing for all `k`; it is a fair pure-truncation baseline, but not a routing-optimized truncation Pareto search.

## 6. `common_eval.py`

**Verdict: ✅ CORRECT** — shared MED and PPA harnesses faithfully invoke the project口径.

**Evidence**

- `measure_med` builds the provided RTL with:
  - top module `MUL`;
  - Power_das harness `/home/lee/Power_das/verilate/mul_err_wrap.cpp`;
  - default vector count `16_000_000`;
  - parse line prefix `masked`.
- `mul_err_wrap.cpp` masks both values with `0x7fffffff`, computes `golden = (uint32_t)a * (uint32_t)b & MASK31`, and wraps the signed difference onto the 31-bit ring using `[-2^30, 2^30)`.
- Power_das trainer `CompressorRouting._measure_error_verilator` uses the same harness path, Verilator build style, top module, and `masked` parsing.
- `measure_ppa` imports `/home/lee/Power_das/run_power_sweep.py` and calls `evaluate_single_routing(idx, rtl_str, 16, target_delay)`, with default `target_delay=1.5`.
- `measure_ppa` switches to a writable local staging cwd under `/tmp/claude-1000/dc_stage`, avoiding the non-writable Power_das build directory while preserving the same remote evaluator.
- `run_power_sweep.evaluate_single_routing` parses:
  - `总功耗` in W and returns `power_mw = power * 1000`;
  - `芯片总面积` as area;
  - `极限工作延迟` as delay. Existing CSVs show the expected negative DC delay sign convention.

**Discrepancies / caveats**

- None found. I did not call `measure_ppa`, per the no-remote constraint.

## 7. `evoapproxlib/`

**Verdict: ⚠️ MINOR ISSUE** — EvoApprox-style RTL is wrapped and evaluated correctly, but the characterized v2022 snapshot is not the checked-in `/home/lee/Baselines/evoapproxlib` tree.

**Evidence**

- `/home/lee/Baselines/evoapproxlib` is EvoApproxLib LITE; its README points to a separate `v2022` branch.
- The checked-in LITE tree contains 31 unique `16x16_unsigned` Verilog design names.
- Power_das current v2022 characterization also contains 31 RTL/wrapper designs, but the intersection with the checked-in LITE names is `0/31`.
- Current v2022 wrappers use the correct project口径:
  - example `MUL_mul16u_9DU.v`: `module MUL(input clk, input [15:0] a, input [15:0] b, output [30:0] out)`, instantiates `mul16u_9DU`, and assigns `out = O_full[30:0]`.
- Current v2022 characterization files:
  - 31 wrappers, 31 RTL files.
  - `error_v2022.csv` includes project MED/bias/WCE and the original library MAE field.
  - `ppa_v2022.csv` includes DC/XA area/power/delay with `success=True`.

**Discrepancies / caveats**

- Provenance issue: the consumed v2022 RTL under `/home/lee/Power_das/outputs/2026-06-24_evo_v2022/rtl` is not present in `/home/lee/Baselines/evoapproxlib`; the exact v2022 source snapshot should be pinned or mirrored under Baselines.
- `combined_16bit_v2022.csv` is slightly stale versus `error_v2022.csv` for 9 of 31 designs. The largest observed MED delta is `0.922497` at MED about `1137` (`<0.1%`), so plots are effectively unaffected, but the join is not byte-consistent.

## Overall Summary

| Item | Verdict |
|---|---|
| `scalable_axmul/drum.py` | ✅ CORRECT |
| `scalable_axmul/mitchell.py` | ✅ CORRECT |
| `Zhang_TCASII23_ProbAdj/` | ⚠️ MINOR ISSUE |
| `ELEX4_N/` | ❌ BUG |
| `trunc_dadda_baseline/` | ⚠️ MINOR ISSUE |
| `common_eval.py` | ✅ CORRECT |
| `evoapproxlib/` | ⚠️ MINOR ISSUE |

## Resolution — ELEX4_N ❌ structural fix (2026-06-29)

The ELEX ❌ (item 4) has been **structurally fixed**. The fake `satN4` popcount saturation was
replaced with the **real gate-level netlists transcribed from the paper's Fig 2/3**, each validated
against the paper's published per-compressor error probabilities used as an exact checksum:

| N-4 cell | recovered netlist (p = real PP, np = ¬p) | P(err) recovered | paper |
|---|---|---|---|
| 5-4 | thermometer min(popcount,4) | 1/1024 | 1/1024 ✓ |
| 6-4 | w4=(p0p1)\|(p2p3)\|(p4p5) | 23/2048 | 23/2048 ✓ |
| 7-4 | w4=p6\|(p0p1)\|(p2p3)\|(p4p5) | 859/16384 | 859/16384 ✓ |
| 8-4 | two 4-2 (OAI211) | 6487/65536 = 1−(243/256)² | 6487/65536 ✓ |

Stage-2 uses the real Table-II approximate 4-2, simplified to `C=(w1&w2)|(w3^w4)`, `S=(w1^w2)|(w3&w4)`.
golden↔RTL kept same-source (GOps/ROps dual-mode in `el4_common.py`); **verilator re-verified
golden==RTL: mul1_8/mul2_8 exhaustive 65536, mul1_16/mul2_16 300k random+corner — all PASS**.

Result (8-bit exhaustive): **MUL1-8 now ER 88.59% / NMED 0.722e-3 / MRED 0.573e-2 — matches paper
Table III exactly** (was 0.315e-3, 2.3× too accurate). MUL2-8 ER 99.83%, bias −382.6 = paper MED
exactly; NMED 8.43e-3 vs paper 5.884e-3 residual is the unrecoverable Fig-6 stage-2 error-balancing
schedule (documented in SPEC §7 / RECON §2, same category as Zhang's unpublished 16-bit schedule).
Project-口径 16-bit re-measured (MUL1-16 MED 4.25e6, MUL2-16 MED 5.11e5) + remote DC re-run; the
comparison overlay was regenerated. **Revised ELEX verdict: ✅ MUL1 structural / ⚠️ MUL2 schedule residual.**

## Follow-Up Actions

1. Regenerate or delete stale Zhang `vec_proposedH_16.txt`; make the vector-generation command part of the local verification flow.
2. For Zhang, update `calibrate.py` so its printed validation rows include the shipped `zhang_gen.config()` settings, not only nearby search candidates.
3. Decide whether Zhang 16-bit should target the paper Table IV NMED or keep the current text-derived unpublished schedule; document the choice in one place.
4. Fix ELEX if it is meant to be a structural paper reproduction: transcribe the actual N-4 compressor truth tables/netlists and instantiate the Table-II 4-2 stage in the top datapath.
5. If ELEX is intentionally only a functional/saturation model, rename and label it as such in SPEC/plots so it is not presented as the ELEX N-4 structural baseline.
6. Align truncation generated/measured artifacts: either emit only `k01..k24`, or measure and add `k25`; refresh `err_logs/raw.txt`.
7. Pin the exact EvoApprox v2022 source snapshot used for the Power_das wrappers under `/home/lee/Baselines`, and regenerate `combined_16bit_v2022.csv` from the current error/PPA CSVs.
