#!/usr/bin/env python3
"""Shared logic for the Zhang TCAS-II'23 reproduction (golden model <-> RTL, same source).

Recovered cells (verified against the paper's probability analysis):

  Proposed 4-gate approximate 4-2 compressor  (4 same-weight bits p1..p4 -> C @w+1, S @w):
      C = p1 | p2
      S = (p3 | p4) & ~(p1 ^ p2)
      value = 2*C + S ;  exact = popcount ;  error nonzero only at 6 of 16 patterns:
        {0011:-1, 0100:+1, 0111:-1, 1000:+1, 1011:-1, 1111:-1}
      With P(pp=1)=1/4: P(+1)=54/256=0.211, P(-1)=16/256=0.0625  (matches paper exactly).

  Esposito 4-2 / 3-2 approximate compressors (identical-weight outputs, saturating):
      value = min(popcount, 2) at the SAME weight (no carry).  "two-or-more inputs -> 11".

Net-effect (value) model used everywhere: a column's compressor replaces its bits by an
integer VALUE; the product is P = const + Σ_k value_k · 2^k and the final RCA / synthesizer
performs the exact carry addition.  A proposed compressor's 2*C+S already spans weights k
(S) and k+1 (C) via the ·2^k weighting, so carries land in the right column automatically.
This is exactly the closed-form Σ err·2^col framing.
"""


# ---------------- cell logic ----------------
def prop42(p1, p2, p3, p4):
    """Proposed 4-gate approximate 4-2 compressor. Returns (C, S); value = 2C+S."""
    C = p1 | p2
    S = (p3 | p4) & (~(p1 ^ p2) & 1)
    return C, S


def prop42_value(bits4):
    """value (0..3) of the proposed compressor on a 4-bit tuple/list."""
    p1, p2, p3, p4 = bits4
    C, S = prop42(p1, p2, p3, p4)
    return 2 * C + S


def espo_value(bits):
    """Esposito approximate compressor (3-2 or 4-2): min(popcount, 2), identical-weight."""
    return min(sum(bits), 2)


# ---------------- self-test against the paper's probability analysis ----------------
def _selftest():
    err = {}
    for v in range(16):
        b = [(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]  # p1..p4 (p1 MSB label)
        e = prop42_value(b) - sum(b)
        if e:
            err[f"{b[0]}{b[1]}{b[2]}{b[3]}"] = e
    # expected nonzero-error patterns
    exp = {"0011": -1, "0100": +1, "0111": -1, "1000": +1, "1011": -1, "1111": -1}
    assert err == exp, f"compressor error map mismatch: {err}"
    # probability rates with P(pp=1)=1/4
    def pr(pat):
        pr_ = 1.0
        for ch in pat:
            pr_ *= (1 / 4) if ch == "1" else (3 / 4)
        return pr_
    p_plus = sum(pr(k) for k, e in exp.items() if e > 0)
    p_minus = sum(pr(k) for k, e in exp.items() if e < 0)
    assert abs(p_plus - 54 / 256) < 1e-12, p_plus
    assert abs(p_minus - 16 / 256) < 1e-12, p_minus
    print("[selftest] proposed compressor error map:", err)
    print(f"[selftest] P(+1)={p_plus:.6f} (=54/256={54/256:.6f})  "
          f"P(-1)={p_minus:.6f} (=16/256={16/256:.6f})  -> matches paper")
    # esposito sanity
    assert [espo_value([1,1,0,0]), espo_value([1,1,1,0]), espo_value([0,1,0,0])] == [2, 2, 1]
    print("[selftest] esposito value(>=2 -> 2): OK")
    print("[selftest] ALL PASS")


if __name__ == "__main__":
    _selftest()
