#!/usr/bin/env python3
"""Zhang TCAS-II'23 dual-mode reduction core.

One reduction body, two backends:
  - GoldenOps: bits are ints {0,1}; ops compute -> exact golden value.
  - RtlOps:    bits are Verilog wire names; ops emit gate assigns -> structural netlist.
Because both backends run the identical reduction, golden == RTL by construction, and the
emitted netlist is the actual gate-level structure (proposed/Esposito/FA/HA cells) so DC
synthesises the paper's design rather than re-inventing the compressor tree.

Cells:
  proposed 4-2 : C=p1|p2 ; S=(p3|p4)&~(p1^p2)            (4 gates, paper's contribution)
  Esposito 4-2 : w1=OR4 ; w2=(>=2 of 4)   value=min(cnt,2), same weight
  Esposito 3-2 : w1=OR3 ; w2=(>=2 of 3)   value=min(cnt,2), same weight
  exact FA/HA  : standard
The Esposito (w1,w2) split here is value-equivalent to the paper's gates (sum=min(cnt,2));
only wire labelling differs, which does not affect the multiplier output.
"""


class GoldenOps:
    def AND(self, a, b): return a & b
    def OR(self, a, b): return a | b
    def XOR(self, a, b): return a ^ b
    def NOT(self, a): return a ^ 1


class RtlOps:
    def __init__(self):
        self.lines = []
        self.k = 0

    def _w(self, expr):
        n = f"n{self.k}"; self.k += 1
        self.lines.append(f"    wire {n} = {expr};")
        return n

    def AND(self, a, b): return self._w(f"{a} & {b}")
    def OR(self, a, b): return self._w(f"{a} | {b}")
    def XOR(self, a, b): return self._w(f"{a} ^ {b}")
    def NOT(self, a): return self._w(f"~{a}")


# ---------------- cells ----------------
def prop42(ops, p1, p2, p3, p4):
    C = ops.OR(p1, p2)
    S = ops.AND(ops.OR(p3, p4), ops.NOT(ops.XOR(p1, p2)))
    return C, S


def espo(ops, xs):
    """Esposito approximate compressor, exact gate-level netlist from Zhang Table I
    (two same-weight outputs, value = min(popcount, 2); no XOR — these are the cheap cells).
      4-2:  w1=(p3|p4)|(p1&p2)   w2=(p1|p2)|(p3&p4)     (6 AND/OR gates)
      3-2:  w1=a|b               w2=c|(a&b)             (3 gates)
      2  :  w1=a|b               w2=a&b                 (value-preserving, =HA)
    Verified: reproduces the paper's P(w=0) = 135/256 (4-2) and 36/64, 45/64 (3-2)."""
    n = len(xs)
    if n == 4:
        p1, p2, p3, p4 = xs
        w1 = ops.OR(ops.OR(p3, p4), ops.AND(p1, p2))
        w2 = ops.OR(ops.OR(p1, p2), ops.AND(p3, p4))
        return [w1, w2]
    if n == 3:
        a, b, c = xs
        return [ops.OR(a, b), ops.OR(c, ops.AND(a, b))]
    if n == 2:
        a, b = xs
        return [ops.OR(a, b), ops.AND(a, b)]
    return list(xs)


def FA(ops, a, b, c):
    s = ops.XOR(ops.XOR(a, b), c)
    cy = ops.OR(ops.OR(ops.AND(a, b), ops.AND(b, c)), ops.AND(a, c))
    return s, cy


def HA(ops, a, b):
    return ops.XOR(a, b), ops.AND(a, b)


# ---------------- per-column passes ----------------
def exact_pass(ops, bits):
    stay, carry, i, n = [], [], 0, len(bits)
    while i + 3 <= n:
        s, c = FA(ops, bits[i], bits[i + 1], bits[i + 2]); stay.append(s); carry.append(c); i += 3
    rem = bits[i:]
    if len(rem) == 2:
        s, c = HA(ops, rem[0], rem[1]); stay.append(s); carry.append(c)
    elif len(rem) == 1:
        stay.append(rem[0])
    return stay, carry


def proposed_pass(ops, bits, and_on, max_comp=10 ** 9):
    stay, carry, i, n, used = [], [], 0, len(bits), 0
    while i + 4 <= n and used < max_comp:
        p1, p2, p3, p4 = bits[i], bits[i + 1], bits[i + 2], bits[i + 3]
        C, S = prop42(ops, p1, p2, p3, p4)
        stay.append(S); carry.append(C)
        if and_on:
            stay.append(ops.AND(p3, p4))      # error-correction AND gate (always a wire/bit)
        used += 1
        i += 4
    s, c = exact_pass(ops, bits[i:])
    return stay + s, carry + c


def esposito_pass(ops, bits):
    stay, i, n = [], 0, len(bits)
    while i + 4 <= n:
        stay += espo(ops, bits[i:i + 4]); i += 4
    rem = bits[i:]
    if len(rem) == 3:
        stay += espo(ops, rem); rem = []
    s, c = exact_pass(ops, rem)
    return stay + s, c


# ---------------- full reduction ----------------
def reduce_columns(ops, cols, cfg):
    """cols: list[ list[bit] ] indexed by weight. Returns final cols (each height<=2).
    bit = int (golden) or wire-name str (rtl)."""
    W = len(cols)
    cols = [list(c) for c in cols]
    stage = 0
    while max(len(c) for c in cols) > 2:
        new = [[] for _ in range(W)]
        for k in range(W):
            bits = cols[k]
            if k in cfg["inexact"]:
                if cfg["hybrid"] and k in cfg["espo_cols"]:
                    if stage == 0:
                        stay, carry = esposito_pass(ops, bits)                    # level 1: Esposito
                    elif stage == 1:
                        stay, carry = proposed_pass(ops, bits, k in cfg["and_cols"], max_comp=1)  # level 2
                    elif stage == 2 and cfg.get("post_l2", "exact") == "esposito":
                        stay, carry = esposito_pass(ops, bits)                    # level 3: Esposito (16-bit)
                    elif stage == 2 and cfg.get("post_l2", "exact") == "proposed":
                        stay, carry = proposed_pass(ops, bits, False)
                    else:
                        stay, carry = exact_pass(ops, bits)                       # finish exact (FA/HA)
                else:
                    stay, carry = proposed_pass(ops, bits, cfg.get("and_all", False))
            else:
                stay, carry = exact_pass(ops, bits)
            new[k].extend(stay)
            if k + 1 < W:
                new[k + 1].extend(carry)
        cols = new
        stage += 1
        if stage > 14:
            break
    return cols
