#!/usr/bin/env python3
"""Sayadi/Timarchi/Sheikh-Akbari TCSI 2023 approximate multiplier reconstruction.

"Two Efficient Approximate Unsigned Multipliers by Developing New Configuration
for Approximate 4:2 Compressors", IEEE TCAS-I 70(4):1649-1659, 2023.
DOI 10.1109/TCSI.2023.3242558.

Building blocks (Tables II/III/IV, cross-checked against Fig.4/6/7 Karnaugh maps):
  AC6G-n   : sum = x1|x2|x3|x4 (6-gate), carry per Table II (16 variants)
  ACFGI-n  : sum = 1 (const),  carry = x_n            (gate-free, 4 variants)
  ACFGII-n : sum = x_i, carry = x_j per Table IV      (gate-free, 12 variants)
All compressors are 4:2 without Cin/Cout: carry has weight 2^(col+1).

Multiplier structure (Sec III-C/D, generalised per the paper's n-bit rules):
  columns 1..2N-1 (col c has weight 2^(c-1));  truncated: cols 1..N/2 (dropped,
  no compensation); middle (ACFGI for mul1 / ACFGII for mul2): cols N/2+1 ..
  floor(2(2N-1)/3); upper (AC6G): .. 2N-3; above that exact HA/FA.
  log2(N)-1 compressor stages, then a 2-row final RCA.

Reduction schedule (derived from Fig.9/10, reproduces them exactly for N=8):
  per stage, per column (ascending), pick the MINIMAL number of boxes b s.t.
      leftovers + b + b_prev_col <= H_next   (H_next = ...,8,4,2)
  boxes are filled top-down 4,4,...,remainder; leftover pps pass through.
  Next-stage input order per column: [leftover pps, own box sums S1..Sb,
  carries C1..Cb' from col-1]  (matches the item order drawn in Fig.9/10).
  On the FINAL stage, middle-zone columns box everything with >=2 inputs
  (gate-free compressors shrink the final adder), upper zone stays minimal
  (HA/FA cost gates).  Under-filled boxes are padded with const-0 at the
  bottom ports (x_{k+1..4}=0), as evidenced by Fig.9 col-10 stage-1.
"""
import numpy as np

# ---------------------------------------------------------------- compressors
# AC6G Table II: n -> (sum grouping (info only), carry lambda)
_AC6G_CARRY = {
    1:  lambda x1, x2, x3, x4: (x1 & (x3 | x4)) | (x2 & x3),
    2:  lambda x1, x2, x3, x4: (x1 & (x3 | x4)) | (x2 & x4),
    3:  lambda x1, x2, x3, x4: (x1 & (x3 | x4)) | (x3 & x4),
    4:  lambda x1, x2, x3, x4: (x2 & (x3 | x4)) | (x1 & x3),
    5:  lambda x1, x2, x3, x4: (x2 & (x3 | x4)) | (x1 & x4),
    6:  lambda x1, x2, x3, x4: (x2 & (x3 | x4)) | (x3 & x4),
    7:  lambda x1, x2, x3, x4: (x3 & (x1 | x2)) | (x1 & x2),
    8:  lambda x1, x2, x3, x4: (x4 & (x1 | x2)) | (x1 & x2),
    9:  lambda x1, x2, x3, x4: (x1 & (x2 | x4)) | (x2 & x3),
    10: lambda x1, x2, x3, x4: (x1 & (x2 | x4)) | (x3 & x4),
    11: lambda x1, x2, x3, x4: (x3 & (x2 | x4)) | (x1 & x2),
    12: lambda x1, x2, x3, x4: (x3 & (x2 | x4)) | (x1 & x4),
    13: lambda x1, x2, x3, x4: (x1 & (x2 | x3)) | (x2 & x4),
    14: lambda x1, x2, x3, x4: (x1 & (x2 | x3)) | (x3 & x4),
    15: lambda x1, x2, x3, x4: (x4 & (x2 | x3)) | (x1 & x2),
    16: lambda x1, x2, x3, x4: (x4 & (x2 | x3)) | (x1 & x3),
}
# ACFGII Table IV: n -> (sum port, carry port)  (1-indexed x ports)
_ACFGII = {1: (1, 2), 2: (1, 3), 3: (1, 4), 4: (2, 1), 5: (2, 3), 6: (2, 4),
           7: (3, 1), 8: (3, 2), 9: (3, 4), 10: (4, 1), 11: (4, 2), 12: (4, 3)}


def comp_eval(family, k, xs):
    """Evaluate compressor on 4 boolean arrays (or python ints 0/1) -> (sum, carry)."""
    x1, x2, x3, x4 = xs
    if family == 'AC6G':
        return (x1 | x2 | x3 | x4), _AC6G_CARRY[k](x1, x2, x3, x4)
    if family == 'ACFGI':
        one = np.ones_like(x1) if isinstance(x1, np.ndarray) else 1
        return one, xs[k - 1]
    if family == 'ACFGII':
        s, c = _ACFGII[k]
        return xs[s - 1], xs[c - 1]
    if family == 'FA':
        return x1 ^ x2 ^ x3, (x1 & x2) | (x3 & (x1 ^ x2))
    if family == 'HA':
        return x1 ^ x2, x1 & x2
    raise ValueError(family)


# Verilog expression templates (ports as verilog operand strings)
def comp_verilog(family, k, p):
    x1, x2, x3, x4 = p
    if family == 'AC6G':
        grp = {1: (1, 3), 2: (1, 3), 3: (1, 3), 4: (2, 3), 5: (2, 3), 6: (2, 3),
               7: (3, 1), 8: (4, 1), 9: (1, 2), 10: (1, 2), 11: (3, 2), 12: (3, 2),
               13: (1, 2), 14: (1, 2), 15: (4, 2), 16: (4, 2)}
        pair2 = {1: (2, 3), 2: (2, 4), 3: (3, 4), 4: (1, 3), 5: (1, 4), 6: (3, 4),
                 7: (1, 2), 8: (1, 2), 9: (2, 3), 10: (3, 4), 11: (1, 2), 12: (1, 4),
                 13: (2, 4), 14: (3, 4), 15: (1, 2), 16: (1, 3)}
        or_arg = {1: (3, 4), 2: (3, 4), 3: (3, 4), 4: (3, 4), 5: (3, 4), 6: (3, 4),
                  7: (1, 2), 8: (1, 2), 9: (2, 4), 10: (2, 4), 11: (2, 4), 12: (2, 4),
                  13: (2, 3), 14: (2, 3), 15: (2, 3), 16: (2, 3)}
        a = grp[k][0]; oa, ob = or_arg[k]; b1, b2 = pair2[k]
        s = f"({x1} | {x2} | {x3} | {x4})"
        c = f"(({p[a-1]} & ({p[oa-1]} | {p[ob-1]})) | ({p[b1-1]} & {p[b2-1]}))"
        return s, c
    if family == 'ACFGI':
        return "1'b1", p[k - 1]
    if family == 'ACFGII':
        s, c = _ACFGII[k]
        return p[s - 1], p[c - 1]
    if family == 'FA':
        return (f"({x1} ^ {x2} ^ {x3})",
                f"(({x1} & {x2}) | ({x3} & ({x1} ^ {x2})))")
    if family == 'HA':
        return f"({x1} ^ {x2})", f"({x1} & {x2})"
    raise ValueError(family)


# ---------------------------------------------------------------- schedule
class Box:
    __slots__ = ('stage', 'col', 'idx', 'inputs', 'family', 'k', 'searchable')

    def __init__(self, stage, col, idx, inputs, family=None, k=None, searchable=False):
        self.stage, self.col, self.idx = stage, col, idx
        self.inputs = inputs          # list of sigs, len<=4, padded conceptually with const0
        self.family, self.k = family, k
        self.searchable = searchable  # True => Algorithm-1 slot

    @property
    def slot(self):
        return (self.stage, self.col)

    def __repr__(self):
        return f"Box(s{self.stage},c{self.col},#{self.idx},{self.family}-{self.k},in={self.inputs})"


def zones(N):
    ncols = 2 * N - 1
    trunc_hi = N // 2                      # cols 1..N/2 truncated
    mid_hi = (2 * ncols) // 3              # middle: trunc_hi+1 .. mid_hi
    upper_hi = 2 * N - 3                   # upper: mid_hi+1 .. 2N-3
    return trunc_hi, mid_hi, upper_hi


def build_schedule(N, mid_family):
    """Build the full reduction tree. Returns (boxes, final_cols, n_stages).
    sig encodings: ('pp', i, j) | ('s', stage, col, idx) | ('c', stage, col, idx) | ('const0',)
    final_cols: {col: [sig, ...]} with len<=2 (rows of the final RCA)."""
    ncols = 2 * N - 1
    trunc_hi, mid_hi, upper_hi = zones(N)
    n_stages = N.bit_length() - 1 - 1      # log2(N) - 1
    H = [2 ** (n_stages - s + 1) for s in range(1, n_stages + 1)]  # N=16: [8,4,2]; N=8: [4,2]

    cols = {c: [] for c in range(1, ncols + 1)}
    for i in range(N):
        for j in range(N):
            c = i + j + 1
            if c > trunc_hi:
                cols[c].append(('pp', i, j))
    # order rows top-down by b index (j) then a index, like Fig.8
    for c in cols:
        cols[c].sort(key=lambda s: (s[2], s[1]))

    boxes = []
    for stage in range(1, n_stages + 1):
        Hn = H[stage - 1]
        final = (stage == n_stages)
        nxt = {c: [] for c in range(1, ncols + 1)}
        b_prev = 0
        for c in range(1, ncols + 1):
            sig = cols[c]
            m = len(sig)
            in_mid = trunc_hi < c <= mid_hi
            in_up = mid_hi < c <= upper_hi
            # minimal number of boxes s.t. leftovers + b + b_prev <= H_next
            bmax = (m + 3) // 4
            b = 0
            while b < bmax and max(m - 4 * b, 0) + b + b_prev > Hn:
                b += 1
            assert max(m - 4 * b, 0) + b + b_prev <= Hn, (stage, c, m, b, b_prev)
            if final and in_mid and m >= 2:
                b = (m + 3) // 4          # gate-free: box everything
            # split boxed bits top-down 4,4,...; rebalance so no box gets <2
            boxed = m if 4 * b >= m else 4 * b
            sizes = []
            for i in range(b):
                rest_boxes = b - i - 1
                take = min(4, boxed - 2 * rest_boxes)
                sizes.append(take)
                boxed -= take
            assert all(sz >= 2 for sz in sizes), (stage, c, m, b, sizes)
            leftovers = sig[sum(sizes):]
            col_boxes = []
            off = 0
            for idx, sz in enumerate(sizes):
                ins = sig[off:off + sz]
                off += sz
                if in_mid:
                    fam, kk, srch = mid_family, None, True
                elif in_up and sz >= 4:
                    fam, kk, srch = 'AC6G', None, True
                else:
                    fam, kk, srch = ('FA' if sz == 3 else 'HA' if sz == 2 else 'HA'), 0, False
                    if sz >= 4:           # top cols shouldn't hit this; guard
                        fam, kk, srch = 'AC6G', None, True
                bx = Box(stage, c, idx, ins, fam, kk, srch)
                boxes.append(bx)
                col_boxes.append(bx)
            # next stage inputs: leftovers, own sums, carries from col-1
            nxt[c].extend(leftovers)
            nxt[c].extend(('s', stage, c, i) for i in range(len(col_boxes)))
            if c > 1:
                nxt[c].extend(('c', stage, c - 1, i) for i in range(b_prev))
            b_prev = len(col_boxes)
        # carries of last column would go to ncols+1 (bit 2N-1): keep a virtual col
        cols = nxt
    return boxes, cols, n_stages


# ---------------------------------------------------------------- simulation
def simulate(N, boxes, final_cols, assignment, a, b):
    """Vectorised golden model. assignment: {(stage,col): (family,k)} for searchable
    slots; unassigned searchable slots behave value-exact (ideal), used only
    during Algorithm-1 greedy. a,b: uint32/uint64 arrays. Returns approx product
    (int64 array, full width, untruncated bits already dropped)."""
    a = np.asarray(a, dtype=np.uint32); b = np.asarray(b, dtype=np.uint32)
    cache = {}
    ideal_corr = np.zeros(a.shape, dtype=np.int64)  # value correction for ideal slots

    def val(sig):
        if sig in cache:
            return cache[sig]
        kind = sig[0]
        if kind == 'pp':
            _, i, j = sig
            v = ((a >> np.uint32(i)) & 1).astype(np.uint8) & ((b >> np.uint32(j)) & 1).astype(np.uint8)
        elif kind == 'const0':
            v = np.zeros(a.shape, dtype=np.uint8)
        else:
            raise KeyError(sig)
        cache[sig] = v
        return v

    zero = np.zeros(a.shape, dtype=np.uint8)
    for bx in boxes:
        xs = [val(s) for s in bx.inputs] + [zero] * (4 - len(bx.inputs))
        fam, k = bx.family, bx.k
        if bx.searchable:
            asg = assignment.get(bx.slot)
            if asg is None:
                # ideal slot: emit (sum=popcount&1-ish) — keep value-exact:
                # output 2 bits that carry as much as possible, correct the rest.
                tot = xs[0].astype(np.int64) + xs[1] + xs[2] + xs[3]
                c = (tot >= 2).astype(np.uint8)
                s = (tot - 2 * c.astype(np.int64)).clip(0, 1).astype(np.uint8)
                ideal_corr += (tot - (2 * c.astype(np.int64) + s)) << (bx.col - 1)
                cache[('s', bx.stage, bx.col, bx.idx)] = s
                cache[('c', bx.stage, bx.col, bx.idx)] = c
                continue
            fam, k = asg
        s, c = comp_eval(fam, k, xs)
        if isinstance(s, int):
            s = np.full(a.shape, s, dtype=np.uint8)
        cache[('s', bx.stage, bx.col, bx.idx)] = s.astype(np.uint8)
        cache[('c', bx.stage, bx.col, bx.idx)] = c.astype(np.uint8)

    out = ideal_corr
    for c, sigs in final_cols.items():
        for sig in sigs:
            out = out + (cache[sig] if sig in cache else val(sig)).astype(np.int64) * (1 << (c - 1))
    return out


def metrics(N, approx, a, b):
    exact = a.astype(np.int64) * b.astype(np.int64)
    ed = np.abs(approx - exact)
    maxp = (2 ** N - 1) ** 2
    er = float(np.mean(ed != 0)) * 100
    nmed = float(np.mean(ed)) / maxp
    nz = exact != 0
    mred = float(np.mean(ed[nz] / exact[nz]))
    return {'ER%': er, 'NMED': nmed, 'MRED': mred, 'MaxED': int(ed.max()),
            'MED': float(np.mean(ed))}
