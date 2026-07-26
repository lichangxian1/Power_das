#!/usr/bin/env python3
"""ELEX2024 N-4 近似乘法器复刻 —— 共享 cell 逻辑（golden 模型与 RTL 生成器同源, 防漂移）。

> Zhang et al., IEICE Electronics Express 21(14), 2024. DOI 10.1587/elex.21.20240189

结构性复刻（2026-06-29 重写, 取代旧的纯 min(cnt,4) 饱和近似）:
- 用论文 Fig 2/3 的**真实门级网表**实现 5-4/6-4/7-4/8-4 近似压缩器, 每个都已用论文给出的
  误差概率逐一对齐: 5-4=1/1024, 6-4=23/2048, 7-4=859/16384, 8-4=6487/65536(=1-(243/256)^2)。
- 用论文 Table II 的近似 4-2 压缩器(stage 2), 化简为紧凑布尔式并与真值表逐项核对:
      C = (w1 & w2) | (w3 ^ w4)        # 高权重输出
      S = (w1 ^ w2) | (w3 & w4)        # 同权重输出,  value = 2C + S
- 数据通路(列分区)与论文一致:
  · MUL1(高精度): 列0-3截断(常数'1000'); 列4-10近似(stage1 用 N-4 压缩器, **stage2 精确**);
    最高列精确; 10-bit RCA。 => 每近似列贡献 = N-4 四位输出之和 × 2^k。
  · MUL2(超低功耗): 低 8 列截断(常数"00111000"); 列8用7-4、列9用6-4(stage1), **stage2 用近似4-2**;
    高位精确; 7-bit RCA。 => 列8/9贡献 = (2C+S)(近似4-2作用在 N-4 四位输出) × 2^k。
- 8-bit 验收: MUL1-8 NMED=0.722e-3(论文0.722e-3, **精确吻合**), MUL2-8 NMED≈8.3e-3
  (论文5.884e-3; bias=-382.6 精确等于论文 MED, 结构忠实, 残差源于 Fig 6 低清点阵图里不可辨认的
   stage-2 误差平衡/进位走线 —— 同 Zhang 16-bit 未公开调度性质, 见 RECON_REPORT)。
- 16-bit(论文未定义, 项目口径用): 沿用论文 N-4 cell, 列高>8 时用 8-4 递归组合(我方有据扩展)。

golden(GOps, 整数) 与 RTL(ROps, 线名字符串)走**同一份** n4_bits/apx42, 故 golden==RTL 由构造保证。
"""

# ----------------------------------------------------------------------------
# dual-mode 原语：golden 用整数 0/1，RTL 累积 wire 赋值并返回线名
# ----------------------------------------------------------------------------
class GOps:
    """golden: 直接在 0/1 整数上算布尔。"""
    ZERO = 0
    @staticmethod
    def AND(a, b): return a & b
    @staticmethod
    def OR(a, b): return a | b
    @staticmethod
    def XOR(a, b): return a ^ b


class ROps:
    """RTL: 每个布尔运算 emit 一条 `wire tN = ...;`, 返回线名(字符串)。"""
    ZERO = "1'b0"

    def __init__(self):
        self.lines = []
        self._n = 0

    def _w(self):
        self._n += 1
        return f"t{self._n}"

    def _bin(self, a, b, op):
        w = self._w()
        self.lines.append(f"    wire {w} = {a} {op} {b};")
        return w

    def AND(self, a, b): return self._bin(a, b, "&")
    def OR(self, a, b):  return self._bin(a, b, "|")
    def XOR(self, a, b): return self._bin(a, b, "^")


# ----------------------------------------------------------------------------
# 近似 N-4 压缩器（真实网表, Fig 2/3）—— 输入同权重 bits, 输出 4 个同权重位 [w1,w2,w3,w4]
# 误差概率均已对齐论文（见模块 docstring）。golden/RTL 共用。
# ----------------------------------------------------------------------------
def _or_reduce(ops, items):
    items = list(items)
    acc = items[0]
    for x in items[1:]:
        acc = ops.OR(acc, x)
    return acc


def _and_reduce(ops, items):
    items = list(items)
    acc = items[0]
    for x in items[1:]:
        acc = ops.AND(acc, x)
    return acc


def _atleast(ops, bits, j):
    """对称布尔: popcount(bits) >= j, 即所有 j 元子集与 之或。"""
    import itertools
    return _or_reduce(ops, [_and_reduce(ops, c) for c in itertools.combinations(bits, j)])


def _c54(ops, b):
    """5-4: 4 个同权重输出位, 和 = min(popcount,4); 仅 5 个全 1 时 -1 (P=(1/4)^5=1/1024)。
    饱和温度计码 w_j = (popcount >= j), j=1..4。golden/RTL 共用故等价。"""
    return [_atleast(ops, b, 1), _atleast(ops, b, 2),
            _atleast(ops, b, 3), _atleast(ops, b, 4)]


def _c64(ops, b):
    p0, p1, p2, p3, p4, p5 = b
    w1 = ops.OR(p0, p1)
    w2 = ops.OR(p2, p3)
    w3 = ops.OR(p4, p5)
    w4 = ops.OR(ops.OR(ops.AND(p0, p1), ops.AND(p2, p3)), ops.AND(p4, p5))
    return [w1, w2, w3, w4]


def _c74(ops, b):
    p0, p1, p2, p3, p4, p5, p6 = b
    w1 = ops.OR(p0, p1)
    w2 = ops.OR(p2, p3)
    w3 = ops.OR(p4, p5)
    carry = ops.OR(ops.OR(ops.AND(p0, p1), ops.AND(p2, p3)), ops.AND(p4, p5))
    w4 = ops.OR(p6, carry)
    return [w1, w2, w3, w4]


def _c84(ops, b):
    """8-4 = 两个独立的近似 4-2(同权重): {p0..p3}->w1,w2 与 {p4..p7}->w3,w4。"""
    p0, p1, p2, p3, p4, p5, p6, p7 = b
    w1 = ops.OR(ops.OR(ops.AND(p0, p1), p2), p3)
    w2 = ops.OR(ops.OR(ops.AND(p2, p3), p0), p1)
    w3 = ops.OR(ops.OR(ops.AND(p6, p7), p4), p5)
    w4 = ops.OR(ops.OR(ops.AND(p4, p5), p6), p7)
    return [w1, w2, w3, w4]


def n4_bits(ops, bits):
    """任意列高 -> 4 个同权重输出位。n<=4 直通; 5..8 用真实 cell; >8 用 8-4 递归组合。"""
    bits = list(bits)
    n = len(bits)
    if n == 0:
        return [ops.ZERO] * 4
    if n <= 4:
        return bits + [ops.ZERO] * (4 - n)
    if n == 5:
        return _c54(ops, bits)
    if n == 6:
        return _c64(ops, bits)
    if n == 7:
        return _c74(ops, bits)
    if n == 8:
        return _c84(ops, bits)
    # n > 8: 先把前 8 位用 8-4 压成 4 位, 与剩余位合并后递归（论文未定义 16-bit, 我方有据扩展）
    head = n4_bits(ops, bits[:8])
    return n4_bits(ops, head + bits[8:])


def n4_value(ops, bits):
    """近似 N-4 压缩器输出的数值(= 4 位之和)。MUL1 stage2 精确, 故列贡献 = 此值。
    一律 = sum(n4_bits)，golden 与 RTL 走同一份 cell 逻辑，无特判。"""
    return sum(n4_bits(ops, bits))


# ----------------------------------------------------------------------------
# 近似 4-2 压缩器（Table II）—— stage 2; 紧凑式与真值表逐项核对
#   C = (w1 & w2) | (w3 ^ w4)   (高权重)
#   S = (w1 ^ w2) | (w3 & w4)   (同权重)
# ----------------------------------------------------------------------------
def apx42(ops, w):
    w1, w2, w3, w4 = w
    C = ops.OR(ops.AND(w1, w2), ops.XOR(w3, w4))
    S = ops.OR(ops.XOR(w1, w2), ops.AND(w3, w4))
    return C, S


def apx42_value(ops, w):
    C, S = apx42(ops, w)
    return 2 * C + S


# ----------------------------------------------------------------------------
# 列结构 / 分区 / 截断常数（沿用已验证的设定）
# ----------------------------------------------------------------------------
def height(N, k):
    return min(k + 1, N, 2 * N - 1 - k)


def col_pps(N, k):
    """列 k 的部分积 (i,j) 列表(i+j=k), 按行 i 升序 = 规范配对顺序。"""
    return [(i, k - i) for i in range(N) if 0 <= k - i < N]


def col_counts(a, b, N):
    cnt = [0] * (2 * N - 1)
    for i in range(N):
        if not (b >> i) & 1:
            continue
        for j in range(N):
            if (a >> j) & 1:
                cnt[i + j] += 1
    return cnt


def col_bits_int(a, b, N, k):
    """列 k 的部分积比特值列表(规范顺序), 供 golden 调用。"""
    return [((a >> j) & 1) & ((b >> i) & 1) for (i, j) in col_pps(N, k)]


def regions(N, design):
    """返回 (trunc_cols:set, sat_cols:set)。常数另算。"""
    W = 2 * N - 1
    if design == "mul1":
        trunc = set(range(0, 4))
        sat = {k for k in range(W) if k not in trunc and height(N, k) > 4}
    elif design == "mul2":
        trunc = set(range(0, N))
        sat = {N, N + 1}
    else:
        raise ValueError(design)
    return trunc, sat


# 论文给定的 8-bit 截断常数（直接采用）
PAPER_CONST_8 = {"mul1": 8, "mul2": 0x38}   # '1000' / "00111000"
# 16-bit 截断常数: 联合(NMED+MRED)最优, 冻结(确定性, 保证 golden==RTL)。mul1-16 取 8(同论文 MUL1)。
FROZEN_CONST_16 = {"mul1": 8, "mul2": 12552}  # 0x8, 0x3108


def best_trunc_const(N, design, sample=None, seed=0):
    """遍历候选常数取 (NMED+MRED) 联合最小者(复刻论文按 NMED/MRED 选最优)。"""
    trunc, sat = regions(N, design)
    nbits = max(trunc) + 1 if trunc else 0
    span = 1 << nbits
    import random
    rng = random.Random(seed)
    if N <= 8:
        pairs = [(a, b) for a in range(1 << N) for b in range(1 << N)]
    else:
        pairs = [(rng.randrange(1 << N), rng.randrange(1 << N)) for _ in range(sample or 200000)]
    base = []
    for (a, b) in pairs:
        hi = _hi_value(a, b, N, design, trunc, sat)
        base.append((a * b, hi))
    Dmax = ((1 << N) - 1) ** 2

    def metrics(c):
        ae = 0; rs = 0.0; rn = 0
        for (ex, hi) in base:
            e = abs(hi + c - ex); ae += e
            if ex:
                rs += e / ex; rn += 1
        return ae / len(base) / Dmax, (rs / rn if rn else 0.0)

    def obj(c):
        nmed, mred = metrics(c); return nmed + mred

    step = max(1, span // 256)
    bestc = min(range(0, span, step), key=obj)
    lo, hi_ = max(0, bestc - step), min(span - 1, bestc + step)
    bestc = min(range(lo, hi_ + 1), key=obj)
    nmed, mred = metrics(bestc)
    return bestc, nmed, mred


def trunc_const(N, design):
    if N == 8:
        return PAPER_CONST_8[design]
    if N == 16:
        return FROZEN_CONST_16[design]
    c, _, _ = best_trunc_const(N, design)
    return c


# ----------------------------------------------------------------------------
# golden 乘法器（整数）—— 与 RTL 同源(共用 n4_bits/apx42)
# ----------------------------------------------------------------------------
def _hi_value(a, b, N, design, trunc, sat):
    """常数以外的高位贡献(用于截断常数搜索, 不含 trunc_const)。"""
    P = 0
    for k in range(2 * N - 1):
        if k in trunc:
            continue
        bits = col_bits_int(a, b, N, k)
        if k in sat:
            if design == "mul1":
                P += n4_value(GOps, bits) << k
            else:
                P += apx42_value(GOps, n4_bits(GOps, bits)) << k
        else:
            P += sum(bits) << k
    return P


def approx_mul(a, b, N, design, tconst):
    """golden 近似乘积。结构与 RTL 完全一致。"""
    trunc, sat = regions(N, design)
    return tconst + _hi_value(a, b, N, design, trunc, sat)
