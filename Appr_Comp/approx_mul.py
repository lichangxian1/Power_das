#!/usr/bin/env python3
"""阶段3 Phase A：手工低位替换近似乘法器 + EDA-free 位精确验证。

- 用 cell_policy 把低位列的 3:2/2:2 换成近似 cell，emit 可综合 RTL（追加近似 module 定义）。
- 一个 Python 位精确模拟器，逐位复刻 compressor_tree.emit_verilog_fused_assignment 的布线，
  在「值域」验证核心恒等式：  out = a*b + Σ_compressors e_local·2^col   （精确成立）
  并穷举/采样算实测误差 ER/MED/NMED/WCE/mean-signed，对比解析估计 Σ bias·2^col。

用法： python Appr_Comp/approx_mul.py --bit 8 --thresh 4 --fa comp32_apx_neg_2 --ha comp22_apx_neg_1
"""
import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from utils.mul import Mul
from utils.compressor_tree import CompressorTree, get_initial_partial_product
from gen_verilog import emit_module

LIB = json.load(open(os.path.join(HERE, "library.json")))["cells"]
NAME_BY = {n: c for n, c in LIB.items()}
EXACT32 = next(n for n, c in LIB.items() if c.get("is_exact") and c["type"] == "32")
EXACT22 = next(n for n, c in LIB.items() if c.get("is_exact") and c["type"] == "22")
SEL = json.load(open(os.path.join(HERE, "selected_compressors.json")))["selected"]


def luts(cell_name, t):
    """返回 (sum_lut, carry_lut, bias)；cell_name=None -> 精确 FA/HA。"""
    name = cell_name or (EXACT32 if t == 0 else EXACT22)
    c = NAME_BY[name]
    return c["sum_lut"], c["carry_lut"], c["weighted_signed_error"]


def pp_values(a, b, bit_width, initial_pp):
    """AND 编码：复刻 emit_pp_encoder 的 pp_col[pp_index] = a[i]&b[j]。"""
    abit = [(a >> i) & 1 for i in range(bit_width)]
    bbit = [(b >> i) & 1 for i in range(bit_width)]
    cols = []
    for col in range(len(initial_pp)):
        off = max(0, col - bit_width + 1)
        h = int(initial_pp[col])
        cols.append([abit[i + off] & bbit[col - i - off] for i in range(h)])
    return cols


def simulate(a, b, bit_width, initial_pp, assignment, policy):
    """位精确模拟 fused 压缩树，返回 (out_full, e_local_sum)。"""
    from collections import deque
    column_num = len(initial_pp)
    inwl = [deque(v) for v in pp_values(a, b, bit_width, initial_pp)]
    e_sum = 0
    for stage in assignment:
        outwl = [deque() for _ in range(column_num)]
        for col in range(column_num):
            for (s, c, t, idx) in stage[col]:
                cell = policy(s, c, t, idx)
                if t == 0:
                    ins = [inwl[col].pop() for _ in range(3)]  # a,b,cin
                    sl, cl, _ = luts(cell, 0)
                    pat = ins[0] * 4 + ins[1] * 2 + ins[2]
                    exact_val = ins[0] + ins[1] + ins[2]
                else:
                    ins = [inwl[col].pop() for _ in range(2)]  # a,cin
                    sl, cl, _ = luts(cell, 1)
                    pat = ins[0] * 2 + ins[1]
                    exact_val = ins[0] + ins[1]
                sv, cv = sl[pat], cl[pat]
                e_sum += ((sv + 2 * cv) - exact_val) * (1 << col)
                outwl[col].appendleft(sv)
                if col + 1 < column_num:
                    outwl[col + 1].appendleft(cv)
            while outwl[col]:
                inwl[col].appendleft(outwl[col].pop())
    a_val = b_val = 0
    for col in range(column_num):
        w = list(inwl[col])
        a_val += (w[0] if len(w) >= 1 else 0) << col
        b_val += (w[1] if len(w) >= 2 else 0) << col
    return a_val + b_val, e_sum


def make_policy(thresh, fa_cell, ha_cell):
    def resolve(x):
        if not x or str(x).lower() == "none":
            return None                       # 不近似
        return SEL[x]["name"] if x in SEL else x  # alias -> module名，否则当作 module名
    fa_name = resolve(fa_cell)
    ha_name = resolve(ha_cell)

    def policy(s, c, t, idx):
        if c >= thresh:
            return None
        if t == 0 and fa_name:
            return fa_name
        if t == 1 and ha_name:
            return ha_name
        return None
    return policy, fa_name, ha_name


def used_approx_cells(assignment, policy):
    used = set()
    for stage in assignment:
        for col in stage:
            for (s, c, t, idx) in col:
                cell = policy(s, c, t, idx)
                if cell:
                    used.add(cell)
    return used


def approx_modules_src(used):
    src = "\n// ===== approximate compressor cells =====\n"
    pat3 = ["".join(p) for p in itertools.product("01", repeat=3)]
    pat2 = ["".join(p) for p in itertools.product("01", repeat=2)]
    for name in sorted(used):
        c = NAME_BY[name]
        if c["type"] == "32":
            src += emit_module(name, ["a", "b", "cin"], pat3, c["sum_lut"], c["carry_lut"],
                               f"{name} bias={c['weighted_signed_error']:+.3f}")
        else:
            src += emit_module(name, ["a", "cin"], pat2, c["sum_lut"], c["carry_lut"],
                               f"{name} bias={c['weighted_signed_error']:+.3f}")
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bit", type=int, default=8)
    ap.add_argument("--ct", default="dadda", choices=["dadda", "wallace"])
    ap.add_argument("--thresh", type=int, default=4, help="列 < thresh 用近似 cell")
    ap.add_argument("--fa", default="comp32_apx_neg_2", help="低位 3:2 用的近似 alias (或 module名/None)")
    ap.add_argument("--ha", default="comp22_apx_neg_1", help="低位 2:2 用的近似 alias")
    ap.add_argument("--samples", type=int, default=0, help=">0 用随机采样，否则 8-bit 穷举")
    args = ap.parse_args()

    mul = Mul(args.bit, "and", args.ct)
    initial_pp = mul.initial_pp
    assignment = mul.ct.compressor_assignment_fused()
    policy, fa_name, ha_name = make_policy(args.thresh, args.fa, args.ha)
    used = used_approx_cells(assignment, policy)
    print(f"[approxmul] bit={args.bit} ct={args.ct} thresh={args.thresh}")
    print(f"[approxmul] 低位 3:2->{fa_name}  2:2->{ha_name}  used={sorted(used)}")

    # emit RTL
    rtl = os.path.join(HERE, "build", f"MUL_approx_{args.bit}.v")
    mul.emit_verilog(rtl_path=rtl, assignment=assignment, cell_policy=policy,
                     extra_modules_src=approx_modules_src(used))
    print(f"[approxmul] RTL -> {rtl}")

    # exact policy baseline (sanity: out_full == a*b)
    exact_policy = lambda s, c, t, idx: None

    # 输入集合
    if args.samples > 0:
        import random
        rng = random.Random(0)
        pairs = [(rng.randrange(1 << args.bit), rng.randrange(1 << args.bit))
                 for _ in range(args.samples)]
    else:
        pairs = list(itertools.product(range(1 << args.bit), range(1 << args.bit)))

    n = len(pairs)
    id_ok = exact_ok = 0
    err_sum = abs_sum = wce = 0
    nz = 0
    maxprod = ((1 << args.bit) - 1) ** 2
    for a, b in pairs:
        out_e, _ = simulate(a, b, args.bit, initial_pp, assignment, exact_policy)
        exact_ok += (out_e == a * b)
        out, e_local = simulate(a, b, args.bit, initial_pp, assignment, policy)
        id_ok += (out - a * b == e_local)        # 恒等式逐输入验证
        err = out - a * b
        err_sum += err
        abs_sum += abs(err)
        wce = max(wce, abs(err))
        nz += (err != 0)

    print("\n=== 验证 (n=%d 输入对) ===" % n)
    print(f"  exact 树 out==a*b           : {exact_ok}/{n}  {'OK' if exact_ok==n else 'FAIL'}")
    print(f"  恒等式 out-a*b==Σe·2^col     : {id_ok}/{n}  {'OK' if id_ok==n else 'FAIL'}")
    print("\n=== 近似乘法器误差 (值域, vs 真实 a*b) ===")
    print(f"  ER  (error rate)            : {nz/n:.4f}")
    print(f"  MED (mean |error|)          : {abs_sum/n:.3f}")
    print(f"  NMED (MED / maxprod)        : {abs_sum/n/maxprod:.3e}")
    print(f"  WCE (max |error|)           : {wce}")
    print(f"  mean signed error (bias)    : {err_sum/n:+.3f}")

    # 解析估计：Σ_compressor bias·2^col （P(1)=1/4 假设，深层会偏）
    analytic = 0.0
    for stage in assignment:
        for col in stage:
            for (s, c, t, idx) in col:
                cell = policy(s, c, t, idx)
                if cell:
                    analytic += NAME_BY[cell]["weighted_signed_error"] * (1 << c)
    print(f"\n  解析 Σ bias·2^col (一阶估计)  : {analytic:+.3f}  "
          f"(vs 实测 mean {err_sum/n:+.3f}；深层列输入非 1/4 故有偏差)")


if __name__ == "__main__":
    main()
