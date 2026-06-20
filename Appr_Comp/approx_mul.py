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


def expected_delta(k, bit_width, initial_pp):
    """截断列 [0,k) 的期望丢失值 E[Δ] = Σ_{c<k} P(1)·n_c·2^c（AND: P(1)=1/4, n_c=列高）。
    截断恒为正损失（丢掉的都是正权值），故引入确定性负偏置 −E[Δ]。"""
    e = 0.0
    for c in range(min(k, len(initial_pp))):
        e += 0.25 * int(initial_pp[c]) * (1 << c)
    return e


def correction_constant(k, bit_width, initial_pp, mode):
    """校正常数 C：'none'=0；'bias'=round(E[Δ])（解析零偏置）。"""
    if mode == "none" or k <= 0:
        return 0
    return int(round(expected_delta(k, bit_width, initial_pp)))


def delta_actual(a, b, bit_width, k, initial_pp_full):
    """单输入真实丢失值 Δ(a,b) = Σ_{c<k} 2^c·(该列 PP 位和)，用满 pp 复刻。"""
    cols = pp_values(a, b, bit_width, initial_pp_full)
    d = 0
    for c in range(min(k, len(initial_pp_full))):
        d += sum(cols[c]) << c
    return d


def run_metrics(pairs, bit_width, pp_sim, assignment, policy, C, k, initial_pp_full):
    """跑一组输入，返回 (metrics, id_ok)。out_corrected = simulate(截断树)+C。
    恒等式：out − a*b == −Δ(a,b) + Σ_cells e·2^col（+0；C 是事后常数，单列出）。"""
    n = len(pairs)
    err_sum = abs_sum = wce = nz = id_ok = 0
    maxprod = ((1 << bit_width) - 1) ** 2
    for a, b in pairs:
        out, e_local = simulate(a, b, bit_width, pp_sim, assignment, policy)
        d = delta_actual(a, b, bit_width, k, initial_pp_full) if k > 0 else 0
        id_ok += ((out - a * b) == (-d + e_local))
        err = (out + C) - a * b
        err_sum += err
        abs_sum += abs(err)
        wce = max(wce, abs(err))
        nz += (err != 0)
    return {
        "ER": nz / n, "MED": abs_sum / n, "NMED": abs_sum / n / maxprod,
        "WCE": wce, "bias": err_sum / n,
    }, id_ok


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
    ap.add_argument("--trunc", type=int, default=0, help="截断最低 k 列（不生成 PP/压缩器）")
    ap.add_argument("--correct", default="bias", choices=["none", "bias"],
                    help="截断校正常数 C：none=0；bias=round(E[Δ])（解析零偏置）")
    args = ap.parse_args()

    mul_full = Mul(args.bit, "and", args.ct)
    initial_pp_full = mul_full.initial_pp
    k = args.trunc

    # 截断：把最低 k 列高度置 0，用截断后的 pp 重建压缩树（低列自然无压缩器）。
    if k > 0:
        import numpy as np
        pp_sim = np.array(initial_pp_full, dtype=int).copy()
        pp_sim[:k] = 0
        ct = CompressorTree.dadda(pp_sim) if args.ct == "dadda" else CompressorTree.wallace(pp_sim)
        assignment = ct.compressor_assignment_fused()
    else:
        pp_sim = initial_pp_full
        assignment = mul_full.ct.compressor_assignment_fused()
    initial_pp = pp_sim
    policy, fa_name, ha_name = make_policy(args.thresh, args.fa, args.ha)
    used = used_approx_cells(assignment, policy)
    print(f"[approxmul] bit={args.bit} ct={args.ct} thresh={args.thresh} trunc={k} correct={args.correct}")
    print(f"[approxmul] 低位 3:2->{fa_name}  2:2->{ha_name}  used={sorted(used)}")

    # emit RTL（仅 k=0；截断 RTL 发射属 Layer 2，搜索 loop 接入时再做）
    if k == 0:
        rtl = os.path.join(HERE, "build", f"MUL_approx_{args.bit}.v")
        mul_full.emit_verilog(rtl_path=rtl, assignment=assignment, cell_policy=policy,
                              extra_modules_src=approx_modules_src(used))
        print(f"[approxmul] RTL -> {rtl}")

    # 输入集合
    if args.samples > 0:
        import random
        rng = random.Random(0)
        pairs = [(rng.randrange(1 << args.bit), rng.randrange(1 << args.bit))
                 for _ in range(args.samples)]
    else:
        pairs = list(itertools.product(range(1 << args.bit), range(1 << args.bit)))
    n = len(pairs)
    none_policy = lambda s, c, t, idx: None

    # exact-tree sanity（k=0 整树应 out==a*b）
    exact_ok = sum(simulate(a, b, args.bit, initial_pp, assignment, none_policy)[0] == a * b
                   for a, b in pairs) if k == 0 else None

    C_bias = correction_constant(k, args.bit, initial_pp_full, "bias")

    if k > 0:
        # 三场景对比：截断负偏置 → 常数校正归零 → 叠正负 cell 抵消
        EΔ = expected_delta(k, args.bit, initial_pp_full)
        print(f"\n=== 截断 [0,{k}) ：E[Δ]={EΔ:.2f}  →  校正常数 C=round(E[Δ])={C_bias} ===")
        scen = [
            ("截断 only (C=0)",                none_policy, 0),
            (f"截断+常数校正 (C={C_bias})",      none_policy, C_bias),
            ("截断+cell抵消 (C=0)",            policy,      0),
            (f"截断+常数+cell (C={C_bias})",    policy,      C_bias),
        ]
        print(f"\n  {'场景':<24}{'恒等式':>10}{'ER':>8}{'MED':>10}{'WCE':>8}{'bias':>10}")
        for name, pol, C in scen:
            m, id_ok = run_metrics(pairs, args.bit, initial_pp, assignment, pol, C, k, initial_pp_full)
            tag = "OK" if id_ok == n else f"FAIL({id_ok}/{n})"
            print(f"  {name:<24}{tag:>10}{m['ER']:>8.3f}{m['MED']:>10.2f}"
                  f"{int(m['WCE']):>8d}{m['bias']:>+10.2f}")
        print("\n  解读：截断引入 −E[Δ] 负偏置 → 两条抵消杠杆：常数 C（归零）或正偏置 cell（部分抵消 + 省面积）。")
        print("  RL 协同优化 (k, C, cell 符号)：bias 项压净偏置→0，WCE 项(④)压尾巴，PPA 项吃面积/功耗。")
    else:
        m, id_ok = run_metrics(pairs, args.bit, initial_pp, assignment, policy, 0, 0, initial_pp_full)
        print("\n=== 验证 (n=%d) ===" % n)
        print(f"  exact 树 out==a*b           : {exact_ok}/{n}  {'OK' if exact_ok==n else 'FAIL'}")
        print(f"  恒等式 out-a*b==Σe·2^col     : {id_ok}/{n}  {'OK' if id_ok==n else 'FAIL'}")
        print(f"  ER={m['ER']:.4f}  MED={m['MED']:.3f}  NMED={m['NMED']:.3e}  "
              f"WCE={int(m['WCE'])}  bias={m['bias']:+.3f}")

    # 解析估计：−Δ + C + Σ_cell bias·2^col（一阶 P=1/4）vs 实测
    analytic = (-expected_delta(k, args.bit, initial_pp_full) + C_bias) if k > 0 else 0.0
    for stage in assignment:
        for col in stage:
            for (s, c, t, idx) in col:
                cell = policy(s, c, t, idx)
                if cell:
                    analytic += NAME_BY[cell]["weighted_signed_error"] * (1 << c)
    print(f"\n  解析 (−E[Δ]+C+Σ bias·2^col)   : {analytic:+.3f}  (一阶估计，深层列非 1/4 故略偏)")


if __name__ == "__main__":
    main()
