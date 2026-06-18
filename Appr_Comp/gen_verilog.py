#!/usr/bin/env python3
"""阶段 1（前半）：把阶段 0 枚举出的候选压缩器生成 DC 可综合的 Verilog 模块。

输入：Appr_Comp/cand_32.json, cand_22.json （由 enumerate_compressors.py 产出）
输出：
    Appr_Comp/rtl/comp32_lib.v   —— 660 个 3:2 置换等价代表，每个一个 module
    Appr_Comp/rtl/comp22_lib.v   —— 36  个 2:2 代表
    Appr_Comp/rtl/manifest.json  —— module_name -> {真值表, 误差指标, canon_key, ...}

设计要点
--------
- 端口与框架里的 FA/HA 完全对齐，方便后续 declare_fa/ha 只换 cell 名即可实例化：
      3:2  ->  module comp32_xxxx (a, b, cin, sum, cout);   // 同 FA
      2:2  ->  module comp22_xx   (a, cin, sum, cout);       // 同 HA（两输入是 a, cin！）
- 每个「输入置换等价类」只生成一个代表（综合后 area/power 对输入置换不变）。
  代表 = 该类的规范真值表（canon_key 本身就是该类字典序最小成员，仍是合法 |e|<=1 候选）。
- 逻辑用真值表 minterm 的 SOP 连续赋值（assign），交给 DC compile_ultra 去优化映射。
  常数输出退化为 1'b0 / 1'b1。
- 模块名由规范真值表编码而来（确定性、可逆）：8 或 4 个 0~3 的值 -> 每值 2bit -> hex。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")


def encode_name(prefix, v_approx):
    """v_approx (list of 0..3) -> '<prefix>_<hex>'，每值 2bit 拼成整数再转 hex。"""
    bits = 0
    for i, v in enumerate(v_approx):
        bits |= (v & 0b11) << (2 * i)
    width_hex = (2 * len(v_approx) + 3) // 4  # 需要的 hex 位数
    return f"{prefix}_{bits:0{width_hex}x}"


def sop(lut, patterns, input_names):
    """从真值表某一位输出 (lut) 生成 SOP 表达式字符串。

    lut[i] 是 pattern i 的输出位；patterns[i] 是形如 '010' 的输入串。
    """
    n_pat = len(patterns)
    ones = [i for i, b in enumerate(lut) if b == 1]
    if not ones:
        return "1'b0"
    if len(ones) == n_pat:
        return "1'b1"
    terms = []
    for i in ones:
        bits = patterns[i]
        lits = []
        for name, bit in zip(input_names, bits):
            lits.append(name if bit == "1" else f"~{name}")
        terms.append("(" + " & ".join(lits) + ")")
    return " | ".join(terms)


def emit_module(name, input_names, patterns, sum_lut, carry_lut, comment):
    ports = ", ".join(input_names + ["sum", "cout"])
    ins = ", ".join(input_names)
    sum_expr = sop(sum_lut, patterns, input_names)
    cout_expr = sop(carry_lut, patterns, input_names)
    return (
        f"// {comment}\n"
        f"module {name} ({ports});\n"
        f"    input  {ins};\n"
        f"    output sum, cout;\n"
        f"    assign sum  = {sum_expr};\n"
        f"    assign cout = {cout_expr};\n"
        f"endmodule\n\n"
    )


def pick_representatives(candidates):
    """每个 canon_key 选一个代表：其 v_approx 串 == canon_key（规范成员）。

    返回 (代表列表, 每类成员数 dict)。
    """
    by_key = {}
    for c in candidates:
        by_key.setdefault(c["canon_key"], []).append(c)
    reps = []
    member_count = {}
    for key, members in by_key.items():
        member_count[key] = len(members)
        canon = [m for m in members if "".join(map(str, m["v_approx"])) == key]
        reps.append(canon[0] if canon else members[0])
    # 稳定排序：exact 在前，其余按 canon_key
    reps.sort(key=lambda c: (c["group"] != "exact", c["canon_key"]))
    return reps, member_count


def build_library(cand_path, prefix, input_names, header):
    with open(cand_path) as f:
        data = json.load(f)
    cands = data["candidates"]
    patterns = cands[0]["patterns"]  # 所有候选共享相同 pattern 顺序
    reps, member_count = pick_representatives(cands)

    v_src = f"// {header}\n// 自动生成 by Appr_Comp/gen_verilog.py — 请勿手改\n\n"
    manifest = {}
    for c in reps:
        name = encode_name(prefix, c["v_approx"])
        comment = (
            f"{prefix} canon={c['canon_key']} group={c['group']} "
            f"bias={c['weighted_signed_error']:+.4f} wae={c['weighted_absolute_error']:.4f} "
            f"ER={c['error_rate']:.4f} maxe={c['max_error']} "
            f"{'EXACT' if c['group'] == 'exact' else ''}"
        ).strip()
        v_src += emit_module(name, input_names, patterns,
                             c["sum_lut"], c["carry_lut"], comment)
        manifest[name] = {
            "type": prefix.replace("comp", ""),  # '32' / '22'
            "canon_key": c["canon_key"],
            "class_size": member_count[c["canon_key"]],
            "is_exact": c["group"] == "exact",
            "group": c["group"],
            "weighted_signed_error": c["weighted_signed_error"],
            "weighted_absolute_error": c["weighted_absolute_error"],
            "error_rate": c["error_rate"],
            "max_error": c["max_error"],
            "positive_error_prob": c["positive_error_prob"],
            "negative_error_prob": c["negative_error_prob"],
            "sum_lut": c["sum_lut"],
            "carry_lut": c["carry_lut"],
            "v_approx": c["v_approx"],
            "ports": input_names + ["sum", "cout"],
        }
    return v_src, manifest, len(reps)


def main():
    os.makedirs(RTL_DIR, exist_ok=True)
    manifest_all = {}

    src32, man32, n32 = build_library(
        os.path.join(HERE, "cand_32.json"), "comp32",
        ["a", "b", "cin"], "3:2 approximate compressor library")
    with open(os.path.join(RTL_DIR, "comp32_lib.v"), "w") as f:
        f.write(src32)
    manifest_all.update(man32)

    src22, man22, n22 = build_library(
        os.path.join(HERE, "cand_22.json"), "comp22",
        ["a", "cin"], "2:2 approximate compressor library")
    with open(os.path.join(RTL_DIR, "comp22_lib.v"), "w") as f:
        f.write(src22)
    manifest_all.update(man22)

    with open(os.path.join(RTL_DIR, "manifest.json"), "w") as f:
        json.dump(manifest_all, f, indent=2)

    print(f"3:2 代表模块: {n32}  -> rtl/comp32_lib.v")
    print(f"2:2 代表模块: {n22}  -> rtl/comp22_lib.v")
    print(f"manifest 共 {len(manifest_all)} 个 module -> rtl/manifest.json")
    # 打印一个 exact 样例供肉眼核对
    ex = [k for k, v in manifest_all.items() if v["is_exact"] and v["type"] == "32"][0]
    print(f"\nexact 3:2 模块名: {ex}  (应等价于 FA)")


if __name__ == "__main__":
    main()
