#!/usr/bin/env python3
"""ZERO cell 进库（幂等）：输出恒 0 的 22/32/42 压缩器 = 槽位粒度截断。

误差统计按库口径 P(bit=1)=0.25：
  ZERO22: bias=-0.5   wae=0.5   er=1-0.75^2=0.4375     maxe=2
  ZERO32: bias=-0.75  wae=0.75  er=1-0.75^3=0.578125   maxe=3
  ZERO42: bias=-1.0   wae=1.0   er=1-0.75^4=0.68359375 maxe=4
area/power/delay = 0.0（真实表征：模块综合为常数，锥体清扫是额外红利）——
依赖两处 falsy 修复（solver.py SlotSpace / arith_das._propose_cell_add）已落地。
group='N'（不能用 'Z'：want-sign 里 'Z' 是通配符）。42 条目不得带 patterns 键。

改动文件（各留 .bak_zero 备份）：
  library.json / selected_compressors.json            (comp32_zero, comp22_zero)
  library42_native.json / selected_compressors42_native.json (comp42n_zero)
  rtl/comp42n_lib.v                                    (module comp42n_zero)
"""
import json
import os
import shutil

APPR = os.path.dirname(os.path.abspath(__file__))


def patch_json(path, fn):
    p = os.path.join(APPR, path)
    bak = p + ".bak_zero"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    data = json.load(open(p))
    changed = fn(data)
    if changed:
        json.dump(data, open(p, "w"), indent=1)
    print(f"{'PATCHED' if changed else 'ok(已存在)'} {path}")


def main():
    z22 = {"type": "22", "canon_key": "0000", "class_size": 1, "is_exact": False,
           "group": "N", "weighted_signed_error": -0.5, "weighted_absolute_error": 0.5,
           "error_rate": 0.4375, "max_error": 2,
           "positive_error_prob": 0.0, "negative_error_prob": 0.4375,
           "sum_lut": [0, 0, 0, 0], "carry_lut": [0, 0, 0, 0]}
    z32 = {"type": "32", "canon_key": "00000000", "class_size": 1, "is_exact": False,
           "group": "N", "weighted_signed_error": -0.75, "weighted_absolute_error": 0.75,
           "error_rate": 0.578125, "max_error": 3,
           "positive_error_prob": 0.0, "negative_error_prob": 0.578125,
           "sum_lut": [0] * 8, "carry_lut": [0] * 8}

    def lib_fn(data):
        c = data["cells"]
        if "comp22_zero" in c and "comp32_zero" in c:
            return False
        c["comp22_zero"] = z22
        c["comp32_zero"] = z32
        return True
    patch_json("library.json", lib_fn)

    def sel_fn(data):
        s = data["selected"]
        if "comp22_zero" in s and "comp32_zero" in s:
            return False
        s["comp32_zero"] = {"name": "comp32_zero", "type": "32", "group": "N",
                            "bias": -0.75, "wae": 0.75, "er": 0.578125, "maxe": 3,
                            "area": 0.0, "power_mw": 0.0, "delay_ns": 0.0,
                            "alias": "comp32_zero"}
        s["comp22_zero"] = {"name": "comp22_zero", "type": "22", "group": "N",
                            "bias": -0.5, "wae": 0.5, "er": 0.4375, "maxe": 2,
                            "area": 0.0, "power_mw": 0.0, "delay_ns": 0.0,
                            "alias": "comp22_zero"}
        return True
    patch_json("selected_compressors.json", sel_fn)

    z42 = {"type": "42", "pattern_bits": 4, "family": "zero", "group": "N",
           "is_exact": False, "v_lut": [0] * 16, "sum_lut": [0] * 16,
           "carry_lut": [0] * 16, "cout_lut": [0] * 16,
           "bias": -1.0, "wae": 1.0, "er": 0.68359375, "maxe": 4,
           "area": 0.0, "power_mw": 0.0, "delay_ns": 0.0,
           "alias": "comp42n_zero"}

    def n42_fn(data):
        c = data["cells"]
        if "comp42n_zero" in c:
            return False
        c["comp42n_zero"] = dict(z42)
        return True
    patch_json("library42_native.json", n42_fn)
    patch_json("selected_compressors42_native.json", n42_fn)

    rtl = os.path.join(APPR, "rtl/comp42n_lib.v")
    src = open(rtl).read()
    if "module comp42n_zero" not in src:
        shutil.copy(rtl, rtl + ".bak_zero")
        with open(rtl, "a") as f:
            f.write(
                "\n// comp42n_zero: ZERO cell (槽位粒度截断) — 输出恒 0，"
                "bias=-1.0 wae=1.0\n"
                "module comp42n_zero (a, b, c, d, sum, carry, cout);\n"
                "    input  a, b, c, d;\n"
                "    output sum, carry, cout;\n"
                "    assign sum   = 1'b0;\n"
                "    assign carry = 1'b0;\n"
                "    assign cout  = 1'b0;\n"
                "endmodule\n")
        print("PATCHED rtl/comp42n_lib.v (+comp42n_zero)")
    else:
        print("ok(已存在) rtl/comp42n_lib.v")


if __name__ == "__main__":
    main()
