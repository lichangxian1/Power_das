#!/usr/bin/env python3
"""恒零 cell 进统一 substd 菜单（幂等，.bak_zero_unified 备份）。

背景（07-13 双评审发现 #1）：统一菜单 selected_compressors_all_substd.json 只收
substd 家族，真·恒零 cell（输出恒 0 = 槽位级截断，budget_sweep 证明的档间主力）
不在搜索空间；且 N/Z/P 是偏置符号分组，恒零 cell 按约定挂 N 组（add_zero_cells.py），
group=='Z' 检测两头都错。本脚本注入三个恒零 cell 并打 **const_zero: true** 标志
（trainer._zero_entry_of 改读该标志）。

注入内容：
  selected: comp22_zero / comp32_zero / comp42n_zero（area/power/delay=0，PPA<std 平凡成立）
  cells:    comp42n_zero 的 v_lut + pattern_bits=4（Branch A 的 42 LUT backing 在同文件）
  rtl/comp42s_standalone.v: 追加 module comp42n_zero（从 comp42n_lib.v 抄，42 抽取单文件）
误差统计按库口径 P(bit=1)=0.25（与 add_zero_cells.py 相同）。
"""
import json
import os
import re
import shutil

APPR = os.path.dirname(os.path.abspath(__file__))
MENU = os.path.join(APPR, "selected_compressors_all_substd.json")
LIB42 = os.path.join(APPR, "library42_native.json")
RTL_SRC = os.path.join(APPR, "rtl", "comp42n_lib.v")
RTL_DST = os.path.join(APPR, "rtl", "comp42s_standalone.v")

ZEROS = {
    "comp22_zero": {"type": "22", "group": "N", "bias": -0.5, "wae": 0.5,
                    "er": 0.4375, "maxe": 2},
    "comp32_zero": {"type": "32", "group": "N", "bias": -0.75, "wae": 0.75,
                    "er": 0.578125, "maxe": 3},
    "comp42n_zero": {"type": "42", "group": "N", "bias": -1.0, "wae": 1.0,
                     "er": 0.68359375, "maxe": 4},
}


def backup(p, suffix=".bak_zero_unified"):
    if not os.path.exists(p + suffix):
        shutil.copy(p, p + suffix)


def main():
    menu = json.load(open(MENU))
    sel, cells = menu["selected"], menu.setdefault("cells", {})
    changed = False
    for name, meta in ZEROS.items():
        if name not in sel:
            sel[name] = {
                "name": name, "type": meta["type"], "group": meta["group"],
                "alias": name, "bias": meta["bias"], "wae": meta["wae"],
                "er": meta["er"], "maxe": meta["maxe"],
                "area": 0.0, "power_mw": 0.0, "delay_ns": 0.0,
                "const_zero": True,
            }
            changed = True
        elif not sel[name].get("const_zero"):
            sel[name]["const_zero"] = True
            changed = True
    # 42 恒零的 LUT backing（Branch A 从统一菜单自己的 cells 段取）
    if "comp42n_zero" not in cells:
        lib42 = json.load(open(LIB42))["cells"]
        src = lib42.get("comp42n_zero")
        assert src is not None, "library42_native.json 缺 comp42n_zero（先跑 add_zero_cells.py）"
        entry = dict(src)
        entry["pattern_bits"] = 4          # 无 cin 端口，发射不接 .cin
        cells["comp42n_zero"] = entry
        changed = True
    if changed:
        backup(MENU)
        json.dump(menu, open(MENU, "w"), indent=1)
    print(f"{'PATCHED' if changed else 'ok(已存在)'} {os.path.basename(MENU)}")

    # RTL：把 comp42n_zero 模块追加进 42 抽取单文件
    dst = open(RTL_DST).read()
    if "module comp42n_zero" not in dst:
        src = open(RTL_SRC).read()
        m = re.search(r"module comp42n_zero\b.*?endmodule", src, re.S)
        assert m, "comp42n_lib.v 缺 module comp42n_zero（先跑 add_zero_cells.py）"
        backup(RTL_DST)
        with open(RTL_DST, "a") as f:
            f.write("\n\n" + m.group(0) + "\n")
        print(f"PATCHED {os.path.basename(RTL_DST)} (+module comp42n_zero)")
    else:
        print(f"ok(已存在) {os.path.basename(RTL_DST)}")


if __name__ == "__main__":
    main()
