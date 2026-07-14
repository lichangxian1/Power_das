#!/usr/bin/env python3
"""comp42s（Sayadi 启发 sub-std 近似 4:2）进库——幂等 + 备份。

前置：先跑 `python -m Appr_Comp.cellsolver.gen_substd42_cells --run` 产出
cellsolver/substd42_char.json（DC 表征，dc_char 同口径）。

改动文件（各留 .bak_substd42 备份）：
  rtl/comp42n_lib.v        追加 8 个结构化 module（trainer 按名原样抽取发射）
  library42_native.json    追加 8 个条目（library 目录册；wae/bias/LUT/表征 PPA）

--menu 额外把 7 个探针通过的 cell 写进 selected_compressors42_native.json
（主 T42 训练菜单，.bak_substd42 备份；用户 2026-07-12 批准）：
orha_n/z/p、or4ao_n、or4oa_p、or4_n、thru_n。area 存 standalone 标称
（与全库同口径；solver 侧按 ×0.262 校准，探针实测 1.68 见
outputs/2026-07-12_orha_probe/PROBE_ORHA.md）。
⚠ 菜单从 13 → 20 条：launcher 必须传 --approx42_max_types ≥ 20，否则新 cell
（按 dict 顺序在尾部）会被截断——deepk launcher 的 13 连 zero 都截了。
"""
import json
import os
import re
import shutil
import sys

APPR = os.path.dirname(os.path.abspath(__file__))
CHAR = os.path.join(APPR, "cellsolver", "substd42_char.json")
MENU_CELLS = ["comp42s_orha_n", "comp42s_orha_z", "comp42s_orha_p",
              "comp42s_or4ao_n", "comp42s_or4oa_p", "comp42s_or4_n",
              "comp42s_thru_n"]


def main():
    cells = json.load(open(CHAR))["cells"]
    bad = {n: c for n, c in cells.items() if not c.get("area") and c["area"] != 0.0}
    assert not bad, f"missing char area: {list(bad)}"

    # 1) library42_native.json
    lib_p = os.path.join(APPR, "library42_native.json")
    bak = lib_p + ".bak_substd42"
    if not os.path.exists(bak):
        shutil.copy(lib_p, bak)
    lib = json.load(open(lib_p))
    added = [n for n in cells if n not in lib["cells"]]
    for n in added:
        lib["cells"][n] = cells[n]
    if added:
        json.dump(lib, open(lib_p, "w"), indent=1)
    print(f"library42_native.json: +{len(added)} cells" if added else
          "library42_native.json: ok(已存在)")

    # 2) rtl/comp42n_lib.v
    rtl_p = os.path.join(APPR, "rtl", "comp42n_lib.v")
    src42s = open(os.path.join(APPR, "rtl", "comp42s_standalone.v")).read()
    dst = open(rtl_p).read()
    todo = [n for n in cells if f"module {n} " not in dst]
    if todo:
        if not os.path.exists(rtl_p + ".bak_substd42"):
            shutil.copy(rtl_p, rtl_p + ".bak_substd42")
        chunks = []
        for n in todo:
            m = re.search(rf"(?ms)(//[^\n]*\n)?^module {re.escape(n)} .*?^endmodule\n",
                          src42s)
            assert m, n
            chunks.append(m.group(0))
        with open(rtl_p, "a") as f:
            f.write("\n// ==== comp42s: Sayadi-inspired sub-std cells "
                    "(add_substd42_cells.py) ====\n" + "\n".join(chunks))
        print(f"rtl/comp42n_lib.v: +{len(todo)} modules")
    else:
        print("rtl/comp42n_lib.v: ok(已存在)")

    # 3) --menu: 写进主 T42 训练菜单
    if "--menu" in sys.argv:
        menu_p = os.path.join(APPR, "selected_compressors42_native.json")
        if not os.path.exists(menu_p + ".bak_substd42"):
            shutil.copy(menu_p, menu_p + ".bak_substd42")
        menu = json.load(open(menu_p))
        add = [n for n in MENU_CELLS if n not in menu["cells"]]
        for n in add:
            e = dict(cells[n])
            e["name"] = n
            e["alias"] = n
            menu["cells"][n] = e
        if add:
            menu.setdefault("meta", {})["substd42_note"] = (
                "comp42s 7 cells added 2026-07-12 (probe-passed, "
                "outputs/2026-07-12_orha_probe/PROBE_ORHA.md); "
                "launcher MUST pass --approx42_max_types >= %d" % len(menu["cells"]))
            json.dump(menu, open(menu_p, "w"), indent=1)
        print(f"selected_compressors42_native.json: +{len(add)} -> "
              f"{len(menu['cells'])} cells (approx42_max_types 需 ≥ {len(menu['cells'])})"
              if add else "selected_compressors42_native.json: ok(已存在)")


if __name__ == "__main__":
    main()
