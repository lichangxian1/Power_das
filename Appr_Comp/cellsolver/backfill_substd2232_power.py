#!/usr/bin/env python3
"""把 substd2232_char.json 的 dyn power 回填进 library.json + 统一菜单（幂等+备份）。

前置：char_substd2232.py --run 产出 cellsolver/substd2232_char.json。
改动（各留 .bak_pwr 备份）：
  library.json                          cells[name].power_mw / dyn_w
  selected_compressors_all_substd.json  selected[key].power_mw（22/32 段）
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
APPR = os.path.dirname(HERE)
CHAR = os.path.join(HERE, "substd2232_char.json")


def bak(p):
    if not os.path.exists(p + ".bak_pwr"):
        shutil.copy(p, p + ".bak_pwr")


def main():
    ch = json.load(open(CHAR))["cells"]

    lib_p = os.path.join(APPR, "library.json")
    bak(lib_p)
    lib = json.load(open(lib_p))
    n = 0
    for name, c in ch.items():
        if c.get("dyn_mw") is None or name not in lib["cells"]:
            continue
        e = lib["cells"][name]
        e["power_mw"] = round(c["dyn_mw"], 7)
        e["dyn_w"] = c["dyn_mw"] / 1e3
        if c.get("tmax") is not None:
            e["delay_ns"] = c["tmax"]
        n += 1
    json.dump(lib, open(lib_p, "w"), indent=1)
    print(f"library.json: 回填 {n} cell power")

    menu_p = os.path.join(APPR, "selected_compressors_all_substd.json")
    bak(menu_p)
    menu = json.load(open(menu_p))
    m = 0
    for k, v in menu["selected"].items():
        nm = v.get("name")
        if nm in ch and ch[nm].get("dyn_mw") is not None:
            v["power_mw"] = round(ch[nm]["dyn_mw"], 7)
            if ch[nm].get("tmax") is not None:
                v["delay_ns"] = ch[nm]["tmax"]
            m += 1
    json.dump(menu, open(menu_p, "w"), indent=1, ensure_ascii=False)
    print(f"selected_compressors_all_substd.json: 回填 {m} cell power")


if __name__ == "__main__":
    main()
