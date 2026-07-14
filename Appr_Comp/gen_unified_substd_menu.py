#!/usr/bin/env python3
"""生成"一份菜单管全部 22/32/42"的统一 substd 菜单（用户 2026-07-12 要求）。

准则：菜单内每个近似 cell 的 PPA 都优于原生标准单元（substd 硬标准或 in-context
探针证据），即 outputs/2026-07-11_cell_pareto/substd_cell_pareto.png 里选中的 cell：
  - T22：exact + 6 个 substd（standalone area < HA1D0=2.184）
  - T32：exact + 6 个 substd（standalone area < FA1D0=2.856）
  - T42：exact(CT42) + 7 个 comp42s（or4_n/thru_n 过 standalone；orha_n/z/p、
         or4ao_n、or4oa_p 过 in-context 探针，真实 ~1.2-1.7 « 2xFA1D0=5.712，
         见 outputs/2026-07-12_orha_probe/PROBE_ORHA.md）。
排除：原 12 个 comp42n（standalone 10.9-17.1µm²，全在 std 之上，无探针证据）。

产物：Appr_Comp/selected_compressors_all_substd.json，一文件三用：
  - key "selected"：训练菜单（--approx_lib_path，22/32/42 都在这里选）
  - key "cells"   ：42 的 LUT 源（--approx42_library_path，供 native-4 免 cin 识别）
  - key "meta.anchor_area"：exact CT42 锚点面积（外环 area 打分用）

配套 backing store（组件库，非菜单，无需改）：
  --approx_library_path  Appr_Comp/library.json          （22/32 LUT）
  --approx42_rtl_path    Appr_Comp/rtl/comp42n_lib.v      （42 结构化 RTL）

loader 说明：菜单含 type==42 条目 → 走 _load_approx42_table Branch A（显式菜单），
**不再受 --approx42_max_types 截断**（原 20-cell 截断坑消失）；但必须含一个
group==exact 的 CT42 条目（否则 type_table_42[0] 非 exact，断言失败）。

用法: python -m Appr_Comp.gen_unified_substd_menu
"""
import json
import os

APPR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(APPR, "selected_compressors_all_substd.json")

# 42 菜单 cell（全部 PPA<std）。exact 锚点面积用 CT42_BAL medium（与现 42 菜单一致）。
COMP42S = ["comp42s_orha_n", "comp42s_orha_z", "comp42s_orha_p",
           "comp42s_or4ao_n", "comp42s_or4oa_p", "comp42s_or4_n", "comp42s_thru_n"]
CT42_ANCHOR_AREA = 17.304


def main():
    substd = json.load(open(os.path.join(APPR, "selected_compressors_substd.json")))["selected"]
    lib42 = json.load(open(os.path.join(APPR, "library42_native.json")))["cells"]

    selected = {}
    # 1) 22/32 段：直接搬 substd 菜单（exact + 各 6 个）
    for k, v in substd.items():
        selected[k] = dict(v)

    # 2) 42 exact 锚点（Branch A 必须显式含 group==exact）
    selected["comp42_exact"] = {
        "name": "CT42", "type": "42", "group": "exact", "alias": "comp42_exact",
        "bias": 0.0, "wae": 0.0, "er": 0.0, "maxe": 0,
        "area": CT42_ANCHOR_AREA, "power_mw": None, "delay_ns": None,
    }
    # 3) 42 近似段：7 个 comp42s（菜单条目 = 元数据；LUT 从 "cells" 查）
    cells = {}
    for i, n in enumerate(COMP42S, 1):
        c = lib42[n]
        assert c.get("pattern_bits") == 4, f"{n} 非 native-4！"
        selected[n] = {
            "name": n, "type": "42", "group": c["group"], "alias": f"comp42_substd_{i}",
            "bias": c["bias"], "wae": c["wae"], "er": c["er"], "maxe": c["maxe"],
            "area": c["area"], "power_mw": c.get("dyn_mw"), "delay_ns": c.get("tmax"),
        }
        cells[n] = dict(c)   # 完整 LUT 条目，供 approx42_library_path 用

    out = {
        "meta": {
            "purpose": "统一 substd 菜单：22/32/42 全在此选，所有 cell PPA<std",
            "anchor_area": CT42_ANCHOR_AREA,
            "backing": {"approx_library_path": "Appr_Comp/library.json",
                        "approx42_rtl_path": "Appr_Comp/rtl/comp42n_lib.v"},
            "note": "comp42s 探针证据 outputs/2026-07-12_orha_probe/PROBE_ORHA.md；"
                    "Branch A 菜单不受 approx42_max_types 截断",
            "counts": {"T22": sum(1 for v in selected.values() if v["type"] == "22"),
                       "T32": sum(1 for v in selected.values() if v["type"] == "32"),
                       "T42": sum(1 for v in selected.values() if v["type"] == "42")},
        },
        "selected": selected,
        "cells": cells,
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    c = out["meta"]["counts"]
    print(f"wrote {OUT}")
    print(f"  T22={c['T22']} (1 exact + {c['T22']-1} substd)")
    print(f"  T32={c['T32']} (1 exact + {c['T32']-1} substd)")
    print(f"  T42={c['T42']} (1 exact + {c['T42']-1} comp42s), cells LUT={len(cells)}")


if __name__ == "__main__":
    main()
