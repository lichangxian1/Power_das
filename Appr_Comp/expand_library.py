#!/usr/bin/env python3
"""扩样：comp22 全 256 空间；comp32 |e|<=2 取 lowest-wae 2000 个新 rep（+ 已做的 660）。

重写 cand_32.json / cand_22.json 为「待综合代表」集合（每个 canon 取规范成员），
供 gen_verilog.py 重新生成 lib + manifest；随后 char_driver.py --incremental 只补新 cell。

已做集合从现有 library.json 的 cell.canon_key 推断（按名字稳定，不会重复综合）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from enumerate_compressors import enumerate_compressors

N_NEW_32 = 2000          # comp32 新增的 lowest-wae rep 数
MAXE_32 = 2              # comp32 放宽到 |e|<=2
MAXE_22 = 3              # comp22 max_e=3 即覆盖全部 256


def reps_by_canon(cands):
    """每个 canon_key 取规范成员（v_approx 串 == canon_key）。"""
    byk = {}
    for c in cands:
        byk.setdefault(c["canon_key"], []).append(c)
    out = {}
    for k, members in byk.items():
        canon = [m for m in members if "".join(map(str, m["v_approx"])) == k]
        out[k] = canon[0] if canon else members[0]
    return out


def main():
    lib = json.load(open(os.path.join(HERE, "library.json")))["cells"]
    done32 = {c["canon_key"] for c in lib.values()
              if c.get("type") == "32" and not c.get("error")}
    done22 = {c["canon_key"] for c in lib.values()
              if c.get("type") == "22" and not c.get("error")}

    # ---- comp32: |e|<=2 ----
    print(f"[expand] enumerating comp32 |e|<={MAXE_32} ...")
    reps32 = reps_by_canon(enumerate_compressors(3, MAXE_32, [0.25, 0.25, 0.25]))
    new32 = sorted((r for k, r in reps32.items() if k not in done32),
                   key=lambda x: x["weighted_absolute_error"])
    take32 = new32[:N_NEW_32]
    sel32_keys = set(done32) | {r["canon_key"] for r in take32}
    sel32 = [reps32[k] for k in sel32_keys if k in reps32]
    json.dump({"meta": {"max_e": MAXE_32,
                        "note": f"done {len(done32)} + lowest-wae {len(take32)} new",
                        "n": len(sel32)},
               "candidates": sel32}, open(os.path.join(HERE, "cand_32.json"), "w"))
    wae_lo = take32[0]["weighted_absolute_error"] if take32 else None
    wae_hi = take32[-1]["weighted_absolute_error"] if take32 else None
    print(f"[expand] comp32: done={len(done32)} new_take={len(take32)} "
          f"(wae {wae_lo:.3f}..{wae_hi:.3f}) total_sel={len(sel32)}")

    # ---- comp22: full 256 ----
    print(f"[expand] enumerating comp22 full space (max_e={MAXE_22}) ...")
    reps22 = reps_by_canon(enumerate_compressors(2, MAXE_22, [0.25, 0.25]))
    sel22 = list(reps22.values())
    json.dump({"meta": {"max_e": MAXE_22, "note": "full 256 space", "n": len(sel22)},
               "candidates": sel22}, open(os.path.join(HERE, "cand_22.json"), "w"))
    print(f"[expand] comp22: total_sel={len(sel22)} reps (done={len(done22)} "
          f"-> new={len(sel22) - len(done22)})")

    print(f"\n[expand] 下一步: python Appr_Comp/gen_verilog.py  "
          f"&&  python Appr_Comp/char_driver.py --run --incremental")


if __name__ == "__main__":
    main()
