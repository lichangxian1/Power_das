#!/usr/bin/env python3
"""恢复用：从远端孤儿 sandbox 取回 ppa_char.rpt，解析后并入 library.json。

背景：char_driver --run 的本地编排进程被杀（exit 144，疑似后台时限），但远端 dc_shell
仍把 1062 个 comp32 表征完并写满 reports/ppa_char.rpt。这里直接取回+合并，避免重跑 DC。

复用 char_driver.parse_ppa；合并口径与 char_driver.main() 完全一致
（entry = manifest[mod] + ppa[mod]，保留已有 cell）。

用法:
    python Appr_Comp/harvest_char.py --sandbox char_dd7431            # 真合并
    python Appr_Comp/harvest_char.py --sandbox char_dd7431 --dry-run  # 只看统计不写
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from char_driver import parse_ppa, EDA_USER, EDA_HOST, EDA_PORT, EDA_WORK_ROOT

RTL_DIR = os.path.join(HERE, "rtl")
LIB = os.path.join(HERE, "library.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", required=True, help="远端 sandbox 目录名，如 char_dd7431")
    ap.add_argument("--static-prob", type=float, default=0.25)
    ap.add_argument("--toggle-rate", type=float, default=0.125)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    remote_rpt = f"{EDA_WORK_ROOT.rstrip('/')}/{args.sandbox}/reports/ppa_char.rpt"
    print(f"[harvest] fetching {EDA_USER}@{EDA_HOST}:{remote_rpt}")
    res = subprocess.run(
        ["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"cat {remote_rpt}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    text = res.stdout or ""
    ppa = parse_ppa(text)
    ok = sum(1 for v in ppa.values() if not v.get("error"))
    err = sum(1 for v in ppa.values() if v.get("error"))
    print(f"[harvest] parsed PPA lines: {ok} ok / {err} error / {len(ppa)} total")
    if ok == 0:
        print("[harvest] ⚠️ 无有效 PPA，停止。前 40 行：")
        print("\n".join(text.splitlines()[:40]))
        sys.exit(1)

    man = json.load(open(os.path.join(RTL_DIR, "manifest.json")))
    names = [l.strip() for l in open(os.path.join(RTL_DIR, "module_list.txt")) if l.strip()]
    existing = json.load(open(LIB)).get("cells", {}) if os.path.exists(LIB) else {}
    print(f"[harvest] module_list={len(names)}  existing_lib={len(existing)}")

    missing = [m for m in names if m not in ppa]
    if missing:
        print(f"[harvest] ⚠️ {len(missing)} 个 module 在 rpt 里没有 PPA（远端可能没跑完）："
              f" e.g. {missing[:5]}")

    library = dict(existing)
    added = 0
    for mod in names:
        if mod not in ppa:
            continue
        entry = dict(man[mod])
        entry.update(ppa[mod])
        library[mod] = entry
        added += 1
    print(f"[harvest] 将并入 {added} 个新 cell -> 总 {len(library)}")

    if args.dry_run:
        print("[harvest] dry-run，未写盘。")
        return
    with open(LIB, "w") as f:
        json.dump({"meta": {"static_prob": args.static_prob,
                            "toggle_rate": args.toggle_rate}, "cells": library}, f, indent=2)
    print(f"[harvest] -> {LIB}  ({len(library)} cells)")


if __name__ == "__main__":
    main()
