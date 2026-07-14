#!/usr/bin/env python3
"""cell 相加 vs DC 综合差距测试。

从 outputs/ 抽取 N 个含近似压缩器的 MUL.v：
  1) 解析 RTL 实例化的 cell（FA/HA/CT42/comp*），按两种口径求面积和：
     - lib   : library.json standalone 表征（FA=comp32_e994 10.92 等）
     - inctx : arith_das._EXACT_AREA_INCTX 锚点（FA=2.856/HA=2.184/CT42=5.712），
               approx cell 仍用 library 面积（与训练 objective 同口径）
     功耗和只有 lib 口径：Σ(dyn_w+leak_w)（表征条件 static_prob=0.25 toggle=0.125）。
  2) 远端 DC（sandbox_base_dcpwr，compile @1.5ns，vectorless power）综合整个 MUL。
  3) 输出 CSV + 比值统计。断点续跑：已 success 的 design 跳过。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os

import run_power_sweep

FIELDS = [
    "design", "rtl_path", "n_fa", "n_ha", "n_ct42", "n_nocarry", "n_comp",
    "cellsum_area_lib", "cellsum_area_inctx", "cellsum_power_mw",
    "area_dc", "power_dc_mw", "delay_dc", "success", "error",
]

INST_RE = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", re.M)
KEYWORDS = {"module", "input", "output", "wire", "assign", "endmodule"}

# 标 standalone 表征口径的 exact 锚点（library.json 中的 exact 条目 / CT42 单测表征）
LIB_EXACT = {"FA": ("comp32_e994", None), "HA": ("comp22_94", None)}
CT42_AREA_LIB = 18.14  # outputs/2026-07-07_ct42_char, compile_ultra standalone
# in-context 锚点（trainer/arith_das.py _EXACT_AREA_INCTX）
INCTX = {"FA": 2.856, "HA": 2.184, "CT42": 5.712}


def load_library() -> dict:
    cells = json.loads((ROOT / "Appr_Comp/library.json").read_text())["cells"]
    native = json.loads((ROOT / "Appr_Comp/library42_native.json").read_text())
    for name, cell in (native.get("cells") or native).items():
        cells.setdefault(name, cell)
    return cells


def parse_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in INST_RE.finditer(text):
        name = m.group(1)
        if name in KEYWORDS:
            continue
        counts[name] = counts.get(name, 0) + 1
    counts.pop("CompressorTree", None)  # 顶层结构实例，不是 cell
    counts.pop("MUL", None)
    return counts


def cellsum(counts: dict[str, int], lib: dict) -> dict:
    fa = counts.get("FA", 0) + counts.get("FA_no_carry", 0)
    ha = counts.get("HA", 0) + counts.get("HA_no_carry", 0)
    ct42 = counts.get("CT42", 0)
    nocarry = counts.get("FA_no_carry", 0) + counts.get("HA_no_carry", 0)
    fa_lib = lib["comp32_e994"]
    ha_lib = lib["comp22_94"]
    area_lib = fa * fa_lib["area"] + ha * ha_lib["area"] + ct42 * CT42_AREA_LIB
    area_inctx = fa * INCTX["FA"] + ha * INCTX["HA"] + ct42 * INCTX["CT42"]
    pw_w = (
        fa * (fa_lib.get("dyn_w", 0) + fa_lib.get("leak_w", 0))
        + ha * (ha_lib.get("dyn_w", 0) + ha_lib.get("leak_w", 0))
        + ct42 * 2 * (fa_lib.get("dyn_w", 0) + fa_lib.get("leak_w", 0))  # CT42 无单独功耗表征，按 2xFA 计
    )
    n_comp = 0
    for name, n in counts.items():
        if name in ("FA", "HA", "CT42", "FA_no_carry", "HA_no_carry"):
            continue
        n_comp += n
        cell = lib.get(name, {})
        a = cell.get("area", 0.0) or 0.0
        area_lib += n * a
        area_inctx += n * a
        pw_w += n * ((cell.get("dyn_w", 0) or 0) + (cell.get("leak_w", 0) or 0))
    return {
        "n_fa": fa, "n_ha": ha, "n_ct42": ct42, "n_nocarry": nocarry, "n_comp": n_comp,
        "cellsum_area_lib": round(area_lib, 4),
        "cellsum_area_inctx": round(area_inctx, 4),
        "cellsum_power_mw": round(pw_w * 1e3, 6),
    }


def pick_rtls(n: int, seed: int) -> list[Path]:
    """去重（按内容 md5）后抽 n 个含近似 cell 的 MUL.v。"""
    seen: set[str] = set()
    pool: list[Path] = []
    for p in sorted((ROOT / "outputs").rglob("MUL.v")):
        text = p.read_text()
        if not re.search(r"^\s+comp[0-9A-Za-z_]+\s+\w+\s*\(", text, re.M):
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        pool.append(p)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-delay", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/2026-07-14_cellsum_vs_dc/results.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run_power_sweep.EDA_BASE_DIR = os.environ.get(
        "EDA_BASE_DIR_DC", "/home/lchangxian/sandbox/sandbox_base_dcpwr"
    )

    lib = load_library()
    rtls = pick_rtls(args.n, args.seed)
    print(f"候选 RTL: {len(rtls)}", flush=True)

    rows: dict[str, dict] = {}
    if args.out.is_file():
        with args.out.open(newline="", encoding="utf-8") as f:
            rows = {r["design"]: r for r in csv.DictReader(f)}

    def save() -> None:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows[k] for k in sorted(rows))

    jobs = []
    for p in rtls:
        design = str(p.relative_to(ROOT / "outputs")).replace("/", "__").removesuffix("__MUL.v")
        prev = rows.get(design)
        if prev and str(prev.get("success", "")).lower() == "true":
            continue
        text = p.read_text()
        base = {"design": design, "rtl_path": str(p.relative_to(ROOT))}
        base.update(cellsum(parse_counts(text), lib))
        jobs.append((base, text))
    print(f"DC jobs={len(jobs)} (已完成 {len(rtls) - len(jobs)})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                run_power_sweep.evaluate_single_routing, 930000 + i, text, 16, args.target_delay
            ): base
            for i, (base, text) in enumerate(jobs)
        }
        done = 0
        for fut in concurrent.futures.as_completed(pending):
            base = pending[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"success": False, "log": repr(exc)}
            row = dict(base)
            row.update({
                "area_dc": res.get("area", ""),
                "power_dc_mw": res.get("power_mw", ""),
                "delay_dc": abs(res["delay"]) if res.get("delay") is not None else "",
                "success": bool(res.get("success")),
                "error": "" if res.get("success") else str(res.get("log", ""))[-500:],
            })
            rows[base["design"]] = row
            done += 1
            save()
            ok = "OK " if row["success"] else "FAIL"
            print(f"[{done}/{len(jobs)}] {ok} {base['design']}: "
                  f"lib={base['cellsum_area_lib']} inctx={base['cellsum_area_inctx']} "
                  f"dc={row['area_dc']}", flush=True)
    save()
    print(f"完成，结果: {args.out}", flush=True)


if __name__ == "__main__":
    main()
