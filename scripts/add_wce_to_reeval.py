#!/usr/bin/env python3
"""给已有训练 run 的 reeval 报告补 WCE 列（离线，不跑 EDA）。

对 run 目录下每个 k*/best_info.json 设计输出：
  wce_mc             — verilator 16M MC 实测 max|e|（真实 WCE 的下估计，来自 best_info.measured_error）
  wce_trunc_analytic — 截断部分的解析 WCE（精确值；从训练日志最后一条 [trunc] 行的 WCE_trunc= 取，
                       该行反映实际使用的常数 C/C*；无截断 -> 0）
  wce_cell_bound     — 近似 cell 部分的可加上界 Σ max_error(cell)·2^col
                       （col 由 best_info.assignment 按 CompressorGraph 顶点枚举顺序复原）
  wce_bound          — wce_trunc_analytic + wce_cell_bound（真实 WCE 的硬上界）

用法：python3 scripts/add_wce_to_reeval.py <run_dir> [--bits 16]
产物：<run_dir>/reeval_xa_wce.csv（若存在 reeval_xa.csv 则按 design 合并其 PPA 列）
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def and_pp_heights(bits):
    n_col = 2 * bits - 1
    return [min(c + 1, bits, 2 * bits - 1 - c) for c in range(n_col)]


def flatten_vertices(pp, assignment):
    """复刻 trainer.arith_das.CompressorGraph 的 vertex_list 枚举顺序，
    返回 node_idx -> (stage, col, type, idx)。"""
    stage_num = len(assignment)
    col_num = len(assignment[0])
    assert col_num == len(pp), f"assignment col_num={col_num} != len(pp)={len(pp)}"

    dec = {t: [[0] * col_num for _ in range(stage_num)] for t in (0, 1, 4)}
    for s in range(stage_num):
        for c in range(col_num):
            for v in assignment[s][c]:
                t = v[2]
                if t not in dec:
                    raise ValueError(f"unexpected vertex type {t}")
                dec[t][s][c] += 1

    slice_size = [[0] * col_num for _ in range(stage_num + 1)]
    slice_size[0] = list(pp)
    for s in range(1, stage_num + 1):
        for c in range(col_num):
            v = (
                slice_size[s - 1][c]
                - 2 * dec[0][s - 1][c]
                - dec[1][s - 1][c]
                - 3 * dec[4][s - 1][c]
            )
            if c > 0:
                v += dec[0][s - 1][c - 1] + dec[1][s - 1][c - 1] + 2 * dec[4][s - 1][c - 1]
            slice_size[s][c] = v

    vertex_list = []
    for c in range(col_num):
        for pp_idx in range(pp[c]):
            vertex_list.append((-1, c, 2, pp_idx))
        for s in range(stage_num + 1):
            if s < stage_num:
                for v in assignment[s][c]:
                    vertex_list.append(tuple(v))
            port = (
                3 * dec[0][s][c] + 2 * dec[1][s][c] + 4 * dec[4][s][c]
                if s < stage_num
                else 0
            )
            for visual_idx in range(slice_size[s][c] - port):
                vertex_list.append((s, c, 3, visual_idx))
    return vertex_list


def load_maxe_table():
    table = {}
    for name in ("library.json", "library42_pair32_func.json"):
        p = ROOT / "Appr_Comp" / name
        if not p.exists():
            continue
        data = json.load(open(p))
        cells = data.get("cells", data)
        for cname, entry in cells.items():
            if isinstance(entry, dict) and "max_error" in entry:
                table[cname] = abs(entry["max_error"])
    return table


def parse_trunc_wce(log_path):
    if not log_path.exists():
        return None
    wce = 0
    found = False
    for line in open(log_path, errors="ignore"):
        m = re.search(r"\[trunc\].*WCE_trunc=(\d+)", line)
        if m:
            wce = int(m.group(1))
            found = True
    return wce if found else 0  # 无 [trunc] 行 = 未截断 -> 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bits", type=int, default=16)
    args = ap.parse_args()
    base = Path(args.run_dir).resolve()
    pp = and_pp_heights(args.bits)
    maxe = load_maxe_table()

    ppa = {}
    ppa_csv = base / "reeval_xa.csv"
    if ppa_csv.exists():
        for row in csv.DictReader(open(ppa_csv)):
            ppa[row["design"]] = row

    rows = []
    for p in sorted(x for x in base.glob("k*") if x.is_dir()):
        bi = p / "best_info.json"
        if not bi.exists():
            continue
        info = json.load(open(bi))
        measured = info.get("measured_error") or {}
        wce_mc = measured.get("wce_mc")
        wce_trunc = parse_trunc_wce(base / f"{p.name}.log")

        cell_bound = 0
        cell_note = ""
        approx_cells = info.get("approx_cells") or {}
        if approx_cells:
            try:
                vlist = flatten_vertices(pp, info["assignment"])
                for nid, cname in approx_cells.items():
                    col = vlist[int(nid)][1]
                    me = maxe.get(cname)
                    if me is None:
                        cell_note = f"unknown maxe for {cname}"
                        cell_bound = None
                        break
                    cell_bound += me * (2 ** col)
            except Exception as e:  # noqa: BLE001
                cell_bound, cell_note = None, f"col-recovery failed: {e}"

        wce_bound = (
            wce_trunc + cell_bound
            if (wce_trunc is not None and cell_bound is not None)
            else None
        )
        row = {
            "design": p.name,
            "n_approx": len(approx_cells),
            "wce_mc": wce_mc,
            "wce_trunc_analytic": wce_trunc,
            "wce_cell_bound": cell_bound,
            "wce_bound": wce_bound,
            "note": cell_note,
        }
        for k in ("mred", "med", "area_dc", "power_xa_mw"):
            row[k] = ppa.get(p.name, {}).get(k, "")
        rows.append(row)

    if not rows:
        print(f"no k*/best_info.json under {base}", file=sys.stderr)
        sys.exit(1)

    out = base / "reeval_xa_wce.csv"
    cols = ["design", "mred", "med", "n_approx", "wce_mc",
            "wce_trunc_analytic", "wce_cell_bound", "wce_bound",
            "area_dc", "power_xa_mw", "note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out}")
    hdr = f"{'design':<18}{'n_apx':>6}{'wce_mc':>10}{'wce_trunc':>10}{'cell_bnd':>10}{'wce_bound':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['design']:<18}{r['n_approx']:>6}{str(r['wce_mc']):>10}"
              f"{str(r['wce_trunc_analytic']):>10}{str(r['wce_cell_bound']):>10}{str(r['wce_bound']):>10}"
              f"  {r['note']}")


if __name__ == "__main__":
    main()
