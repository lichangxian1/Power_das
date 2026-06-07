"""
把 area_budget_sweep 的最终 MUL.v 发到远端 DC 服务器综合，获取真实 PPA。

用法：
    python scripts/dc_eval_sweep_results.py \
        --sweep_dir outputs/area_budget_sweep/20260605_105819 \
        --target_delay 2.0 \
        --workers 4
"""

import argparse
import concurrent.futures
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_power_sweep import evaluate_single_routing

BUDGETS = [1160, 1200, 1240, 1280, 1320]
MODES   = ["eda", "proxy"]


def collect_netlists(sweep_dir):
    entries = []
    for mode in MODES:
        for budget in BUDGETS:
            path = os.path.join(sweep_dir, f"power_source_{mode}", f"area_budget_{budget}", "MUL.v")
            if not os.path.exists(path):
                print(f"  [WARN] 找不到: {path}")
                continue
            with open(path) as f:
                content = f.read()
            entries.append({"mode": mode, "budget": budget, "path": path, "content": content})
    return entries


def dc_eval(entry, target_delay, idx):
    print(f"  → 提交 DC: mode={entry['mode']}  budget={entry['budget']}  (idx={idx})")
    result = evaluate_single_routing(
        idx=idx,
        verilog_content=entry["content"],
        bit_width=16,
        target_delay=target_delay,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir",    default="outputs/area_budget_sweep/20260605_105819")
    parser.add_argument("--target_delay", type=float, default=2.0, help="DC 目标延迟 (ns)")
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--out",          default=None)
    args = parser.parse_args()

    sweep_dir = os.path.join(_REPO_ROOT, args.sweep_dir) if not os.path.isabs(args.sweep_dir) else args.sweep_dir
    out_path  = args.out or os.path.join(sweep_dir, "dc_eval_results.json")

    print(f"\n{'='*65}")
    print(f"  DC 综合评估: {sweep_dir}")
    print(f"  target_delay = {args.target_delay} ns   workers = {args.workers}")
    print(f"{'='*65}\n")

    entries = collect_netlists(sweep_dir)
    print(f"  共 {len(entries)} 个网表待综合\n")

    results = [None] * len(entries)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(dc_eval, entry, args.target_delay, i): i
            for i, entry in enumerate(entries)
        }
        for fut in concurrent.futures.as_completed(futs):
            i   = futs[fut]
            res = fut.result()
            entry = entries[i]
            results[i] = {"mode": entry["mode"], "budget": entry["budget"], "dc": res}
            if res.get("success") and not res.get("logic_failed"):
                print(f"  ✓ mode={entry['mode']:5s}  budget={entry['budget']}  "
                      f"area={res['area']:.2f}  delay={abs(res.get('delay',0)):.4f}ns  "
                      f"power={res.get('power_mw', float('inf')):.4f}mW")
            else:
                print(f"  ✗ mode={entry['mode']:5s}  budget={entry['budget']}  FAILED: "
                      f"{str(res.get('log',''))[:80]}")

    # ── 汇总表 ──
    print(f"\n{'='*65}")
    print(f"  汇总 (DC target_delay={args.target_delay}ns)")
    print(f"{'='*65}")
    print(f"{'Mode':>8}  {'Budget':>8}  {'DC area':>10}  {'DC delay':>10}  {'DC power':>12}")
    print(f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    for r in sorted(results, key=lambda x: (x["mode"], x["budget"]) if x else ("z", 0)):
        if r is None:
            continue
        dc = r["dc"]
        if dc.get("success") and not dc.get("logic_failed"):
            print(f"{r['mode']:>8}  {r['budget']:>8}  "
                  f"{dc['area']:>10.2f}  {abs(dc.get('delay',0)):>10.4f}ns  "
                  f"{dc.get('power_mw', float('inf')):>10.4f}mW")
        else:
            print(f"{r['mode']:>8}  {r['budget']:>8}  {'FAILED':>10}  {'—':>10}  {'—':>12}")

    # ── 保存 ──
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"target_delay": args.target_delay, "results": results}, f, indent=2, default=str)
    print(f"\n  结果已保存: {out_path}\n")


if __name__ == "__main__":
    main()
