#!/usr/bin/env python3
"""
DC timing-constraint PPA sweep for three 16-bit multipliers.

Designs:
  - eda    : RL-searched compressor tree (eda power source)   -> power_source_eda/unconstrained/MUL.v
  - proxy  : RL-searched compressor tree (proxy power source) -> power_source_proxy/unconstrained/MUL.v
  - dw     : standard DesignWare baseline (behavioral a*b, DC datapath infers DW)

For every (design x target_delay) the netlist is shipped to the remote DC server
(via run_power_sweep.evaluate_single_routing) and synthesized under that timing
constraint. We collect (area, delay, power) and draw two line+point charts:
    1) y = delay (ns), x = area (um^2)
    2) y = delay (ns), x = power (mW)

Usage:
    python scripts/dc_timing_sweep_ppa.py \
        --run_dir outputs/area_budget_sweep/20260613_163000 \
        --delays "0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4" \
        --workers 6
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_power_sweep
from run_power_sweep import evaluate_single_routing

# Standard DesignWare baseline: same interface as the searched MUL.v
# (clk, a[15:0], b[15:0], out[30:0]); DC's datapath optimizer infers/maps an
# unsigned multiplier (DW02_mult family). out keeps the lower 31 bits, matching
# the AND-array searched designs and the remote testbench's masked ground truth.
DW_RTL = """module MUL(
    input wire clk,
    input wire [15:0] a,
    input wire [15:0] b,
    output wire [30:0] out
);
    assign out = a * b;
endmodule
"""

DEFAULT_DELAYS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]

DESIGNS = ["eda", "proxy", "dw"]
LABELS = {"eda": "EDA-guided", "proxy": "Proxy-guided", "dw": "DesignWare (a*b)"}
COLORS = {"eda": "#d62728", "proxy": "#1f77b4", "dw": "#2ca02c"}
MARKERS = {"eda": "s", "proxy": "o", "dw": "^"}


def load_designs(run_dir):
    contents, paths = {}, {}
    netlists = {
        "eda": os.path.join(run_dir, "power_source_eda", "unconstrained", "MUL.v"),
        "proxy": os.path.join(run_dir, "power_source_proxy", "unconstrained", "MUL.v"),
    }
    for name, path in netlists.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} netlist not found: {path}")
        with open(path) as f:
            contents[name] = f.read()
        paths[name] = path
    contents["dw"] = DW_RTL
    paths["dw"] = "<generated behavioral a*b>"
    return contents, paths


def _eval_one(idx, design, td, contents, bit_width):
    res = evaluate_single_routing(
        idx=idx,
        verilog_content=contents[design],
        bit_width=bit_width,
        target_delay=td,
    )
    ok = res.get("success") and not res.get("logic_failed")
    if ok:
        area = res.get("area")
        delay = abs(res.get("delay")) if res.get("delay") is not None else None
        power = res.get("power_mw")
        entry = {"target_delay": td, "area": area, "delay": delay, "power_mw": power}
        msg = (
            f"  OK  {design:5s} constraint={td:>4.2f}ns -> "
            f"area={area:>9.2f}  delay={delay:>7.4f}ns  power={power:>8.4f}mW"
        )
    else:
        log = res.get("log")
        entry = {
            "target_delay": td,
            "area": None,
            "delay": None,
            "power_mw": None,
            "failed": True,
            "logic_failed": res.get("logic_failed", False),
            "error_log": (log[-1500:] if isinstance(log, str) else log),
        }
        log_tail = (log.strip().splitlines()[-1] if isinstance(log, str) and log.strip() else "")
        msg = (
            f"  XX  {design:5s} constraint={td:>4.2f}ns FAILED "
            f"(success={res.get('success')} logic_failed={res.get('logic_failed')}) "
            f"| {log_tail[:160]}"
        )
    return design, td, entry, msg


def run_jobs(jobs, contents, workers, bit_width):
    """jobs: list of (design, target_delay). Returns list of (design, td, entry)."""
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_eval_one, i, design, td, contents, bit_width): (design, td)
            for i, (design, td) in enumerate(jobs)
        }
        for fut in concurrent.futures.as_completed(futs):
            design, td, entry, msg = fut.result()
            print(msg, flush=True)
            out.append((design, td, entry))
    return out


def run_sweep(contents, delays, workers, bit_width):
    jobs = [(design, td) for design in DESIGNS for td in delays]
    flat = run_jobs(jobs, contents, workers, bit_width)
    results = {design: [] for design in DESIGNS}
    for design, _td, entry in flat:
        results[design].append(entry)
    for design in DESIGNS:
        results[design].sort(key=lambda r: r["target_delay"])
    return results


def _is_failed(entry):
    return (
        entry.get("area") is None
        or entry.get("delay") is None
        or entry.get("power_mw") is None
    )


def save_artifacts(results, out_dir, run_dir, delays, paths, annotate=False):
    json_path = os.path.join(out_dir, "dc_timing_sweep.json")
    with open(json_path, "w") as f:
        json.dump(
            {"run_dir": run_dir, "delays": delays, "paths": paths, "results": results},
            f,
            indent=2,
        )
    print(f"Wrote {json_path}", flush=True)

    csv_path = os.path.join(out_dir, "dc_timing_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "target_delay_ns", "area_um2", "delay_ns", "power_mw"])
        for design in DESIGNS:
            for r in results[design]:
                w.writerow(
                    [design, r["target_delay"], r.get("area"), r.get("delay"), r.get("power_mw")]
                )
    print(f"Wrote {csv_path}", flush=True)

    plot_curves(results, out_dir, annotate=annotate)


def _good(results, design):
    return [
        r
        for r in results[design]
        if r.get("area") is not None
        and r.get("delay") is not None
        and r.get("power_mw") is not None
    ]


def plot_curves(results, out_dir, annotate=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("area", "DC Area (μm²)", "delay_vs_area.png", "Delay vs Area"),
        ("power_mw", "DC Power (mW)", "delay_vs_power.png", "Delay vs Power"),
    ]
    written = []
    for xkey, xlabel, fname, title in specs:
        fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
        for design in DESIGNS:
            pts = sorted(_good(results, design), key=lambda r: r[xkey])
            if not pts:
                continue
            xs = [p[xkey] for p in pts]
            ys = [p["delay"] for p in pts]
            ax.plot(
                xs,
                ys,
                marker=MARKERS[design],
                color=COLORS[design],
                linewidth=2.0,
                markersize=6.5,
                label=LABELS[design],
            )
            if annotate:
                for p in pts:
                    ax.annotate(
                        f"{p['target_delay']:.1f}",
                        (p[xkey], p["delay"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                    )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("DC Delay (ns)", fontsize=11)
        ax.set_title(f"{title}  (DC timing-constraint sweep, 16-bit)", fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
        print(f"Wrote plot: {path}", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", default="outputs/area_budget_sweep/20260613_163000"
    )
    parser.add_argument(
        "--delays", default=None, help="space/comma separated target delays (ns)"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--run_timeout",
        type=int,
        default=None,
        help=(
            "Override run_power_sweep.EDA_RUN_TIMEOUT (seconds) for this run. "
            "Tight constraints (e.g. 0.8ns) can exceed the default 1200s."
        ),
    )
    parser.add_argument("--bit_width", type=int, default=16)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--annotate", action="store_true", default=False)
    parser.add_argument(
        "--retry_failed",
        default=None,
        help=(
            "Path to an existing dc_timing_sweep.json. Reruns only the failed "
            "points (area/delay/power == null), merges them back, and rewrites "
            "the JSON/CSV/plots in place."
        ),
    )
    args = parser.parse_args()

    if args.run_timeout is not None:
        run_power_sweep.EDA_RUN_TIMEOUT = args.run_timeout
        print(f"[override] EDA_RUN_TIMEOUT = {args.run_timeout}s", flush=True)

    run_dir = (
        args.run_dir
        if os.path.isabs(args.run_dir)
        else os.path.join(_REPO_ROOT, args.run_dir)
    )

    # ── Retry mode: only rerun failed points from an existing sweep JSON ──────
    if args.retry_failed:
        json_path = (
            args.retry_failed
            if os.path.isabs(args.retry_failed)
            else os.path.join(_REPO_ROOT, args.retry_failed)
        )
        with open(json_path) as f:
            data = json.load(f)
        results = data["results"]
        delays = data.get("delays")
        prev_run_dir = data.get("run_dir", run_dir)
        paths = data.get("paths", {})
        out_dir = os.path.dirname(json_path)
        contents, _paths = load_designs(prev_run_dir)

        failed_jobs = [
            (design, entry["target_delay"])
            for design in DESIGNS
            for entry in results.get(design, [])
            if _is_failed(entry)
        ]
        print("=" * 70)
        print("DC timing-constraint PPA sweep -- RETRY FAILED")
        print(f"  json    : {json_path}")
        print(f"  workers : {args.workers}")
        print(f"  failed  : {len(failed_jobs)} -> {failed_jobs}")
        print("=" * 70, flush=True)
        if not failed_jobs:
            print("Nothing to retry; all points already succeeded.")
            return

        flat = run_jobs(failed_jobs, contents, args.workers, args.bit_width)
        for design, td, entry in flat:
            lst = results[design]
            for i, e in enumerate(lst):
                if abs(float(e["target_delay"]) - float(td)) < 1e-9:
                    lst[i] = entry
                    break
            else:
                lst.append(entry)
        for design in DESIGNS:
            results[design].sort(key=lambda r: r["target_delay"])

        save_artifacts(results, out_dir, prev_run_dir, delays, paths, args.annotate)
        print("\nSummary (good points / total):")
        for design in DESIGNS:
            print(f"  {design:5s}: {len(_good(results, design))}/{len(results[design])}")
        print(f"\nAll artifacts in: {out_dir}", flush=True)
        return

    # ── Normal full sweep ────────────────────────────────────────────────────
    if args.delays:
        delays = [
            float(x) for x in args.delays.replace(",", " ").split() if x.strip()
        ]
    else:
        delays = DEFAULT_DELAYS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(run_dir, f"dc_timing_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    contents, paths = load_designs(run_dir)

    print("=" * 70)
    print(f"DC timing-constraint PPA sweep")
    print(f"  run_dir : {run_dir}")
    print(f"  designs : {DESIGNS}")
    print(f"  delays  : {delays}")
    print(f"  workers : {args.workers}")
    print(f"  total   : {len(DESIGNS) * len(delays)} DC runs")
    print(f"  out_dir : {out_dir}")
    print("=" * 70, flush=True)

    results = run_sweep(contents, delays, args.workers, args.bit_width)

    save_artifacts(results, out_dir, run_dir, delays, paths, args.annotate)

    # console summary
    print("\nSummary (good points / total):")
    for design in DESIGNS:
        print(f"  {design:5s}: {len(_good(results, design))}/{len(results[design])}")
    print(f"\nAll artifacts in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
