import argparse
import csv
import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POWER_SOURCES = ("proxy", "eda")


def _fmt_budget(value):
    if value is None:
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value)


def _budget_from_dir(path):
    match = re.search(r"area_budget_(.+)$", path.name)
    if match is None:
        return None
    raw = match.group(1).replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _has_best_info(sweep_dir, source):
    source_dir = sweep_dir / f"power_source_{source}"
    return source_dir.is_dir() and any(source_dir.glob("area_budget_*/best_info.json"))


def _has_dc_comparison(sweep_dir, source):
    return (sweep_dir / f"power_source_{source}" / "dc_comparison.json").is_file()


def _find_latest_sweep(root, data_source):
    candidates = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if data_source in {"dc", "abc"}:
            has_data = all(_has_dc_comparison(path, source) for source in _POWER_SOURCES)
        else:
            has_data = all(_has_best_info(path, source) for source in _POWER_SOURCES)
        if has_data:
            candidates.append(path)
    if not candidates:
        expected = (
            "dc_comparison.json files"
            if data_source in {"dc", "abc"}
            else "best_info.json files"
        )
        raise FileNotFoundError(
            f"No sweep containing both proxy and eda {expected} found under {root}"
        )
    return candidates[-1]


def _read_power_mw(info, source, metric):
    if metric == "objective":
        return float(info["power"]) * 1000.0
    if metric == "eda":
        value = info.get("eda_power")
        if value is None and source == "eda":
            value = info.get("power")
        if value is None:
            return None
        return float(value) * 1000.0
    if metric == "proxy":
        value = info.get("proxy_power_mw")
        if value is None and source == "proxy":
            value = float(info["power"]) * 1000.0
        return None if value is None else float(value)
    raise ValueError(f"Unknown power metric: {metric}")


def _load_dc_or_abc_points(sweep_dir, source, data_source):
    comparison_path = sweep_dir / f"power_source_{source}" / "dc_comparison.json"
    if not comparison_path.exists():
        return []

    with comparison_path.open() as f:
        comparison = json.load(f)

    points = []
    for item in comparison.get("results", []):
        values = item.get(data_source, {})
        if data_source == "dc" and not values.get("success"):
            continue
        area = values.get("area")
        power_mw = values.get("power_mw")
        if area is None or power_mw is None:
            continue
        points.append(
            {
                "source": source,
                "budget": item.get("budget"),
                "area": float(area),
                "power_mw": float(power_mw),
                "delay": values.get("delay_ns", values.get("delay")),
                "path": str(comparison_path),
            }
        )
    return points


def _load_best_info_points(sweep_dir, source, metric):
    source_dir = sweep_dir / f"power_source_{source}"
    if not source_dir.is_dir():
        return []

    points = []
    for budget_dir in sorted(source_dir.glob("area_budget_*")):
        info_path = budget_dir / "best_info.json"
        if not info_path.exists():
            continue
        with info_path.open() as f:
            info = json.load(f)
        power_mw = _read_power_mw(info, source, metric)
        if power_mw is None:
            continue
        points.append(
            {
                "source": source,
                "budget": info.get("area_budget", _budget_from_dir(budget_dir)),
                "area": float(info["area"]),
                "power_mw": power_mw,
                "delay": info.get("delay"),
                "path": str(info_path),
            }
        )
    return points


def _load_points(sweep_dir, source, data_source, power_metric):
    if data_source in {"dc", "abc"}:
        return _load_dc_or_abc_points(sweep_dir, source, data_source)
    return _load_best_info_points(sweep_dir, source, power_metric)


def _write_csv(csv_path, points_by_source):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "budget", "area", "power_mw", "delay", "path"],
        )
        writer.writeheader()
        for source in _POWER_SOURCES:
            for point in points_by_source[source]:
                writer.writerow(point)


_LABELS = {
    "proxy": "Proxy-guided",
    "eda":   "EDA-guided",
}


def _plot(points_by_source, output_path, title, y_label, annotate, sort_by, scatter_only=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors  = {"proxy": "#1f77b4", "eda": "#d62728"}
    markers = {"proxy": "o",       "eda": "s"}

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=160)
    for source in _POWER_SOURCES:
        points = list(points_by_source[source])
        # sort by budget so connected lines follow the design intent progression
        points = sorted(points, key=lambda item: (item["budget"] is None, item["budget"]))

        if not points:
            continue
        x_vals = [point["area"]     for point in points]
        y_vals = [point["power_mw"] for point in points]

        if scatter_only:
            ax.scatter(
                x_vals, y_vals,
                marker=markers[source], s=55,
                color=colors[source], zorder=3,
                label=_LABELS[source],
            )
        else:
            ax.plot(
                x_vals, y_vals,
                marker=markers[source],
                linewidth=2.0, markersize=6.5,
                color=colors[source],
                label=_LABELS[source],
            )

        if annotate:
            for point in points:
                ax.annotate(
                    _fmt_budget(point["budget"]),
                    (point["area"], point["power_mw"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("DC Area (μm²)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot area-power curves for proxy and eda sweep results."
    )
    parser.add_argument(
        "--sweep_dir",
        default=None,
        help=(
            "Sweep directory containing power_source_proxy/ and power_source_eda/. "
            "Defaults to the latest matching directory under outputs/area_budget_sweep."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <sweep_dir>/dc_area_power_curve.png.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path. Defaults to <output>.csv when --write_csv is set.",
    )
    parser.add_argument(
        "--write_csv",
        action="store_true",
        help="Also write the plotted points to CSV.",
    )
    parser.add_argument(
        "--data_source",
        choices=["dc", "abc", "best_info"],
        default="dc",
        help=(
            "Which results to plot. Defaults to 'dc', using final DC area and "
            "power from dc_comparison.json."
        ),
    )
    parser.add_argument(
        "--power_metric",
        choices=["objective", "eda", "proxy"],
        default="eda",
        help=(
            "Power value to plot when --data_source best_info is used. "
            "'objective' uses best_info['power']; 'eda' uses eda_power; "
            "'proxy' uses proxy_power_mw when available."
        ),
    )
    parser.add_argument(
        "--sort_by",
        choices=["area", "budget"],
        default="budget",
        help="Point order used to connect each curve (default: budget).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Annotate each point with its area budget (default: on).",
    )
    parser.add_argument(
        "--no_annotate",
        dest="annotate",
        action="store_false",
        help="Disable point annotations.",
    )
    parser.add_argument(
        "--scatter_only",
        action="store_true",
        help="Plot as scatter (no connecting lines).",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    sweep_root = _REPO_ROOT / "outputs" / "area_budget_sweep"
    sweep_dir = (
        Path(args.sweep_dir).resolve()
        if args.sweep_dir
        else _find_latest_sweep(sweep_root, args.data_source)
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else sweep_dir / (
            "dc_area_power_curve.png"
            if args.data_source == "dc"
            else f"{args.data_source}_area_power_curve.png"
        )
    )
    source_name = {
        "dc": "DC",
        "abc": "ABC",
        "best_info": "Training",
    }[args.data_source]
    title = args.title or f"{source_name} Area–Power Sweep  ({sweep_dir.name})"
    y_label = f"{source_name} Power (mW)"

    points_by_source = {
        source: _load_points(sweep_dir, source, args.data_source, args.power_metric)
        for source in _POWER_SOURCES
    }
    missing = [source for source, points in points_by_source.items() if not points]
    if missing:
        raise RuntimeError(
            f"No plottable points for {', '.join(missing)} in {sweep_dir}"
        )

    _plot(
        points_by_source,
        output_path,
        title,
        y_label,
        args.annotate,
        args.sort_by,
        scatter_only=args.scatter_only,
    )
    if args.write_csv:
        csv_path = (
            Path(args.csv).resolve()
            if args.csv
            else output_path.with_suffix(output_path.suffix + ".csv")
        )
        _write_csv(csv_path, points_by_source)
        print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
