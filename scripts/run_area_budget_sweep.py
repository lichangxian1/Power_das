import argparse
import concurrent.futures
import copy
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import trainer
from run_power_sweep import evaluate_single_routing

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_budgets(raw: str) -> List[float]:
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def _fmt_budget(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _dc_eval_all(entries, dc_target_delay, dc_workers):
    """并发提交所有 RTL 到远端 DC，返回 results list。"""
    results = [None] * len(entries)

    def _submit(i, entry):
        with open(entry["rtl_path"]) as f:
            content = f.read()
        logging.info("[DC] 提交 budget=%s ...", entry["budget"])
        res = evaluate_single_routing(
            idx=i,
            verilog_content=content,
            bit_width=16,
            target_delay=dc_target_delay,
        )
        return i, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=dc_workers) as ex:
        futs = {ex.submit(_submit, i, e): i for i, e in enumerate(entries)}
        for fut in concurrent.futures.as_completed(futs):
            i, res = fut.result()
            results[i] = res
            entry = entries[i]
            if res.get("success") and not res.get("logic_failed"):
                logging.info(
                    "[DC] ✓ budget=%s  area=%.2f  delay=%.4fns  power=%.4fmW",
                    entry["budget"], res["area"],
                    abs(res.get("delay", 0)), res.get("power_mw", float("inf")),
                )
            else:
                logging.warning("[DC] ✗ budget=%s  FAILED", entry["budget"])

    return results


def _print_dc_comparison(entries, dc_results, abc_infos, power_source, dc_target_delay):
    sep = "=" * 90
    logging.info(sep)
    logging.info("DC 综合对比结果  (power_source=%s  dc_target_delay=%.1fns)", power_source, dc_target_delay)
    logging.info(sep)
    header = (
        f"{'Budget':>8}  "
        f"{'abc area':>10}  {'abc delay':>10}  {'abc/proxy pwr':>14}  "
        f"{'DC area':>10}  {'DC delay':>10}  {'DC power':>10}  "
        f"{'Δarea':>8}  {'Δpower':>8}"
    )
    logging.info(header)
    logging.info("-" * 90)

    for entry, dc_res, abc_info in zip(entries, dc_results, abc_infos):
        budget = entry["budget"]
        a_area  = abc_info.get("area",  float("nan"))
        a_delay = abc_info.get("delay", float("nan"))
        a_power = abc_info.get("power", float("nan")) * 1000  # W → mW

        if dc_res and dc_res.get("success") and not dc_res.get("logic_failed"):
            d_area  = dc_res["area"]
            d_delay = abs(dc_res.get("delay", float("nan")))
            d_power = dc_res.get("power_mw", float("nan"))
            da_pct  = (d_area  - a_area)  / a_area  * 100 if a_area  else float("nan")
            dp_pct  = (d_power - a_power) / a_power * 100 if a_power else float("nan")
            logging.info(
                "%8s  %10.2f  %10.4fns  %14.4fmW  %10.2f  %10.4fns  %10.4fmW  %+7.1f%%  %+7.1f%%",
                int(budget) if float(budget).is_integer() else budget,
                a_area, a_delay, a_power,
                d_area, d_delay, d_power,
                da_pct, dp_pct,
            )
        else:
            logging.info("%8s  %10.2f  %10.4fns  %14.4fmW  %10s  %10s  %10s  %8s  %8s",
                         int(budget) if float(budget).is_integer() else budget,
                         a_area, a_delay, a_power,
                         "FAILED", "—", "—", "—", "—")

    logging.info(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-delay area-budget Arith-DAS sweeps."
    )
    parser.add_argument(
        "--config",
        default="configs/config_groups/mul_16_and.yaml",
        help="Base config_groups YAML path.",
    )
    parser.add_argument(
        "--budgets",
        default="980 1000 1020 1040 1060",
        help="Area budgets, separated by spaces or commas.",
    )
    parser.add_argument(
        "--power_source",
        choices=["proxy", "eda"],
        default="proxy",
        help="Power value used by the DAS objective.",
    )
    parser.add_argument("--fixed_target_delay", type=float, default=2.0)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--n_processing", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # DC evaluation options
    parser.add_argument("--dc_eval", action="store_true", default=True,
                        help="Send best RTL to remote DC after training (default: on).")
    parser.add_argument("--no_dc_eval", dest="dc_eval", action="store_false",
                        help="Skip DC evaluation after training.")
    parser.add_argument("--dc_target_delay", type=float, default=2.0,
                        help="Target delay (ns) for DC synthesis.")
    parser.add_argument("--dc_workers", type=int, default=4,
                        help="Concurrent DC workers.")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    exp_kwargs = cfg["experiment"]["kwargs"]
    train_kwargs_base = cfg["trainer"]["kwargs"]
    trainer_cls = getattr(trainer, cfg["trainer"]["name"])

    log_level = LOG_LEVELS.get(exp_kwargs.get("log_level", "INFO"), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    seed = args.seed if args.seed is not None else exp_kwargs.get("seed")
    set_seed(seed)

    budgets = _parse_budgets(args.budgets)
    if not budgets:
        raise ValueError("At least one area budget is required")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root is None:
        output_base = os.path.join(_REPO_ROOT, "outputs", "area_budget_sweep", timestamp)
    else:
        output_base = os.path.abspath(args.output_root)
    output_root = os.path.join(output_base, f"power_source_{args.power_source}")
    os.makedirs(output_root, exist_ok=True)

    logging.info("Area budgets: %s", budgets)
    logging.info("Power source: %s", args.power_source)
    logging.info("Output root: %s", output_root)

    # Collect exported RTL paths and abc best_info for DC comparison
    dc_entries  = []   # {budget, rtl_path}
    abc_infos   = []   # best_info summary per budget

    for budget in budgets:
        set_seed(seed)
        budget_tag = _fmt_budget(budget)
        run_dir    = os.path.join(output_root, f"area_budget_{budget_tag}")
        log_dir    = os.path.join(run_dir, "logs")
        build_dir  = os.path.join(run_dir, "build")
        export_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        train_kwargs = copy.deepcopy(train_kwargs_base)
        train_kwargs.update(copy.deepcopy(exp_kwargs))
        train_kwargs.update(
            {
                "area_budget": budget,
                "fixed_target_delay": args.fixed_target_delay,
                "power_source": args.power_source,
                "log_dir": log_dir,
                "build_dir": build_dir,
                "experiment_prefix": f"area_budget_{budget_tag}",
            }
        )
        if args.power_source == "eda":
            train_kwargs["use_power_proxy"] = False
        if args.num_episodes is not None:
            train_kwargs["num_episodes"] = args.num_episodes
        if args.num_samples is not None:
            train_kwargs["num_samples"] = args.num_samples
        if args.num_epochs is not None:
            train_kwargs["num_epochs"] = args.num_epochs
        if args.n_processing is not None:
            train_kwargs["n_processing"] = args.n_processing
        if seed is not None:
            train_kwargs["seed"] = seed

        logging.info("Starting budget=%s", budget)
        trainer_experiment = trainer_cls(**train_kwargs)
        trainer_experiment.run_experiment()
        rtl_path = trainer_experiment.export_best_candidate(export_dir)
        logging.info("Exported budget=%s RTL to %s", budget, rtl_path)

        # Collect abc metrics for comparison
        best_info_path = os.path.join(export_dir, "best_info.json")
        with open(best_info_path) as f:
            best_info = json.load(f)
        abc_infos.append(best_info)
        dc_entries.append({"budget": budget, "rtl_path": rtl_path})

    # ── DC 综合评估 ──────────────────────────────────────────────────────────
    if args.dc_eval:
        logging.info("所有 budget 训练完成，开始 DC 综合评估 (%d 个网表，%d workers) ...",
                     len(dc_entries), args.dc_workers)
        dc_results = _dc_eval_all(dc_entries, args.dc_target_delay, args.dc_workers)

        _print_dc_comparison(dc_entries, dc_results, abc_infos,
                             args.power_source, args.dc_target_delay)

        # 保存 DC 结果到 output_root
        dc_out = []
        for entry, dc_res, abc_info in zip(dc_entries, dc_results, abc_infos):
            dc_out.append({
                "budget":   entry["budget"],
                "rtl_path": entry["rtl_path"],
                "abc": {
                    "area":  abc_info.get("area"),
                    "delay": abc_info.get("delay"),
                    "power_mw": abc_info.get("power", 0) * 1000,
                    "area_feasible": abc_info.get("area_feasible"),
                },
                "dc": {
                    "area":     dc_res["area"]                    if dc_res and dc_res.get("success") else None,
                    "delay_ns": abs(dc_res.get("delay", 0))       if dc_res and dc_res.get("success") else None,
                    "power_mw": dc_res.get("power_mw")            if dc_res and dc_res.get("success") else None,
                    "success":  dc_res.get("success", False)      if dc_res else False,
                },
            })
        dc_json_path = os.path.join(output_root, "dc_comparison.json")
        with open(dc_json_path, "w") as f:
            json.dump({"power_source": args.power_source,
                       "dc_target_delay": args.dc_target_delay,
                       "results": dc_out}, f, indent=2)
        logging.info("DC 对比结果已保存: %s", dc_json_path)


if __name__ == "__main__":
    main()
