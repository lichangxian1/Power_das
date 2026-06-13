import argparse
import concurrent.futures
import copy
import json
import logging
import multiprocessing
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

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
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class ShanghaiFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, SHANGHAI_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


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


def _configure_logging(log_level: int, log_file: Optional[str] = None):
    formatter = ShanghaiFormatter(
        "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
    )
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root_logger.addHandler(handler)


def _parse_budgets(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).replace(",", " ").split() if x.strip()]


def _resolve_budgets(
    cli_budgets: Optional[str],
    train_kwargs_base: Dict[str, Any],
) -> List[float]:
    if cli_budgets is not None:
        return _parse_budgets(cli_budgets)
    return _parse_budgets(train_kwargs_base.get("area_budgets"))


def _parse_power_sources(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    power_sources = []
    for value in raw.replace(",", " ").split():
        value = value.strip()
        if not value:
            continue
        if value not in {"proxy", "eda"}:
            raise ValueError(
                f"Invalid power source {value!r}; expected 'proxy' or 'eda'"
            )
        if value not in power_sources:
            power_sources.append(value)
    return power_sources


def _fmt_budget(value) -> str:
    if value is None:
        return "unconstrained"
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _shanghai_timestamp() -> str:
    return datetime.now(SHANGHAI_TZ).strftime("%Y%m%d_%H%M%S")


def _sanitize_task_name(task_name: str) -> str:
    safe_chars = []
    for char in task_name.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("_")
    return safe_name or "area_budget_sweep"


def _build_train_kwargs(
    budget,
    exp_kwargs,
    train_kwargs_base,
    args_dict,
    output_root,
):
    budget_tag = _fmt_budget(budget)
    # budget is None -> native Arith-DAS unconstrained run; name dir "unconstrained".
    sub_name = "unconstrained" if budget is None else f"area_budget_{budget_tag}"
    run_dir = os.path.join(output_root, sub_name)
    trainer_log_root = args_dict.get("trainer_log_root")
    if trainer_log_root:
        log_dir = os.path.join(trainer_log_root, sub_name)
    else:
        log_dir = os.path.join(run_dir, "logs")
    build_dir = os.path.join(run_dir, "build")

    train_kwargs = copy.deepcopy(train_kwargs_base)
    train_kwargs.pop("area_budgets", None)
    train_kwargs.update(copy.deepcopy(exp_kwargs))
    train_kwargs.update(
        {
            "area_budget": budget,
            "fixed_target_delay": args_dict["fixed_target_delay"],
            "power_source": args_dict["power_source"],
            "log_dir": log_dir,
            "build_dir": build_dir,
            "experiment_prefix": (
                f"{args_dict['power_source']}_{sub_name}"
            ),
        }
    )

    device = args_dict.get("device")
    if device is not None:
        train_kwargs["device"] = device
    if args_dict["power_source"] == "eda":
        train_kwargs["use_power_proxy"] = False
    if args_dict["num_episodes"] is not None:
        train_kwargs["num_episodes"] = args_dict["num_episodes"]
    if args_dict["num_samples"] is not None:
        train_kwargs["num_samples"] = args_dict["num_samples"]
    if args_dict["num_epochs"] is not None:
        train_kwargs["num_epochs"] = args_dict["num_epochs"]
    if args_dict["n_processing"] is not None:
        train_kwargs["n_processing"] = args_dict["n_processing"]
    if args_dict["seed"] is not None:
        train_kwargs["seed"] = args_dict["seed"]

    return run_dir, train_kwargs


def _train_single_budget(job):
    _configure_logging(job["log_level"], job.get("log_file"))
    set_seed(job["seed"])

    budget = job["budget"]
    run_dir, train_kwargs = _build_train_kwargs(
        budget=budget,
        exp_kwargs=job["exp_kwargs"],
        train_kwargs_base=job["train_kwargs_base"],
        args_dict=job["args_dict"],
        output_root=job["output_root"],
    )
    os.makedirs(run_dir, exist_ok=True)

    logging.info(
        "Starting power_source=%s budget=%s",
        job["args_dict"]["power_source"],
        budget,
    )
    logging.info(
        "Training constraints: fixed_target_delay=%s area_budget=%s",
        train_kwargs["fixed_target_delay"],
        train_kwargs["area_budget"],
    )
    if job.get("log_file"):
        logging.info("Budget log file: %s", job["log_file"])
    trainer_cls = getattr(trainer, job["trainer_name"])
    trainer_experiment = None
    try:
        trainer_experiment = trainer_cls(**train_kwargs)
        trainer_experiment.run_experiment()
        rtl_path = trainer_experiment.export_best_candidate(run_dir)
    finally:
        if trainer_experiment is not None and getattr(
            trainer_experiment, "tb_logger", None
        ):
            trainer_experiment.tb_logger.close()

    best_info_path = os.path.join(run_dir, "best_info.json")
    with open(best_info_path) as f:
        best_info = json.load(f)

    logging.info(
        "Finished power_source=%s budget=%s, exported RTL to %s",
        job["args_dict"]["power_source"],
        budget,
        rtl_path,
    )
    return {
        "index": job["index"],
        "power_source": job["args_dict"]["power_source"],
        "budget": budget,
        "rtl_path": rtl_path,
        "best_info": best_info,
        "run_dir": run_dir,
    }


def _run_training_jobs(jobs, parallel_workers):
    results = [None] * len(jobs)
    if parallel_workers == 1:
        for job in jobs:
            result = _train_single_budget(job)
            results[result["index"]] = result
        return results

    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=parallel_workers,
        mp_context=mp_context,
    ) as executor:
        future_to_job = {
            executor.submit(_train_single_budget, job): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception:
                logging.exception("Training failed for budget=%s", job["budget"])
                raise
            results[result["index"]] = result
            logging.info(
                "Collected power_source=%s budget=%s result: %s",
                result["power_source"],
                result["budget"],
                result["rtl_path"],
            )
    return results


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
                _fmt_budget(budget),
                a_area, a_delay, a_power,
                d_area, d_delay, d_power,
                da_pct, dp_pct,
            )
        else:
            logging.info("%8s  %10.2f  %10.4fns  %14.4fmW  %10s  %10s  %10s  %8s  %8s",
                         _fmt_budget(budget),
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
        default=None,
        help=(
            "Area budgets, separated by spaces or commas. "
            "If omitted, trainer.kwargs.area_budgets from the config is used."
        ),
    )
    parser.add_argument(
        "--power_source",
        choices=["proxy", "eda"],
        default="proxy",
        help=(
            "Single power value used by the DAS objective. Used when "
            "--power_sources is not provided."
        ),
    )
    parser.add_argument(
        "--power_sources",
        default=None,
        help=(
            "Comma/space separated power sources to sweep, e.g. 'proxy,eda'. "
            "When provided, this overrides --power_source."
        ),
    )
    parser.add_argument(
        "--fixed_target_delay",
        type=float,
        default=None,
        help=(
            "Fixed ABC target delay. If omitted, "
            "trainer.kwargs.fixed_target_delay from the config is used."
        ),
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        default=False,
        help=(
            "Native Arith-DAS mode: weighted-sum objective with no area budget "
            "and no fixed target delay (full target-delay sweep). Runs exactly "
            "one job per power source. Use to compare proxy vs eda aligned with "
            "the original Arith-DAS baseline."
        ),
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--n_processing", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--task_name",
        default=None,
        help=(
            "Task name used in the run log folder. Defaults to "
            "experiment.kwargs.experiment_prefix."
        ),
    )
    parser.add_argument(
        "--log_root",
        default=None,
        help="Root directory for run log folders. Defaults to <repo>/logs.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=None,
        help=(
            "Number of area-budget trainings to run concurrently. "
            "Defaults to num_budgets * num_power_sources. "
            "Use 1 for serial training."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Optional comma/space separated devices assigned round-robin to "
            "parallel budget jobs, e.g. 'cuda:0,cuda:1'."
        ),
    )
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

    log_level = LOG_LEVELS.get(exp_kwargs.get("log_level", "INFO"), logging.INFO)

    seed = args.seed if args.seed is not None else exp_kwargs.get("seed")

    if args.unconstrained:
        # Native Arith-DAS: no area budget, no fixed delay -> weighted-sum
        # objective + full target-delay sweep. One job per power source.
        budgets = [None]
        fixed_target_delay = None
    else:
        budgets = _resolve_budgets(args.budgets, train_kwargs_base)
        if not budgets:
            raise ValueError("At least one area budget is required")
        fixed_target_delay = (
            args.fixed_target_delay
            if args.fixed_target_delay is not None
            else train_kwargs_base.get("fixed_target_delay", 2.0)
        )
    power_sources = _parse_power_sources(args.power_sources)
    if not power_sources:
        power_sources = [args.power_source]
    if args.devices:
        devices = [x for x in args.devices.replace(",", " ").split() if x.strip()]
        if not devices:
            raise ValueError("--devices was provided but no valid device was parsed")
    else:
        devices = []
    num_training_jobs = len(budgets) * len(power_sources)
    parallel_workers = (
        num_training_jobs if args.parallel_workers is None else args.parallel_workers
    )
    if parallel_workers < 1:
        raise ValueError("--parallel_workers must be >= 1")
    parallel_workers = min(parallel_workers, num_training_jobs)

    timestamp = _shanghai_timestamp()
    task_name = _sanitize_task_name(
        args.task_name
        or exp_kwargs.get("experiment_prefix")
        or cfg["trainer"]["name"]
    )
    if args.log_root is None:
        log_root = os.path.join(_REPO_ROOT, "logs")
    else:
        log_root = os.path.abspath(args.log_root)
    task_log_dir = os.path.join(log_root, f"{timestamp}_{task_name}")
    os.makedirs(task_log_dir, exist_ok=True)
    main_log_file = os.path.join(task_log_dir, "main.log")
    _configure_logging(log_level, main_log_file)

    if args.output_root is None:
        output_base = os.path.join(_REPO_ROOT, "outputs", "area_budget_sweep", timestamp)
    else:
        output_base = os.path.abspath(args.output_root)
    output_roots = {
        power_source: os.path.join(output_base, f"power_source_{power_source}")
        for power_source in power_sources
    }
    for output_root in output_roots.values():
        os.makedirs(output_root, exist_ok=True)

    logging.info("Area budgets: %s", budgets)
    logging.info("Fixed target delay: %s", fixed_target_delay)
    logging.info("Power sources: %s", power_sources)
    logging.info("Output base: %s", output_base)
    for power_source, output_root in output_roots.items():
        logging.info("Output root [%s]: %s", power_source, output_root)
    logging.info("Task log dir: %s", task_log_dir)
    logging.info("Main log file: %s", main_log_file)
    effective_n_processing = (
        args.n_processing
        if args.n_processing is not None
        else exp_kwargs.get("n_processing", train_kwargs_base.get("n_processing", 1))
    )
    logging.info(
        "Training parallel workers: %d jobs; each job n_processing=%s",
        parallel_workers,
        effective_n_processing,
    )
    try:
        total_synthesis_workers = parallel_workers * int(effective_n_processing)
    except (TypeError, ValueError):
        total_synthesis_workers = "unknown"
    logging.info("Approx total synthesis workers: %s", total_synthesis_workers)
    if devices:
        logging.info("Training devices: %s", devices)

    # Collect exported RTL paths and abc best_info for DC comparison
    args_base = {
        "fixed_target_delay": fixed_target_delay,
        "num_episodes": args.num_episodes,
        "num_samples": args.num_samples,
        "num_epochs": args.num_epochs,
        "n_processing": args.n_processing,
        "seed": seed,
    }
    jobs = []
    for power_source in power_sources:
        for budget in budgets:
            index = len(jobs)
            args_dict = dict(args_base)
            args_dict["power_source"] = power_source
            args_dict["trainer_log_root"] = os.path.join(
                task_log_dir, "tensorboard", f"power_source_{power_source}"
            )
            if devices:
                args_dict["device"] = devices[index % len(devices)]
            budget_tag = _fmt_budget(budget)
            jobs.append(
                {
                    "index": index,
                    "budget": budget,
                    "trainer_name": cfg["trainer"]["name"],
                    "exp_kwargs": exp_kwargs,
                    "train_kwargs_base": train_kwargs_base,
                    "args_dict": args_dict,
                    "output_root": output_roots[power_source],
                    "seed": seed,
                    "log_level": log_level,
                    "log_file": os.path.join(
                        task_log_dir,
                        f"{power_source}_"
                        + (
                            "unconstrained"
                            if budget is None
                            else f"area_budget_{budget_tag}"
                        )
                        + ".log",
                    ),
                }
            )

    train_results = _run_training_jobs(jobs, parallel_workers)

    # ── DC 综合评估 ──────────────────────────────────────────────────────────
    if args.dc_eval:
        logging.info(
            "所有训练完成，开始 DC 综合评估 (%d 个网表，%d workers) ...",
            len(train_results),
            args.dc_workers,
        )
        for power_source in power_sources:
            group_results = [
                result
                for result in train_results
                if result["power_source"] == power_source
            ]
            dc_entries = [
                {"budget": result["budget"], "rtl_path": result["rtl_path"]}
                for result in group_results
            ]
            abc_infos = [result["best_info"] for result in group_results]
            logging.info(
                "开始 DC 综合评估 power_source=%s (%d 个网表) ...",
                power_source,
                len(dc_entries),
            )
            dc_results = _dc_eval_all(
                dc_entries, args.dc_target_delay, args.dc_workers
            )

            _print_dc_comparison(
                dc_entries,
                dc_results,
                abc_infos,
                power_source,
                args.dc_target_delay,
            )

            # 保存 DC 结果到对应 power_source 的 output_root
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
            dc_json_path = os.path.join(
                output_roots[power_source], "dc_comparison.json"
            )
            with open(dc_json_path, "w") as f:
                json.dump({"power_source": power_source,
                           "dc_target_delay": args.dc_target_delay,
                           "results": dc_out}, f, indent=2)
            logging.info("DC 对比结果已保存: %s", dc_json_path)


if __name__ == "__main__":
    main()
