import argparse
import copy
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

    for budget in budgets:
        set_seed(seed)
        budget_tag = _fmt_budget(budget)
        run_dir = os.path.join(output_root, f"area_budget_{budget_tag}")
        log_dir = os.path.join(run_dir, "logs")
        build_dir = os.path.join(run_dir, "build")
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


if __name__ == "__main__":
    main()
