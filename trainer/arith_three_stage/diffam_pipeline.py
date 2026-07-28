"""Dedicated-process proxy DiffAM producer overlapped with blocking DC batches."""
from __future__ import annotations

import argparse
import multiprocessing
import traceback
from typing import Iterable, Optional, Sequence

from .candidate import Candidate
from .diffam_proxy_search import DiffAMProxyStage2Search

def _build_worker_engine(config, run_dir: str):
    # Imported lazily so the spawned process completes module initialization
    # before train_three_stage imports the runner back through its public API.
    from scripts.train_three_stage import build_engine

    arguments = argparse.Namespace(
        config=str(config.engine_config_path),
        target_delay=float(config.delay_limit),
        error_vectors=int(config.error_vectors),
        out=run_dir,
        dc_batch=int(config.dc_batch_size),
        stage3_num_epochs=1,
        dc_parallelism=int(config.dc_parallelism),
        device="cpu",
        seed=int(config.seed),
        k_min=int(config.k_min),
        approx_col_window=int(config.approx_col_window),
        approx_lib_path=str(config.approx_lib_path),
        approx42_library_path=str(config.approx42_library_path),
        approx42_rtl_path=str(config.approx42_rtl_path),
        stage3_normalize_advantage=True,
        stage3_episodes_per_elite=5,
    )
    return build_engine(arguments)


def _producer_main(connection, config, run_dir, backbones_payload, state):
    try:
        engine = _build_worker_engine(config, run_dir)
        backbones = [
            Candidate.from_dict(payload) for payload in backbones_payload
        ]
        searcher = DiffAMProxyStage2Search(
            engine, config, run_dir, backbones
        )
        if state:
            searcher.load_state_dict(state)
        while True:
            message = connection.recv()
            command = message.get("command")
            if command == "close":
                connection.send({"ok": True})
                return
            if command != "propose":
                raise ValueError(f"unknown proxy DiffAM producer command {command!r}")
            observations = [
                Candidate.from_dict(payload)
                for payload in message.get("observations") or []
            ]
            if observations:
                searcher.observe_and_fit(observations)
            candidates = searcher.propose(
                backbones,
                size=int(message["size"]),
                round_index=int(message["round_index"]),
                excluded_hashes=message.get("excluded_hashes") or (),
                warm_starts=[
                    Candidate.from_dict(payload)
                    for payload in message.get("warm_starts") or []
                ],
            )
            connection.send(
                {
                    "ok": True,
                    "round_index": int(message["round_index"]),
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "search_state": searcher.state_dict(),
                }
            )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            raise


class DiffAMProxyProducer:
    """One outstanding proposal request, transported without queue threads."""

    def __init__(
        self,
        config,
        run_dir: str,
        backbones: Sequence[Candidate],
        state: Optional[dict] = None,
    ):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_producer_main,
            args=(
                child,
                config,
                run_dir,
                [candidate.to_dict() for candidate in backbones],
                state,
            ),
            name="stage2-diffam-proxy-producer",
            daemon=True,
        )
        self.process.start()
        child.close()
        self.pending_round = None

    def request(
        self,
        *,
        size: int,
        round_index: int,
        excluded_hashes: Iterable[str],
        warm_starts: Sequence[Candidate],
        observations: Sequence[Candidate],
    ) -> None:
        if self.pending_round is not None:
            raise RuntimeError(
                f"proxy DiffAM request {self.pending_round} is still outstanding"
            )
        self.connection.send(
            {
                "command": "propose",
                "size": int(size),
                "round_index": int(round_index),
                "excluded_hashes": list(excluded_hashes),
                "warm_starts": [
                    candidate.to_dict() for candidate in warm_starts
                ],
                "observations": [
                    candidate.to_dict() for candidate in observations
                ],
            }
        )
        self.pending_round = int(round_index)

    def receive(self) -> tuple[list[Candidate], dict]:
        if self.pending_round is None:
            raise RuntimeError("no outstanding proxy DiffAM request")
        message = self.connection.recv()
        expected = self.pending_round
        self.pending_round = None
        if not message.get("ok"):
            raise RuntimeError(
                "proxy DiffAM producer failed: "
                f"{message.get('error')}\n{message.get('traceback', '')}"
            )
        if int(message["round_index"]) != expected:
            raise RuntimeError(
                f"proxy DiffAM producer returned round {message['round_index']}, "
                f"expected {expected}"
            )
        return (
            [
                Candidate.from_dict(payload)
                for payload in message["candidates"]
            ],
            message["search_state"],
        )

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            if self.pending_round is not None:
                # A normal runner closes only after receiving the outstanding
                # request. On an exception, terminate instead of corrupting IPC.
                self.process.terminate()
            else:
                try:
                    self.connection.send({"command": "close"})
                    self.connection.recv()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
        self.process = None
