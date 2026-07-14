#!/usr/bin/env python3
"""Freeze diverse live V5r2 RTL batches before the training build dir is overwritten.

The trainer emits 32 MUL-*.v files per episode and then logs ``processings: 32``.
This watcher captures one complete batch for selected Pareto-bin indices without
modifying or pausing training.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


EP_RE = re.compile(r"Episode\s+(\d+)\s+start")
BIN_RE = re.compile(r"\[v5\]\s+ep\s+(\d+)\s+bin=(\d+)/(\d+)")
PROCESS_RE = re.compile(r"processings:\s*32")


@dataclass
class Segment:
    name: str
    root: Path
    targets: set[int]
    log: Path
    offset: int = 0
    episode: int | None = None
    bin_idx: int | None = None
    bin_count: int | None = None


def latest_state(log: Path) -> tuple[int | None, int | None, int | None, bool]:
    episode = bin_idx = bin_count = None
    processed_for_current = False
    for line in log.read_text(errors="replace").splitlines():
        match = EP_RE.search(line)
        if match:
            episode = int(match.group(1))
            bin_idx = bin_count = None
            processed_for_current = False
        match = BIN_RE.search(line)
        if match:
            episode, bin_idx, bin_count = map(int, match.groups())
        if PROCESS_RE.search(line):
            processed_for_current = True
    return episode, bin_idx, bin_count, processed_for_current


def snapshot(segment: Segment, output: Path, captured: dict) -> bool:
    if segment.episode is None or segment.bin_idx not in segment.targets:
        return False
    key = f"{segment.name}:bin{segment.bin_idx}"
    if key in captured:
        return False
    sources = [segment.root / "build" / f"MUL-{idx}.v" for idx in range(32)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in sources):
        return False
    destination = output / "raw" / segment.name / f"bin{segment.bin_idx:02d}_ep{segment.episode:04d}"
    destination.mkdir(parents=True, exist_ok=False)
    for idx, source in enumerate(sources):
        shutil.copy2(source, destination / f"MUL-{idx:02d}.v")
    captured[key] = {
        "segment": segment.name,
        "episode": segment.episode,
        "bin": segment.bin_idx,
        "bin_count": segment.bin_count,
        "source": str(segment.root),
        "destination": str(destination),
        "n_rtls": len(sources),
        "captured_at_unix": time.time(),
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(captured, indent=2, sort_keys=True) + "\n"
    )
    print(f"captured {key} episode={segment.episode} -> {destination}", flush=True)
    return True


def consume_new_lines(segment: Segment, output: Path, captured: dict) -> None:
    size = segment.log.stat().st_size
    if size < segment.offset:
        segment.offset = 0
    with segment.log.open(errors="replace") as stream:
        stream.seek(segment.offset)
        for line in stream:
            match = EP_RE.search(line)
            if match:
                segment.episode = int(match.group(1))
                segment.bin_idx = segment.bin_count = None
            match = BIN_RE.search(line)
            if match:
                segment.episode, segment.bin_idx, segment.bin_count = map(int, match.groups())
            if PROCESS_RE.search(line):
                # All 32 files have been emitted before this line. A short grace
                # interval avoids racing a delayed filesystem metadata update.
                time.sleep(1.0)
                snapshot(segment, output, captured)
        segment.offset = stream.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/2026-07-13_v5r2_np2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/2026-07-14_xa64_v5r2"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    captured_path = args.output / "capture_manifest.json"
    captured = json.loads(captured_path.read_text()) if captured_path.exists() else {}
    segments = [
        Segment("seg_lo", args.run / "seg_lo", {0, 4, 8, 12}, args.run / "seg_lo.log"),
        Segment("seg_hi", args.run / "seg_hi", {0, 3, 6, 9}, args.run / "seg_hi.log"),
    ]
    for segment in segments:
        episode, bin_idx, bin_count, processed = latest_state(segment.log)
        segment.episode, segment.bin_idx, segment.bin_count = episode, bin_idx, bin_count
        segment.offset = segment.log.stat().st_size
        if processed:
            snapshot(segment, args.output, captured)

    expected = {"seg_lo:bin0", "seg_lo:bin4", "seg_lo:bin8", "seg_lo:bin12",
                "seg_hi:bin0", "seg_hi:bin3", "seg_hi:bin6", "seg_hi:bin9"}
    while not expected.issubset(captured):
        for segment in segments:
            consume_new_lines(segment, args.output, captured)
        missing = sorted(expected - set(captured))
        print(f"waiting; captured={len(expected) - len(missing)}/8 missing={missing}", flush=True)
        time.sleep(args.poll_seconds)
    print("capture complete: 8 strata, 256 RTLs", flush=True)


if __name__ == "__main__":
    main()
