#!/usr/bin/env python3
"""Recover completed reeval_xa_glob results from its line-buffered log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RESULT_RE = re.compile(
    r"^(?P<design>\S+): success=(?P<success>True|False) "
    r"area=(?P<area>\S+) power_mw=(?P<power>\S+) delay=(?P<delay>\S+)\s*$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = {}
    for line in args.log.read_text(errors="replace").splitlines():
        match = RESULT_RE.match(line)
        if not match:
            continue
        values = match.groupdict()
        rows[values["design"]] = {
            "design": values["design"],
            "med": "",
            "area_dc": values["area"],
            "power_xa_mw": values["power"],
            "delay": values["delay"],
            "success": values["success"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["design", "med", "area_dc", "power_xa_mw", "delay", "success"]
        )
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    print(f"checkpointed {len(rows)} XA results -> {args.output}")


if __name__ == "__main__":
    main()
