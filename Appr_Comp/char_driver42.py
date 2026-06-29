#!/usr/bin/env python3
"""DC characterize the generated 4:2 compressor library.

This is intentionally small and reuses char_driver.py for:
  - remote sandbox command construction
  - PPA report parsing
  - EDA connection environment variables

Inputs:
  Appr_Comp/rtl/comp42_lib.v
  Appr_Comp/rtl/manifest42.json

Output:
  Appr_Comp/library42_pair32.json
"""
import argparse
import json
import os
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import char_driver as cd  # noqa: E402


def write_module_list(manifest, limit=None):
    names = sorted(manifest)
    if limit:
        names = names[:limit]
    path = os.path.join(cd.RTL_DIR, "module_list42.txt")
    with open(path, "w") as f:
        f.write("\n".join(names) + "\n")
    return path, names


def rsync_files(remote, module_list_path):
    pairs = [
        (os.path.join(cd.RTL_DIR, "comp42_lib.v"), f"{remote}/src/rtl/comp42_lib.v"),
        (module_list_path, f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(cd.SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]
    return [
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -p {cd.EDA_PORT}",
            local,
            f"{cd.EDA_USER}@{cd.EDA_HOST}:{dest}",
        ]
        for local, dest in pairs
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="actually run remote DC")
    ap.add_argument("--limit", type=int, default=None, help="only characterize first N modules")
    ap.add_argument("--static-prob", type=float, default=0.25)
    ap.add_argument("--toggle-rate", type=float, default=0.125)
    ap.add_argument("--out", default=os.path.join(HERE, "library42_pair32.json"))
    ap.add_argument("--keep-remote", action="store_true")
    args = ap.parse_args()

    manifest_path = os.path.join(cd.RTL_DIR, "manifest42.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    module_list_path, names = write_module_list(manifest, args.limit)
    print(f"[char42] module_list42: {len(names)} modules -> {module_list_path}")

    uid = uuid.uuid4().hex[:6]
    remote = f"{cd.EDA_WORK_ROOT.rstrip('/')}/char42_{uid}"
    setup_cmd, run_cmd = cd.build_remote_cmds(
        remote, args.static_prob, args.toggle_rate, arcs=False
    )
    rsyncs = rsync_files(remote, module_list_path)

    if not args.run:
        print("\n===== DRY-RUN：以下命令将在 --run 时执行 =====\n")
        print(f"# setup\nssh -p {cd.EDA_PORT} {cd.EDA_USER}@{cd.EDA_HOST} {cd._q(setup_cmd)}\n")
        print("# rsync")
        for cmd in rsyncs:
            print("   " + " ".join(cd._q(x) for x in cmd))
        print(f"\n# run\nssh -p {cd.EDA_PORT} {cd.EDA_USER}@{cd.EDA_HOST} {cd._q(run_cmd)}")
        print(f"\n# cleanup\nssh -p {cd.EDA_PORT} {cd.EDA_USER}@{cd.EDA_HOST} {cd._q(f'rm -rf {remote}')}")
        return

    print(f"[char42] setup remote: {remote}")
    subprocess.run(
        ["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", setup_cmd],
        check=True,
        timeout=300,
    )
    print("[char42] rsync files ...")
    for cmd in rsyncs:
        subprocess.run(cmd, check=True, timeout=300)

    print(f"[char42] running dc_char.tcl (timeout {cd.RUN_TIMEOUT}s) ...")
    res = subprocess.run(
        ["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", run_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=cd.RUN_TIMEOUT,
    )
    stdout = res.stdout or ""
    ppa_text = stdout.split("====PPA====", 1)[-1]
    ppa = cd.parse_ppa(ppa_text)
    ok = sum(1 for v in ppa.values() if not v.get("error"))
    print(f"[char42] parsed PPA: {ok} ok / {len(ppa)} lines")
    if ok == 0:
        print("[char42] no valid PPA; last 60 log lines:")
        print("\n".join(stdout.splitlines()[-60:]))
        sys.exit(1)

    library = {}
    for name in names:
        entry = dict(manifest[name])
        entry.update(ppa.get(name, {"error": "no_ppa"}))
        library[name] = entry
    with open(args.out, "w") as f:
        json.dump(
            {
                "meta": {
                    "static_prob": args.static_prob,
                    "toggle_rate": args.toggle_rate,
                    "source_manifest": manifest_path,
                },
                "cells": library,
            },
            f,
            indent=2,
        )
    print(f"[char42] -> {args.out} ({len(library)} cells)")

    if not args.keep_remote:
        subprocess.run(
            ["ssh", "-p", cd.EDA_PORT, f"{cd.EDA_USER}@{cd.EDA_HOST}", f"rm -rf {cd._q(remote)}"],
            check=False,
            timeout=120,
        )


if __name__ == "__main__":
    main()
