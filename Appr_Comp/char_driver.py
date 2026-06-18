#!/usr/bin/env python3
"""阶段 1（后半）：批量 DC 表征驱动 —— 把整库一次性推到远端 sandbox 跑 dc_char.tcl。

仿照 run_power_sweep.py 的 ssh/rsync 模式，但是「一次性单 job」而非 ~700 个任务。

安全默认 dry-run：
    python Appr_Comp/char_driver.py            # 只生成 module_list.txt 并打印将执行的远端命令
    python Appr_Comp/char_driver.py --run      # 真正连远端执行（请在错峰时段）

产物（--run 后）：
    Appr_Comp/library.json   —— manifest 误差指标 + DC 实测 area/power/delay_arcs，按 module 名索引

环境变量（与 run_power_sweep.py 同名，可覆盖）：
    EDA_USER EDA_HOST EDA_PORT EDA_BASE_DIR EDA_WORK_ROOT
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")
SCRIPTS_DIR = os.path.join(HERE, "scripts")

EDA_USER = os.environ.get("EDA_USER", "lchangxian")
EDA_HOST = os.environ.get("EDA_HOST", "202.120.39.27")
EDA_PORT = str(os.environ.get("EDA_PORT", "16822"))
EDA_BASE_DIR = os.environ.get("EDA_BASE_DIR", "/home/lchangxian/sandbox/sandbox_base")
EDA_WORK_ROOT = os.environ.get("EDA_WORK_ROOT", f"/home/{EDA_USER}/sandbox/sandbox_sets")
RUN_TIMEOUT = int(os.environ.get("EDA_CHAR_TIMEOUT", "10800"))

# report_power 单位 -> W 的换算
_PUNIT = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12, "NA": None}


def _q(v):
    return shlex.quote(str(v))


def select_modules(man, limit):
    """选 module 名。limit=None -> 全部；否则取「多样化」子集用于流程验证：

    保证含两个 exact（3:2 / 2:2，6 个 arc 全为实数）+ 3:2/2:2 交替（2:2 触发 nan-arc 路径）。
    """
    if not limit or limit >= len(man):
        return sorted(man.keys())
    import itertools
    exact = sorted(k for k, v in man.items() if v["is_exact"])
    picked = list(exact)
    rest = [k for k in sorted(man) if k not in set(picked)]
    t32 = [k for k in rest if man[k]["type"] == "32"]
    t22 = [k for k in rest if man[k]["type"] == "22"]
    interleaved = [x for pair in itertools.zip_longest(t32, t22) for x in pair if x]
    for k in interleaved:
        if len(picked) >= limit:
            break
        picked.append(k)
    return picked[:limit]


def write_module_list(limit=None, only=None):
    """从 manifest.json 生成 module_list.txt（一行一个 module 名）。

    only: 显式 module 名列表（优先于 limit），用于只表征选中的少数 cell。
    """
    with open(os.path.join(RTL_DIR, "manifest.json")) as f:
        man = json.load(f)
    if only:
        names = [m for m in only if m in man]
    else:
        names = select_modules(man, limit)
    path = os.path.join(RTL_DIR, "module_list.txt")
    with open(path, "w") as f:
        f.write("\n".join(names) + "\n")
    return path, names, man


def _norm_power(val, unit):
    if val in (None, "nan") or unit not in _PUNIT or _PUNIT[unit] is None:
        return None
    try:
        return float(val) * _PUNIT[unit]
    except ValueError:
        return None


def _f(x):
    if x in (None, "nan", "NA"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def parse_ppa(report_text):
    """解析 ppa_char.rpt 文本 -> {mod: {area, dyn_w, leak_w, delay_arcs{...}}}。"""
    out = {}
    for line in report_text.splitlines():
        line = line.strip()
        if not line.startswith("PPA mod="):
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        mod = kv.get("mod")
        if mod is None:
            continue
        if "ERROR" in kv:
            out[mod] = {"error": True}
            continue
        out[mod] = {
            "area": _f(kv.get("area")),
            "dyn_w": _norm_power(kv.get("dyn"), kv.get("dyn_u", "NA")),
            "leak_w": _norm_power(kv.get("leak"), kv.get("leak_u", "NA")),
            "tmax": _f(kv.get("tmax")),
            "delay_arcs": {
                "Tas": _f(kv.get("Tas")), "Tac": _f(kv.get("Tac")),
                "Tbs": _f(kv.get("Tbs")), "Tbc": _f(kv.get("Tbc")),
                "Tcs": _f(kv.get("Tcs")), "Tcc": _f(kv.get("Tcc")),
            },
        }
    return out


def build_remote_cmds(remote, sp, tr, arcs):
    """返回 (setup_cmd, run_cmd) 两个 ssh 命令列表。"""
    q_base, q_root, q_remote = _q(EDA_BASE_DIR), _q(EDA_WORK_ROOT), _q(remote)
    setup = (
        f"mkdir -p {q_root} && rm -rf {q_remote} && cp -r {q_base} {q_remote} && "
        f"rm -rf {q_remote}/results/* {q_remote}/reports/* {q_remote}/src/rtl/* || true"
    )
    run = (
        f"cd {q_remote} && LC_ALL=C LANG=C "
        f"dc_shell -x {_q(f'set my_sp {sp}; set my_tr {tr}; set my_arcs {1 if arcs else 0}')} "
        f"-f scripts/dc_char.tcl > dc_char.log 2>&1; echo '====PPA===='; cat reports/ppa_char.rpt"
    )
    return setup, run


def rsync_files(remote):
    pairs = [
        (os.path.join(RTL_DIR, "comp32_lib.v"), f"{remote}/src/rtl/comp32_lib.v"),
        (os.path.join(RTL_DIR, "comp22_lib.v"), f"{remote}/src/rtl/comp22_lib.v"),
        (os.path.join(RTL_DIR, "module_list.txt"), f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]
    cmds = []
    for local, dest in pairs:
        cmds.append([
            "rsync", "-az", "-e", f"ssh -p {EDA_PORT}",
            local, f"{EDA_USER}@{EDA_HOST}:{dest}",
        ])
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="真正连远端执行（默认 dry-run，仅准备文件并打印命令）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只表征前 N 个（多样化采样）用于验证流程；省略=全量 696")
    ap.add_argument("--static-prob", type=float, default=0.25, help="输入 P(1)，默认 0.25")
    ap.add_argument("--toggle-rate", type=float, default=0.125, help="输入翻转率，默认 0.125")
    ap.add_argument("--out", default=None,
                    help="输出 JSON（默认 library.json；--limit 时默认 library_sample.json）")
    ap.add_argument("--arcs", action="store_true",
                    help="额外算 6 个 input->output arc 延迟（慢，仅建议用于已选中的少数 cell）")
    ap.add_argument("--selected", default=None,
                    help="只表征该 JSON(selected_compressors.json)里的 module（配 --arcs 用）")
    ap.add_argument("--incremental", action="store_true",
                    help="只表征 manifest 中 library.json 还没有的 module，结果并入 library.json")
    ap.add_argument("--keep-remote", action="store_true", help="跑完不清理远端 sandbox（排错用）")
    args = ap.parse_args()

    if args.out is None:
        if args.incremental:
            args.out = os.path.join(HERE, "library.json")
        elif args.selected:
            args.out = os.path.join(HERE, "library_selected_arcs.json")
        else:
            args.out = os.path.join(HERE, "library_sample.json" if args.limit else "library.json")

    existing = {}
    only = None
    if args.incremental:
        if os.path.exists(args.out):
            existing = json.load(open(args.out)).get("cells", {})
        man_all = json.load(open(os.path.join(RTL_DIR, "manifest.json")))
        only = [m for m in sorted(man_all) if m not in existing]
        print(f"[char] incremental: manifest={len(man_all)} existing={len(existing)} -> 待补 {len(only)}")
    elif args.selected:
        sel = json.load(open(args.selected))["selected"]
        only = sorted({c["name"] for c in sel.values()})

    list_path, names, man = write_module_list(args.limit, only)
    print(f"[char] module_list.txt: {len(names)} modules -> {list_path}"
          + (f"  (LIMIT={args.limit}, 验证模式)" if args.limit else "")
          + ("  (INCREMENTAL)" if args.incremental else "")
          + ("  (SELECTED, arcs)" if args.selected and not args.incremental else ""))
    if args.selected and not args.incremental:
        print("[char] 模块: " + ", ".join(names))

    uid = uuid.uuid4().hex[:6]
    remote = f"{EDA_WORK_ROOT.rstrip('/')}/char_{uid}"
    setup_cmd, run_cmd = build_remote_cmds(remote, args.static_prob, args.toggle_rate, args.arcs)
    rsyncs = rsync_files(remote)

    if not args.run:
        print("\n===== DRY-RUN：以下命令将在 --run 时执行（错峰时段再跑）=====\n")
        print(f"# 1) setup remote sandbox\nssh -p {EDA_PORT} {EDA_USER}@{EDA_HOST} {_q(setup_cmd)}\n")
        print("# 2) rsync 库文件 + tcl")
        for c in rsyncs:
            print("   " + " ".join(_q(x) for x in c))
        print(f"\n# 3) 运行 dc_char.tcl（单 session，约 20-40 min）\nssh -p {EDA_PORT} {EDA_USER}@{EDA_HOST} {_q(run_cmd)}")
        print(f"\n# 4) 清理\nssh -p {EDA_PORT} {EDA_USER}@{EDA_HOST} {_q(f'rm -rf {remote}')}")
        print("\n确认无误后加 --run 真正执行。")
        return

    # ---- 真正执行 ----
    print(f"[char] setup remote: {remote}")
    subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", setup_cmd],
                   check=True, timeout=300)
    print("[char] rsync files ...")
    for c in rsyncs:
        subprocess.run(c, check=True, timeout=300)
    print(f"[char] running dc_char.tcl (timeout {RUN_TIMEOUT}s) ...")
    res = subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", run_cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, timeout=RUN_TIMEOUT)
    stdout = res.stdout or ""
    ppa_text = stdout.split("====PPA====", 1)[-1]
    ppa = parse_ppa(ppa_text)
    ok = sum(1 for v in ppa.values() if not v.get("error"))
    print(f"[char] parsed PPA: {ok} ok / {len(ppa)} lines")

    if ok == 0:
        print("[char] ⚠️ 无有效 PPA，保留远端供排错。最后 60 行日志：")
        print("\n".join(stdout.splitlines()[-60:]))
        sys.exit(1)

    # 合并 manifest(误差) + PPA(物理) -> library.json
    # incremental 时保留已有 cell，只追加本次新表征的
    library = dict(existing) if args.incremental else {}
    for mod in names:
        entry = dict(man[mod])
        entry.update(ppa.get(mod, {"error": "no_ppa"}))
        library[mod] = entry
    with open(args.out, "w") as f:
        json.dump({"meta": {"static_prob": args.static_prob,
                            "toggle_rate": args.toggle_rate}, "cells": library}, f, indent=2)
    print(f"[char] -> {args.out}  ({len(library)} cells)")

    if not args.keep_remote:
        subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"rm -rf {_q(remote)}"],
                       check=False, timeout=120)


if __name__ == "__main__":
    main()
