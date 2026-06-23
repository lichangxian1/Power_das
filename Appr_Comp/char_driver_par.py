#!/usr/bin/env python3
"""阶段 1（后半）并行版：把 module_list 切成 N 块，开 N 个并发 dc_shell sandbox 同时表征。

仅 DC 综合（无 VCS），可高并发。复用 char_driver 的远端命令与 PPA 解析；每个 worker 一个
独立 sandbox（charp_<uid>_<i>），跑同一个 scripts/dc_char.tcl，只是 module_list.txt 是自己那块。
全部跑完后把各块 PPA 合并 manifest -> library.json（保留已有 cell，追加新表征的）。

用法：
    python Appr_Comp/char_driver_par.py            # dry-run，打印切分与将执行的命令
    python Appr_Comp/char_driver_par.py --run -j 40 # 真跑，40 并发
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
import uuid

# sshd MaxStartups 会在并发 pre-auth 握手过多时丢连接（"Connection closed"）。
# 用 启动错峰 + 退避重试 规避，从而可保持高 -j 而不被踢。
_CONN_ERRS = ("Connection closed", "connection unexpectedly closed",
              "Connection refused", "code 255", "kex_exchange", "Connection reset")


def _attempt(cmd, timeout, tries=5):
    """跑 cmd（check=True）；遇到 ssh/rsync 连接类错误退避重试。"""
    last = None
    for t in range(tries):
        try:
            return subprocess.run(cmd, check=True, timeout=timeout,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            last = e
            se = (e.stderr or b"").decode("utf-8", "ignore")
            if any(k in se for k in _CONN_ERRS) and t < tries - 1:
                time.sleep(2 * (t + 1) + random.uniform(0, 3))
                continue
            raise
    raise last

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import char_driver as cd  # 复用 parse_ppa / build_remote_cmds / EDA_* / _q

RTL_DIR = cd.RTL_DIR
SCRIPTS_DIR = cd.SCRIPTS_DIR
CHUNK_DIR = os.path.join(RTL_DIR, "chunks")


def chunk(lst, n):
    """把 lst 尽量均匀分成 n 块（块大小相差 <=1）。"""
    n = max(1, min(n, len(lst)))
    k, m = divmod(len(lst), n)
    out, i = [], 0
    for x in range(n):
        sz = k + (1 if x < m else 0)
        out.append(lst[i:i + sz])
        i += sz
    return out


def run_worker(idx, mods, uid, sp, tr, stagger=0.0):
    """一个 worker：错峰 -> setup -> rsync(lib+本块module_list+tcl) -> dc_shell -> 解析 PPA。"""
    if stagger:
        time.sleep(idx * stagger)
    remote = f"{cd.EDA_WORK_ROOT.rstrip('/')}/charp_{uid}_{idx:03d}"
    ml_path = os.path.join(CHUNK_DIR, f"ml_{idx:03d}.txt")
    with open(ml_path, "w") as f:
        f.write("\n".join(mods) + "\n")
    setup_cmd, run_cmd = cd.build_remote_cmds(remote, sp, tr, arcs=False)
    ssh = ["ssh", "-p", cd.EDA_PORT, "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=10", f"{cd.EDA_USER}@{cd.EDA_HOST}"]
    pairs = [
        (os.path.join(RTL_DIR, "comp32_lib.v"), f"{remote}/src/rtl/comp32_lib.v"),
        (os.path.join(RTL_DIR, "comp22_lib.v"), f"{remote}/src/rtl/comp22_lib.v"),
        (ml_path, f"{remote}/src/rtl/module_list.txt"),
        (os.path.join(SCRIPTS_DIR, "dc_char.tcl"), f"{remote}/scripts/dc_char.tcl"),
    ]
    t0 = time.time()
    try:
        _attempt(ssh + [setup_cmd], 300)
        for local, dest in pairs:
            _attempt(["rsync", "-az", "-e", f"ssh -p {cd.EDA_PORT} -o BatchMode=yes",
                      local, f"{cd.EDA_USER}@{cd.EDA_HOST}:{dest}"], 300)
        res = subprocess.run(ssh + [run_cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=3600)
        ppa_text = (res.stdout or "").split("====PPA====", 1)[-1]
        ppa = cd.parse_ppa(ppa_text)
        # 清理本 worker 的 sandbox（自建自删）
        subprocess.run(ssh + [f"rm -rf {cd._q(remote)}"], check=False, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = sum(1 for v in ppa.values() if not v.get("error"))
        return idx, mods, ppa, ok, time.time() - t0, None
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", "ignore")[-300:] if e.stderr else str(e)
        return idx, mods, {}, 0, time.time() - t0, f"setup/rsync failed: {err}"
    except Exception as e:  # noqa: BLE001
        return idx, mods, {}, 0, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="真跑（默认 dry-run）")
    ap.add_argument("-j", "--jobs", type=int, default=40, help="并发 worker 数（默认 40）")
    ap.add_argument("--stagger", type=float, default=1.5,
                    help="每个 worker 启动错峰秒数 idx*stagger，避开 sshd MaxStartups 连接突发（默认 1.5）")
    ap.add_argument("--static-prob", type=float, default=0.25)
    ap.add_argument("--toggle-rate", type=float, default=0.125)
    ap.add_argument("--out", default=os.path.join(HERE, "library.json"))
    ap.add_argument("--only-missing", action="store_true",
                    help="只表征 manifest 中 library.json 还没有的 module（默认 True 行为）")
    args = ap.parse_args()

    man = json.load(open(os.path.join(RTL_DIR, "manifest.json")))
    existing = {}
    if os.path.exists(args.out):
        existing = json.load(open(args.out)).get("cells", {})
    todo = [m for m in sorted(man) if m not in existing]
    print(f"[par] manifest={len(man)} existing={len(existing)} -> 待表征 {len(todo)}")
    if not todo:
        print("[par] 没有待表征 module，退出。")
        return

    chunks = chunk(todo, args.jobs)
    sizes = [len(c) for c in chunks]
    print(f"[par] 切成 {len(chunks)} 块, 每块 {min(sizes)}~{max(sizes)} 个, 并发 {args.jobs}")

    if not args.run:
        uid = "DRYRUN"
        s, r = cd.build_remote_cmds(f"{cd.EDA_WORK_ROOT}/charp_{uid}_000",
                                    args.static_prob, args.toggle_rate, False)
        print("\n===== DRY-RUN：每个 worker 将执行（setup -> 4x rsync -> run）=====")
        print(f"# setup:\n  ssh ... {s[:160]}...")
        print(f"# run:\n  ssh ... {r[:160]}...")
        print(f"\n块大小: {sizes}\n确认后加 --run。")
        return

    os.makedirs(CHUNK_DIR, exist_ok=True)
    uid = uuid.uuid4().hex[:6]
    print(f"[par] uid={uid} 启动 {len(chunks)} workers ...")
    t0 = time.time()
    all_ppa, fails, done_ct = {}, [], 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_worker, i, c, uid, args.static_prob, args.toggle_rate,
                          args.stagger)
                for i, c in enumerate(chunks)]
        for fu in cf.as_completed(futs):
            idx, mods, ppa, ok, dt, err = fu.result()
            done_ct += 1
            if err:
                fails.append((idx, err))
                print(f"[par] worker {idx:03d} ✗ ({done_ct}/{len(chunks)}) {err}")
            else:
                all_ppa.update(ppa)
                print(f"[par] worker {idx:03d} ✓ ({done_ct}/{len(chunks)}) "
                      f"{ok}/{len(mods)} PPA in {dt:.0f}s")

    ok_total = sum(1 for v in all_ppa.values() if not v.get("error"))
    print(f"\n[par] 全部完成 {time.time()-t0:.0f}s: PPA ok={ok_total}, "
          f"解析到 {len(all_ppa)} 行, 失败 worker={len(fails)}")
    if fails:
        print("[par] ⚠️ 失败 worker:", fails)
    if ok_total == 0:
        print("[par] ⚠️ 无有效 PPA，未写 library.json。")
        sys.exit(1)

    # 合并：保留已有 cell，追加本次新表征的
    library = dict(existing)
    merged = 0
    for mod in todo:
        if mod in all_ppa and not all_ppa[mod].get("error"):
            entry = dict(man[mod])
            entry.update(all_ppa[mod])
            library[mod] = entry
            merged += 1
    miss = [m for m in todo if m not in all_ppa or all_ppa[m].get("error")]
    with open(args.out, "w") as f:
        json.dump({"meta": {"static_prob": args.static_prob,
                            "toggle_rate": args.toggle_rate,
                            "parallel_jobs": args.jobs}, "cells": library}, f, indent=2)
    print(f"[par] -> {args.out}  ({len(library)} cells, 本次并入 {merged})")
    if miss:
        print(f"[par] ⚠️ {len(miss)} 个 module 未拿到 PPA（如 {miss[:8]} ...）"
              f"可再跑一次 --run 补齐（incremental）。")


if __name__ == "__main__":
    main()
