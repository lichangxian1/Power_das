#!/usr/bin/env python3
"""lr=0 对照裁决：冻结策略 vs PPO（其余逐参数相同），同 3 点双臂对比。
指标: ① found_best objective（终局/任意时刻）② 窗口均值漂移 ③ 记录事件时间分布
④ （若两臂都有 reeval_xa.csv）XA 实测 PPA。
用法: compare_lr0_control.py <ctrl_dir> <treat_dir>"""
import csv
import glob
import os
import re
import sys

CTRL = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/lee/Power_das/outputs/2026-07-09_23_mred_ctrl_lr0"
TREAT = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/lee/Power_das/outputs/2026-07-09_21_mred_warm240eg_np4"
PAT = re.compile(r"\[ep\s+(\d+)/\d+\]\s+obj=([\d.]+).*?\|\| best: obj=([\d.]+)")


def trace(path):
    eps, objs, bests = [], [], []
    for line in open(path, errors="ignore"):
        m = PAT.search(line)
        if m:
            eps.append(int(m.group(1)))
            objs.append(float(m.group(2)))
            bests.append(float(m.group(3)))
    return eps, objs, bests


def stats(objs, w=30):
    n = len(objs)
    if n < 2 * w:
        return None
    a = sum(objs[:w]) / w
    b = sum(objs[-w:]) / w
    best = float("inf")
    upd = []
    for i, o in enumerate(objs):
        if o < best - 1e-9:
            best = o
            upd.append(i)
    thirds = [sum(1 for i in upd if i < n / 3),
              sum(1 for i in upd if n / 3 <= i < 2 * n / 3),
              sum(1 for i in upd if i >= 2 * n / 3)]
    return a, b, 100 * (b - a) / a, thirds


def reeval(d):
    p = os.path.join(d, "reeval_xa.csv")
    if not os.path.exists(p):
        return {}
    return {r["design"]: (float(r["mred"] or 0), float(r["area_dc"]), float(r["power_xa_mw"]))
            for r in csv.DictReader(open(p)) if r.get("success") == "True"}


print(f"CTRL (lr=0):  {CTRL}\nTREAT (PPO):  {TREAT}\n")
print(f"{'tag':<16}{'ep(c/t)':>10} | {'best ctrl':>10}{'best ppo':>10}{'Δ%':>7} | "
      f"{'drift c%':>9}{'drift t%':>9} | records c | t")
deltas = []
for cf in sorted(glob.glob(f"{CTRL}/k*.log")):
    tag = os.path.basename(cf)[:-4]
    tf = f"{TREAT}/{tag}.log"
    if not os.path.exists(tf):
        continue
    ce, co, cb = trace(cf)
    te, to, tb = trace(tf)
    if not ce or not te:
        continue
    n = min(len(co), len(to))  # 对齐到共同进度再比（双臂并行、节奏不同）
    bc, bt = min(cb[:n]), min(tb[:n])
    d = 100 * (bc - bt) / bt
    deltas.append(d)
    sc, st = stats(co[:n]), stats(to[:n])
    fmt = lambda s: (f"{s[2]:>+9.1f}" if s else f"{'--':>9}")
    rec = lambda s: ("/".join(map(str, s[3])) if s else "--")
    print(f"{tag:<16}{len(co):>5}/{len(to):<4} | {bc:>10.4f}{bt:>10.4f}{d:>+7.2f} | "
          f"{fmt(sc)}{fmt(st)} | {rec(sc):>9} | {rec(st)}")

if deltas:
    m = sum(deltas) / len(deltas)
    print(f"\nmean Δbest (ctrl−ppo) = {m:+.2f}%   "
          f"(>+1% ⇒ PPO 有真贡献；|Δ|≤~0.5% ⇒ 冻结策略打平，RL 组件判定为装饰)")
rc, rt = reeval(CTRL), reeval(TREAT)
common = set(rc) & set(rt)
if common:
    print(f"\nXA reeval 对比:")
    print(f"{'tag':<16}{'mred c/t':>22}{'area c/t':>18}{'ΔA%':>7}{'pwr c/t':>16}{'ΔP%':>7}")
    for t in sorted(common):
        c, p = rc[t], rt[t]
        print(f"{t:<16}{c[0]:>11.3e}{p[0]:>11.3e}{c[1]:>9.1f}{p[1]:>9.1f}"
              f"{100*(c[1]-p[1])/p[1]:>+7.1f}{c[2]:>8.3f}{p[2]:>8.3f}{100*(c[2]-p[2])/p[2]:>+7.1f}")
