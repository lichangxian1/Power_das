#!/usr/bin/env python3
"""把 21 设计的真实 DC+XA 结果透视成 exact/GA/greedy 对比,并和 standalone 代理并排。

回答:standalone 代理面积节省(Σ单cell DC 面积差)在整网表综合后剩多少真实面积/功耗。
用法: python analyze_dcxa.py <reeval_xa.csv> <cellsolver_batch 目录(含 summary.csv/grad_summary.json)>
"""
import csv
import json
import os
import sys

DCXA = sys.argv[1]
BATCH = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(DCXA)

# 真实 DC+XA
real = {}
for r in csv.DictReader(open(DCXA)):
    d = r["design"]  # k12_greedy
    kk, v = d.rsplit("_", 1)
    k = int(kk[1:])
    real.setdefault(k, {})[v] = {
        "area": float(r["area_dc"]) if r["area_dc"] else None,
        "power": float(r["power_xa_mw"]) if r["power_xa_mw"] else None,
        "delay": float(r["delay"]) if r["delay"] else None,
        "ok": r["success"] == "True",
    }

# standalone 代理（summary.csv：ga_save/greedy_save）
proxy = {}
sp = os.path.join(BATCH, "summary.csv")
if os.path.exists(sp):
    for r in csv.DictReader(open(sp)):
        proxy[int(r["k"])] = {"ga": float(r["ga_save"]),
                              "greedy": float(r["greedy_save"]),
                              "ga_n": int(r["ga_n"]),
                              "greedy_n": int(r["greedy_n"])}

ks = sorted(real)
print("=" * 108)
print("真实 DC+XA vs standalone 代理（cell 数 / 真实面积µm² / 真实功耗mW；Δ 相对同 k exact）")
print("=" * 108)
hdr = (f"{'k':>3} {'变体':>7} {'ncell':>6} {'area_dc':>9} {'Δarea':>8} {'Δa%':>7} "
       f"{'pow_xa':>8} {'Δpow%':>7} {'delay':>7} | {'代理省µm²':>9} {'真实省µm²':>9} {'代理/真实':>8}")
print(hdr)
rows_out = []
for k in ks:
    ex = real[k].get("exact", {})
    ea = ex.get("area")
    ep = ex.get("power")
    for v in ("exact", "ga", "greedy"):
        rv = real[k].get(v)
        if not rv or rv["area"] is None:
            continue
        da = rv["area"] - ea if ea else 0
        dap = da / ea * 100 if ea else 0
        dpp = (rv["power"] - ep) / ep * 100 if ep and rv["power"] else 0
        prox = proxy.get(k, {}).get(v, 0.0) if v != "exact" else 0.0
        real_sv = ea - rv["area"] if ea else 0.0  # 相对 exact 真实省的面积
        ratio = (real_sv / prox) if prox > 0 else float("nan")
        print(f"{k:>3} {v:>7} {rv.get('area') and (proxy.get(k,{}).get(v+'_n') if v!='exact' else 0) or 0:>6} "
              f"{rv['area']:>9.1f} {da:>8.1f} {dap:>6.1f}% {rv['power']:>8.4f} {dpp:>6.1f}% "
              f"{rv['delay']:>7.2f} | {prox:>9.1f} {real_sv:>9.1f} "
              f"{('—' if v=='exact' else (f'{ratio:.0%}' if prox>0 else 'n/a')):>8}")
        rows_out.append(dict(k=k, variant=v, ncell=proxy.get(k,{}).get(v+'_n',0) if v!='exact' else 0,
                             area_dc=rv["area"], d_area=da, d_area_pct=dap,
                             power_xa=rv["power"], d_power_pct=dpp, delay=rv["delay"],
                             proxy_save=prox, real_save=real_sv,
                             proxy_over_real=None if prox<=0 else real_sv/prox))
    print("-" * 108)

# 汇总:greedy 相对 GA 的真实增量
print("\n关键对比:greedy vs GA(真实,相对同 k exact 归一)")
print(f"{'k':>3} {'GA真实省µm²':>11} {'greedy真实省µm²':>15} {'倍数':>6} {'greedyΔpow%':>12} {'GAΔpow%':>9}")
for k in ks:
    ex = real[k].get("exact", {}); ea = ex.get("area"); ep = ex.get("power")
    ga = real[k].get("ga", {}); gr = real[k].get("greedy", {})
    if not (ea and ga.get("area") and gr.get("area")):
        continue
    ga_sv = ea - ga["area"]; gr_sv = ea - gr["area"]
    mult = gr_sv / ga_sv if ga_sv > 0 else float("inf")
    grp = (gr["power"]-ep)/ep*100 if ep else 0
    gap = (ga["power"]-ep)/ep*100 if ep else 0
    print(f"{k:>3} {ga_sv:>11.1f} {gr_sv:>15.1f} {mult:>5.1f}x {grp:>11.1f}% {gap:>8.1f}%")

json.dump(rows_out, open(os.path.join(BATCH, "dcxa_analysis.json"), "w"),
          indent=2, default=str)
print(f"\n-> {os.path.join(BATCH, 'dcxa_analysis.json')}")
