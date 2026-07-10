#!/usr/bin/env python3
"""多架构 GA vs greedy 真实 DC+XA 裁决。

输入:检查点架构 reeval_xa.csv(k{kk}_i{it}_{variant})+ 可选 finals reeval_xa.csv
(k{kk}_{variant})。按同架构内 exact/GA/greedy 三元组配对,exact 面积去重(同结构),
统计 GA vs greedy 谁赢真实面积、谁赢真实功耗。
用法: python analyze_arch.py <arch reeval_xa.csv> [finals reeval_xa.csv]
"""
import csv
import sys


def load(path, is_final):
    """返回 {arch_id: {variant: {area,power,delay}}}."""
    archs = {}
    for r in csv.DictReader(open(path)):
        d = r["design"]
        parts = d.split("_")
        v = parts[-1]  # exact/ga/greedy
        arch = "_".join(parts[:-1])  # k12_i39 或 k12
        if not r["area_dc"]:
            continue
        archs.setdefault(arch, {})[v] = dict(
            area=float(r["area_dc"]),
            power=float(r["power_xa_mw"]) if r["power_xa_mw"] else None,
            delay=float(r["delay"]) if r["delay"] else None)
    return archs


def main():
    archs = load(sys.argv[1], False)
    if len(sys.argv) > 2:
        for a, v in load(sys.argv[2], True).items():
            archs[a] = v

    # 组三元组 + exact 面积去重（同结构 → 同 exact area，保留一个）
    rows = []
    for arch, vs in sorted(archs.items()):
        if not all(k in vs for k in ("exact", "ga", "greedy")):
            continue
        ea, ep = vs["exact"]["area"], vs["exact"]["power"]
        rows.append(dict(
            arch=arch, k=int(arch.split("_")[0][1:]), exact_area=ea, exact_pow=ep,
            ga_area=vs["ga"]["area"], ga_pow=vs["ga"]["power"],
            greedy_area=vs["greedy"]["area"], greedy_pow=vs["greedy"]["power"],
            ga_dsave=ea - vs["ga"]["area"], gr_dsave=ea - vs["greedy"]["area"],
            ga_dpow=(vs["ga"]["power"] - ep) / ep * 100,
            gr_dpow=(vs["greedy"]["power"] - ep) / ep * 100))
    # 去重:同 exact_area(四舍五入到 0.1)视为同结构
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda x: (x["k"], -x["gr_dsave"])):
        key = (r["k"], round(r["exact_area"], 1))
        if key in seen:
            continue
        seen.add(key); uniq.append(r)

    print("=" * 104)
    print(f"多架构 GA vs greedy(真实 DC+XA;去重后 {len(uniq)} 个独立架构;Δ=相对同架构 exact)")
    print("=" * 104)
    print(f"{'架构':>10}{'k':>3}{'GA省µm²':>9}{'greedy省µm²':>12}{'面积赢':>7}"
          f"{'GAΔpow':>8}{'grΔpow':>8}{'功耗赢':>7}")
    ga_area_win = gr_area_win = ga_pow_win = gr_pow_win = 0
    gr_pow_up = 0
    for r in uniq:
        aw = "greedy" if r["gr_dsave"] > r["ga_dsave"] + 0.5 else (
            "GA" if r["ga_dsave"] > r["gr_dsave"] + 0.5 else "平")
        pw = "greedy" if r["gr_dpow"] < r["ga_dpow"] - 0.3 else (
            "GA" if r["ga_dpow"] < r["gr_dpow"] - 0.3 else "平")
        ga_area_win += aw == "GA"; gr_area_win += aw == "greedy"
        ga_pow_win += pw == "GA"; gr_pow_win += pw == "greedy"
        gr_pow_up += r["gr_dpow"] > 0
        print(f"{r['arch']:>10}{r['k']:>3}{r['ga_dsave']:>9.1f}{r['gr_dsave']:>12.1f}"
              f"{aw:>7}{r['ga_dpow']:>7.1f}%{r['gr_dpow']:>7.1f}%{pw:>7}")
    n = len(uniq)
    print("-" * 104)
    print(f"面积:GA 赢 {ga_area_win}/{n}, greedy 赢 {gr_area_win}/{n}")
    print(f"功耗:GA 赢 {ga_pow_win}/{n}, greedy 赢 {gr_pow_win}/{n};greedy 反升功耗 {gr_pow_up}/{n}")
    # 按 k 分组的面积赢家
    print("\n按 k 分组(面积赢家):")
    for k in sorted(set(r["k"] for r in uniq)):
        sub = [r for r in uniq if r["k"] == k]
        gw = sum(r["gr_dsave"] > r["ga_dsave"] + 0.5 for r in sub)
        print(f"  k{k:02d}: {len(sub)}架构, greedy 面积赢 {gw}/{len(sub)}")


if __name__ == "__main__":
    main()
