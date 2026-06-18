#!/usr/bin/env python3
"""把 dataset/*.pth checkpoint 按 suffix 映射到 outputs/ 对应任务文件夹。

dry-run: 只打印计划 + 写 manifest, 不动文件。
--apply: 实际移动, 并生成 undo 脚本。
"""
import argparse
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "dataset")
OUTPUTS = os.path.join(REPO, "outputs")

# (正则匹配 basename, 目标 outputs 子文件夹, 置信度)  顺序优先, 命中即停
RULES = [
    (r"dag_gnn_delay",                          "2026-06-04_gnn_delay_11k",                   "high"),
    (r"graphormer_onehot",                      "2026-06-02_graphormer_onehot",              "high"),
    (r"v2_16k_phys_E4_strong_scale10",          "2026-06-05_03_e4_phys_v2_16k_strong_scale10","high"),
    (r"v2_16k_phys_E4_seed",                    "2026-06-03_e4_phys_multi_seed",             "high"),
    (r"v2_16k_phys_E4(?!_strong|_seed)",        "2026-06-03_e4_phys_v2_16k",                 "high"),
    (r"C_v2_11k_phys_E4_modeC",                 "2026-06-04_00_e4_phys_modeC_11k",           "high"),
    (r"v2_11k_phys_E4_seed",                    "2026-06-03_e4_phys_multi_seed",             "med"),
    (r"v2_11k_phys_E4_strong_scale10",          "2026-06-05_02_e4_phys_v2_11k_strong_w_scale_high", "low"),
    (r"v2_11k_phys_E4_strong(?!_scale10)",      "2026-06-05_02_e4_phys_v2_11k_strong_rank",  "low"),
    (r"v2_11k_phys_E4(?!_seed|_strong)",        "2026-06-04_00_e4_phys_modeC_11k",           "low"),
    (r"v2_11k_e10_E4_honest",                   "2026-06-01_honest_eval_5fold",              "med"),
    (r"v2_11k_edge10_E4",                       "2026-05-31_e4_v2_11k_edge10_ablation",      "high"),
    (r"onehot_gin_B_v2_11k_onehot",             "2026-06-05_12_gin_B_v2_11k_onehot",         "high"),
    (r"onehot_gin_B_area_v2_13k",               "2026-06-01_gin_baseline_area",              "med"),
    (r"onehot_gin_B_v2_13k_onehot",             "2026-06-01_pure_gin",                       "low"),
    (r"8bit_only_onehot",                       "2026-06-03_pure_gin_8bit_only",             "high"),
    (r"mixed_5k8bit_onehot",                    "2026-06-03_pure_gin_mixed_5k8bit",          "low"),
    (r"mixed_onehot",                           "2026-06-02_pure_gin_mixed_onehot",          "high"),
    (r"area_v2_13k_E4_j",                       "2026-06-01_e4_v2_13k_area",                 "high"),
    (r"area_v2_13k_gin_baseline",               "2026-06-01_gin_baseline_area",              "high"),
    (r"v2_13k_E4_j",                            "2026-06-01_e4_v2_13k",                      "high"),
    (r"v2_13k_(E1|E2_m|E3_t|m234)",             "2026-06-01_ablation_v2_13k_power",          "high"),
    (r"v2_13k_gin_baseline",                    "2026-06-01_gin_baseline",                   "high"),
    (r"v2_13k_baseline",                        "2026-06-01_gin_baseline",                   "med"),
    (r"v2_9k_E4_j",                             "2026-05-30_e4_v2_9k_final",                 "high"),
    (r"v2_7k_e10_m234",                         "2026-05-30_e10_m234_5fold",                 "high"),
    (r"v2_7k_e10_(E1|E2_m|E3_t|E4_j)",          "2026-05-30_ablation_e10",                   "high"),
    (r"v2_7k(_fold|\.pth)",                     "2026-05-29_v2_7k_baseline",                 "high"),
    (r"v2_5k_D",                                "2026-05-26_runD_v2_init",                   "high"),
    (r"v2_5fold",                               "2026-05-27_v2_5fold_validation",            "high"),
]

UNSORTED = "_checkpoints_unsorted"

# 被 configs/ trainer/ scripts/ 引用的 ckpt, 留在 dataset/ 不动, 避免断引用
KEEP = {
    "glitch_power_proxy_gnn.pth",
    "glitch_power_proxy_gnn_D.pth",
    "glitch_power_proxy_gnn_fold0.pth",
    "glitch_power_proxy_gnn_B_v2_11k_phys_E4_fold1.pth",
    "glitch_power_proxy_gnn_B_v2_7k_e10_m234_fold0.pth",
    "glitch_power_dag_gnn_delay.pth",
    "glitch_power_onehot_gin.pth",
}


def classify(name):
    if name in KEEP:
        return None, "keep"
    for pat, dest, conf in RULES:
        if re.search(pat, name):
            return dest, conf
    return UNSORTED, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    pths = sorted(f for f in os.listdir(DATASET) if f.endswith(".pth"))
    plan = [(f, *classify(f)) for f in pths]

    kept = [f for f, dest, conf in plan if dest is None]
    movable = [(f, dest, conf) for f, dest, conf in plan if dest is not None]

    # 分组汇总
    by_dest = {}
    for f, dest, conf in movable:
        by_dest.setdefault(dest, []).append((f, conf))

    manifest = os.path.join(OUTPUTS, "_ckpt_move_manifest.tsv")
    undo = os.path.join(OUTPUTS, "_ckpt_move_undo.sh")
    os.makedirs(OUTPUTS, exist_ok=True)

    with open(manifest, "w") as mf:
        mf.write("src\tdest\tconfidence\n")
        for f, dest, conf in movable:
            mf.write(f"dataset/{f}\toutputs/{dest}/{f}\t{conf}\n")

    print(f"共 {len(plan)} 个 .pth: {len(movable)} 个待移动, {len(kept)} 个保留(被代码引用)\n")
    if kept:
        print("  🔒 保留在 dataset/ (被 configs/code 引用):")
        for f in kept:
            print(f"        {f}")
        print()
    for dest in sorted(by_dest):
        items = by_dest[dest]
        confs = {c for _, c in items}
        flag = "  ⚠ 含低置信" if ("low" in confs or "none" in confs) else ""
        print(f"  outputs/{dest}/  ({len(items)} 个){flag}")
        for f, conf in items:
            if conf in ("low", "none"):
                print(f"        [{conf}] {f}")
    print(f"\nmanifest: {manifest}")

    if not args.apply:
        print("\n(dry-run, 未移动。确认后加 --apply 执行)")
        return

    moved = []
    for f, dest, conf in movable:
        dst_dir = os.path.join(OUTPUTS, dest)
        os.makedirs(dst_dir, exist_ok=True)
        src = os.path.join(DATASET, f)
        dst = os.path.join(dst_dir, f)
        if os.path.exists(dst):
            print(f"  SKIP 已存在: {dst}")
            continue
        shutil.move(src, dst)
        moved.append((dst, src))
    with open(undo, "w") as uf:
        uf.write("#!/bin/bash\n# 撤销 ckpt 移动\nset -e\n")
        for dst, src in moved:
            uf.write(f'mv "{dst}" "{src}"\n')
    os.chmod(undo, 0o755)
    print(f"\n✅ 移动 {len(moved)} 个文件。undo 脚本: {undo}")


if __name__ == "__main__":
    main()
