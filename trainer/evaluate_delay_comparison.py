"""ABC 延迟基线 vs DAGTimingGNN 代理模型对照散点图

用法:
    python trainer/evaluate_delay_comparison.py \
        --ckpt dataset/glitch_power_dag_gnn_delay_B_ens_cp_fold2.pth \
        --data dataset/glitch_power_data_16bit_v2_13k_edge10.pt \
        --num_abc 300 \
        --workers 16 \
        --target_delay 2000

输出: outputs/scatter_delay_comparison_<ckpt_tag>.png
"""
import os
import sys
import argparse
import tempfile
import subprocess
import concurrent.futures
import random
import time
import shutil

import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_proxy import ArithDataset, custom_collate
from proxy_mlp import DAGTimingGNN
from evaluate_abc import reconstruct_verilog

# ─── 节点特征补全 (13 → 16 维: 追加 pin_cap_a/b/c) ────────────────────────────
_FA_CELL = "FA1D0BWP12T40P140"
_HA_CELL = "HA1D0BWP12T40P140"
_CAP_NORM_PF = 0.05115


def _build_pin_cap_lookup(lib_path, fa_cell=_FA_CELL, ha_cell=_HA_CELL):
    """从 Liberty 文件读取 FA/HA 输入 pin 电容，返回 (fa_caps, ha_caps) dict。"""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from enrich_mis_physics_features import build_physical_tables
    fa, ha = build_physical_tables(lib_path, fa_cell, ha_cell)
    return fa.input_caps, ha.input_caps


def _add_pin_cap_features(X, fa_caps, ha_caps):
    """在 X[N, >=13] 后追加 pin_cap_a/b/c (归一化), 返回 X[N, orig+3]。"""
    N = X.shape[0]
    type_idx = X[:, 3:7].argmax(dim=1).long()
    pin_caps = torch.zeros(N, 3, dtype=X.dtype)
    fa_mask = type_idx == 0
    ha_mask = type_idx == 1     
    if fa_mask.any():
        pin_caps[fa_mask, 0] = fa_caps.get("A",  0.0) / _CAP_NORM_PF
        pin_caps[fa_mask, 1] = fa_caps.get("B",  0.0) / _CAP_NORM_PF
        pin_caps[fa_mask, 2] = fa_caps.get("CI", 0.0) / _CAP_NORM_PF
    if ha_mask.any():
        pin_caps[ha_mask, 0] = ha_caps.get("A", 0.0) / _CAP_NORM_PF
        pin_caps[ha_mask, 1] = ha_caps.get("B", 0.0) / _CAP_NORM_PF
    return torch.cat([X, pin_caps], dim=1)


class _PinCapEnrichedSubset(torch.utils.data.Dataset):
    """Subset wrapper that appends 3 pin-cap features to X on-the-fly."""

    def __init__(self, base_subset, fa_caps, ha_caps):
        self.base = base_subset
        self.fa_caps = fa_caps
        self.ha_caps = ha_caps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # item = (X, edge_index, edge_attr, power_norm, power_raw,
        #         node_powers, node_mask, area_norm, delay_norm)
        X = item[0]
        X16 = _add_pin_cap_features(X, self.fa_caps, self.ha_caps)
        return (X16,) + item[1:]

# ─── 路径配置 (与 evaluate_abc.py 一致) ─────────────────────────────────────
T28_DIR  = "/home/changxian/library/t28_official"
LIB_PATH = os.path.join(T28_DIR, "tcbn28hpcplusbwp12t40p140tt0p9v25c.lib")
LEF_PATH = os.path.join(T28_DIR, "tcbn28hpcplusbwp12t40p140.lef")

ABC_CONSTR = """
set_driving_cell BUFFD1BWP12T40P140
set_load 0.05 [all_outputs]
"""

YOSYS_SCRIPT = """
read -sv {rtl_path}
synth -top MUL
dfflibmap -liberty {lib_path}
abc -D {target_delay} -constr {constr_path} -liberty {lib_path}
write_verilog {netlist_path}
"""

# STA 脚本: 提取关键路径 arrival time (即电路延迟), 输出 "wns <ns>"
STA_DELAY_SCRIPT = """
read_lef {lef_path}
read_lib {lib_path}
read_verilog {netlist_path}
link_design MUL

set period 5
create_clock -period $period [get_ports clk]

set clk_period_factor .2
set clk [lindex [all_clocks] 0]
set period [get_property $clk period]
set delay [expr $period * $clk_period_factor]

set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]

set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"

exit
"""


# ─── ABC 延迟提取 ─────────────────────────────────────────────────────────────
def run_abc_delay(sample_idx, verilog_str, target_delay, work_dir):
    """合成并提取关键路径延迟 (ns), 失败返回 None。"""
    os.makedirs(work_dir, exist_ok=True)
    rtl_path     = os.path.join(work_dir, "MUL.v")
    netlist_path = os.path.join(work_dir, "netlist.v")
    constr_path  = os.path.join(work_dir, "constr.sdc")
    yosys_ys     = os.path.join(work_dir, "synth.ys")
    sta_tcl      = os.path.join(work_dir, "sta.tcl")
    yosys_log    = os.path.join(work_dir, "yosys.log")
    sta_log      = os.path.join(work_dir, "sta.log")

    with open(rtl_path,    "w") as f: f.write(verilog_str)
    with open(constr_path, "w") as f: f.write(ABC_CONSTR)
    with open(yosys_ys,    "w") as f:
        f.write(YOSYS_SCRIPT.format(
            rtl_path=rtl_path, lib_path=LIB_PATH,
            target_delay=target_delay, constr_path=constr_path,
            netlist_path=netlist_path,
        ))
    with open(sta_tcl, "w") as f:
        f.write(STA_DELAY_SCRIPT.format(
            lef_path=LEF_PATH, lib_path=LIB_PATH,
            netlist_path=netlist_path,
        ))

    # Yosys 合成
    try:
        with open(yosys_log, "w") as fout:
            ret = subprocess.run(
                ["yosys", yosys_ys],
                stdout=fout, stderr=subprocess.STDOUT, timeout=120,
            )
        if ret.returncode != 0 or not os.path.exists(netlist_path):
            return None
    except Exception:
        return None

    # OpenROAD STA 延迟分析
    try:
        with open(sta_log, "w") as fout:
            subprocess.run(
                ["openroad", sta_tcl],
                stdout=fout, stderr=subprocess.STDOUT, timeout=120,
            )
    except Exception:
        return None

    # 解析 "wns <delay_ns>"
    try:
        with open(sta_log, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "wns":
                    return float(parts[1])
    except Exception:
        pass
    return None


# ─── 指标 ─────────────────────────────────────────────────────────────────────
def _topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20)):
    """delay 越小越好 → 取最小的 K 个"""
    n = len(pred)
    out = {}
    for r in k_ratios:
        k = max(1, int(round(n * r)))
        true_topk = set(np.argsort(true)[:k].tolist())
        pred_topk = set(np.argsort(pred)[:k].tolist())
        out[r] = len(true_topk & pred_topk) / k
    return out


def _metrics(pred, true):
    tau,  tau_p = stats.kendalltau(pred, true)
    rho,  _     = stats.spearmanr(pred, true)
    pr,   _     = stats.pearsonr(pred, true)
    mape = np.mean(np.abs(true - pred) / (np.abs(true) + 1e-8)) * 100
    mse  = np.mean((pred - true) ** 2)
    recalls = _topk_recall(pred, true)
    return {
        "n":        len(pred),
        "tau":      0.0 if np.isnan(tau) else tau,
        "tau_pval": tau_p,
        "rho":      0.0 if np.isnan(rho) else rho,
        "r2":       (pr ** 2) if not np.isnan(pr) else 0.0,
        "mape":     mape,
        "mse":      mse,
        "recalls":  recalls,
    }


def _print_metrics(tag, m):
    r = m["recalls"]
    print(f"  [{tag}] n={m['n']:4d} | τ={m['tau']:+.4f} (p={m['tau_pval']:.1e}) "
          f"| ρ={m['rho']:+.4f} | R²={m['r2']:.4f}")
    print(f"           R@5%={r[0.05]:.3f}  R@10%={r[0.10]:.3f}  R@20%={r[0.20]:.3f}  "
          f"| MAPE={m['mape']:.2f}%  MSE={m['mse']:.2e}")


# ─── 代理模型推理 ─────────────────────────────────────────────────────────────
def _build_dag_model(ckpt, device):
    model = DAGTimingGNN(
        node_feature_dim=ckpt.get("node_feature_dim", 16),
        hidden_dim=ckpt.get("hidden_dim", 96),
        num_gnn_layers=ckpt.get("num_gnn_layers", 4),
        dropout=ckpt.get("dropout", 0.1),
        topo_idx=ckpt.get("topo_idx", 0),
        arrival_idx=ckpt.get("arrival_idx", 7),
        use_edge_feat=ckpt.get("use_edge_feat", True),
        external_edge_attr_dim=ckpt.get("external_edge_attr_dim", 0),
        use_mean_agg=ckpt.get("use_mean_agg", True),
        readout_beta=ckpt.get("readout_beta", 8.0),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  ⚠️ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def _run_proxy_inference(model, loader, device, delay_mean, delay_std):
    """返回 (pred_ns, true_ns, n_nodes) numpy arrays"""
    preds, trues, ns = [], [], []
    with torch.no_grad():
        for (X, edge_index, edge_attr, mask, _pn, pr,
             _npw, _nm, _ar, _dl) in loader:
            X          = X.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred_norm = model(X, edge_index, mask, edge_attr=edge_attr)
            pred_raw  = pred_norm * delay_std + delay_mean
            preds.append(pred_raw.cpu())
            trues.append(pr)
            ns.append(mask.sum(dim=1).long().cpu())
    return (torch.cat(preds).numpy(),
            torch.cat(trues).numpy(),
            torch.cat(ns).numpy())


# ─── 散点面板 (与 evaluate_abc.py 风格一致) ──────────────────────────────────
def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12), lo, hi


def _draw_panel(ax, t, p, title, xlabel="True Delay (ns)", ylabel="Predicted Delay (ns)"):
    t_n, t_lo, t_hi = _minmax(t)
    p_n, p_lo, p_hi = _minmax(p)

    ax.scatter(t_n, p_n, alpha=0.6, c="dodgerblue", edgecolors="k",
               s=25, linewidths=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="ideal (y=x)", zorder=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    xticks = np.linspace(0, 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t_lo + v*(t_hi - t_lo):.3f}" for v in xticks], fontsize=8)
    yticks = np.linspace(0, 1, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{p_lo + v*(p_hi - p_lo):.3f}" for v in yticks], fontsize=8)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)


# ─── 主流程 ──────────────────────────────────────────────────────────────────
def evaluate_delay_comparison(
    ckpt_path,
    data_path,
    num_abc=300,
    workers=8,
    target_delay=2000,
    kfold=5,
    eval_filter_n=730,
    seed=42,
    fig_dir="outputs",
    keep_tmpdir=False,
    batch_size=256,
    lib_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")
    print(f"  延迟对照评估: ABC 基线 vs DAGTimingGNN 代理")
    print(f"  ckpt:             {ckpt_path}")
    print(f"  data:             {data_path}")
    print(f"  device:           {device}")

    # ── 1. 加载 ckpt 和代理模型 ────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n  代理模型 ({ckpt.get('model_class', '?')}): "
          f"dim={ckpt.get('hidden_dim')}  layers={ckpt.get('num_gnn_layers')}  "
          f"best_τ={ckpt.get('best_tau', '?')}")
    model = _build_dag_model(ckpt, device)

    # ── 2. 加载数据集, 构造 val 子集 (KFold 与训练一致) ───────────────────
    full_ds = ArithDataset(data_path, target="delay")
    delay_mean = float(ckpt.get("power_mean", full_ds.power_mean.item()))
    delay_std  = float(ckpt.get("power_std",  full_ds.power_std.item()))
    print(f"  delay 归一化: mean={delay_mean:.4f} ns  std={delay_std:.4f} ns")

    fold_id = ckpt.get("fold_id", 0)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    splits = list(kf.split(range(len(full_ds))))
    _, val_idx = splits[fold_id]
    print(f"  KFold {fold_id+1}/{kfold}: val={len(val_idx)} 样本")

    # 如果模型期望比数据集更多的 node 特征, 补全 pin_cap_a/b/c (13→16)
    data_x_dim = full_ds.data[0]["X"].shape[1]
    model_x_dim = ckpt.get("node_feature_dim", data_x_dim)
    val_base = Subset(full_ds, val_idx)
    _lib = lib_path or LIB_PATH
    if model_x_dim > data_x_dim:
        print(f"  ⚠️  特征维度不匹配: 数据={data_x_dim}, 模型={model_x_dim}; "
              f"从 Liberty 补全 {model_x_dim - data_x_dim} 维 pin_cap 特征...")
        fa_caps, ha_caps = _build_pin_cap_lookup(_lib)
        val_ds = _PinCapEnrichedSubset(val_base, fa_caps, ha_caps)
        print(f"     FA pin_caps: {fa_caps}")
        print(f"     HA pin_caps: {ha_caps}")
    else:
        fa_caps = ha_caps = None
        val_ds = val_base

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=custom_collate, num_workers=0, pin_memory=True)

    # ── 3. 代理模型推理 (全 val 集) ────────────────────────────────────────
    print(f"\n  🚀 代理模型推理 ...")
    proxy_pred, proxy_true, proxy_ns = _run_proxy_inference(
        model, loader, device,
        torch.tensor(delay_mean, device=device),
        torch.tensor(delay_std,  device=device),
    )

    m_proxy_mix = _metrics(proxy_pred, proxy_true)
    _print_metrics("代理 mix (全 val)", m_proxy_mix)

    m_proxy_fix = None
    if eval_filter_n is not None:
        keep_p = (proxy_ns == eval_filter_n)
        if keep_p.sum() >= 16:
            m_proxy_fix = _metrics(proxy_pred[keep_p], proxy_true[keep_p])
            _print_metrics(f"代理 fix N={eval_filter_n}", m_proxy_fix)

    # ── 4. ABC 延迟评估 (全 val fold, 与 proxy 一致) ──────────────────────
    raw_dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    abc_indices = list(val_idx)
    if num_abc > 0 and num_abc < len(abc_indices):
        rng = random.Random(seed)
        rng.shuffle(abc_indices)
        abc_indices = abc_indices[:num_abc]
        print(f"\n  ABC 合成评估: {len(abc_indices)} 样本 (采样)  "
              f"(workers={workers}, target_delay={target_delay} ps)")
    else:
        print(f"\n  ABC 合成评估: {len(abc_indices)} 样本 (全 val fold)  "
              f"(workers={workers}, target_delay={target_delay} ps)")

    tmpdir = tempfile.mkdtemp(prefix="eval_delay_abc_")
    print(f"  临时目录: {tmpdir}")

    def _worker(idx):
        sample = raw_dataset[idx]
        try:
            verilog = reconstruct_verilog(sample)
        except Exception as e:
            return idx, None, None, f"verilog_err: {e}"
        work_dir = os.path.join(tmpdir, f"sample_{idx}")
        delay_ns = run_abc_delay(idx, verilog, target_delay, work_dir)
        gt = float(sample["delay"])
        return idx, gt, delay_ns, None

    abc_results = []
    fail_count = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, idx): idx for idx in abc_indices}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx, gt, pred, err = fut.result()
            done += 1
            if (pred is not None and np.isfinite(pred)
                    and gt is not None and np.isfinite(gt)):
                abc_results.append((idx, gt, pred))
            else:
                fail_count += 1
            if done % 50 == 0 or done == len(abc_indices):
                elapsed = time.time() - t0
                print(f"    进度: {done}/{len(abc_indices)} | 成功: {len(abc_results)} "
                      f"| 失败: {fail_count} | 耗时: {elapsed:.0f}s")

    if not keep_tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if len(abc_results) < 10:
        print(f"\n  ❌ ABC 有效结果 {len(abc_results)} 个，太少，退出。")
        return

    abc_true = np.array([r[1] for r in abc_results])
    abc_pred = np.array([r[2] for r in abc_results])
    abc_ns   = np.array([raw_dataset[r[0]]["X"].shape[0] for r in abc_results])

    print(f"\n  ABC 有效样本: {len(abc_results)} / {len(abc_indices)}")
    m_abc_mix = _metrics(abc_pred, abc_true)
    _print_metrics("ABC  mix (采样)", m_abc_mix)

    m_abc_fix = None
    if eval_filter_n is not None:
        keep_a = (abc_ns == eval_filter_n)
        if keep_a.sum() >= 16:
            m_abc_fix = _metrics(abc_pred[keep_a], abc_true[keep_a])
            _print_metrics(f"ABC  fix N={eval_filter_n}", m_abc_fix)

    print(f"\n  GT   delay 范围: [{abc_true.min():.4f}, {abc_true.max():.4f}] ns")
    print(f"  ABC  delay 范围: [{abc_pred.min():.4f}, {abc_pred.max():.4f}] ns")
    print(f"  代理 delay 范围: [{proxy_pred.min():.4f}, {proxy_pred.max():.4f}] ns")

    # ── 5. 散点图 ──────────────────────────────────────────────────────────
    os.makedirs(fig_dir, exist_ok=True)
    tag = os.path.basename(ckpt_path).replace(".pth", "")
    fig_path = os.path.join(fig_dir, f"scatter_delay_comparison_{tag}.png")

    n_cols = 2 if (m_abc_fix is not None or m_proxy_fix is not None) else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 6))

    # 左: ABC 基线
    abc_title = (f"ABC Baseline (Delay)\n"
                 f"n={m_abc_mix['n']}  τ={m_abc_mix['tau']:+.3f}  "
                 f"R²={m_abc_mix['r2']:.3f}  MAPE={m_abc_mix['mape']:.1f}%")
    _draw_panel(axes[0], abc_true, abc_pred, abc_title,
                xlabel="True Delay (ns) [normalized]",
                ylabel="ABC Delay (ns) [normalized]")

    # 右: 代理模型
    proxy_title = (f"DAGTimingGNN Proxy (fold {fold_id})\n"
                   f"n={m_proxy_mix['n']}  τ={m_proxy_mix['tau']:+.3f}  "
                   f"R²={m_proxy_mix['r2']:.3f}  MAPE={m_proxy_mix['mape']:.1f}%")
    _draw_panel(axes[1], proxy_true, proxy_pred, proxy_title,
                xlabel="True Delay (ns) [normalized]",
                ylabel="Proxy Delay (ns) [normalized]")

    plt.suptitle(f"Delay: ABC Synthesis vs Proxy Model  (TSMC28)\n{tag}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 散点图: {fig_path}")
    print(f"\n{'=' * 70}")

    return {
        "abc":   {"mix": m_abc_mix,   "fix": m_abc_fix},
        "proxy": {"mix": m_proxy_mix, "fix": m_proxy_fix},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="dataset/glitch_power_dag_gnn_delay_B_ens_cp_fold2.pth")
    parser.add_argument("--data", type=str,
                        default="dataset/glitch_power_data_16bit_v2_13k_edge10.pt")
    parser.add_argument("--num_abc", type=int, default=0,
                        help="ABC 评估样本数; 0=全部 val fold (与 proxy 一致)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发 EDA 线程数")
    parser.add_argument("--target_delay", type=int, default=2000,
                        help="ABC 综合目标延迟 (ps)")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--eval_filter_n", type=int, default=730,
                        help="fix-N 子集节点数; 0=关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig_dir", type=str, default="outputs")
    parser.add_argument("--keep_tmpdir", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lib_path", type=str, default=None,
                        help="Liberty 文件路径 (用于补全 pin_cap 特征; 默认使用内置路径)")
    args = parser.parse_args()

    filter_n = None if args.eval_filter_n <= 0 else args.eval_filter_n
    evaluate_delay_comparison(
        ckpt_path=args.ckpt,
        data_path=args.data,
        num_abc=args.num_abc,
        workers=args.workers,
        target_delay=args.target_delay,
        kfold=args.kfold,
        eval_filter_n=filter_n,
        seed=args.seed,
        fig_dir=args.fig_dir,
        keep_tmpdir=args.keep_tmpdir,
        batch_size=args.batch_size,
        lib_path=args.lib_path,
    )
