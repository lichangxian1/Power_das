"""用 Yosys+ABC+OpenROAD 合成评估功耗精准度，与数据集 ground-truth 对比。

流程:
  1. 从数据集图特征 (X, edge_index, edge_attr) 反向还原 Verilog
  2. Yosys+ABC 综合 → netlist
  3. OpenROAD report_power → 功耗估计
  4. 与数据集 power (mW) 对比，输出 tau/rho/R²/MAPE/Recall@K 指标

用法:
    python trainer/evaluate_abc.py \
        --data dataset/glitch_power_data_16bit_v2_13k_edge10.pt \
        --num_samples 500 \
        --workers 16 \
        --target_delay 2000 \
        --fig_dir outputs

注: Yosys 和 OpenROAD 必须在 PATH 中可用。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_initial_partial_product, CompressorTree
from utils.mul import Mul

# ─── 路径配置 ───────────────────────────────────────────────────────────────
T28_DIR  = "/home/changxian/library/t28_official"
LIB_PATH = os.path.join(T28_DIR, "tcbn28hpcplusbwp12t40p140tt0p9v25c.lib")
LEF_PATH = os.path.join(T28_DIR, "tcbn28hpcplusbwp12t40p140.lef")

# TSMC28 容性负载单位为 pF；0.05 pF = 50 fF，典型输出负载
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

STA_SCRIPT = """
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

set_power_activity -input -activity 0.5
report_power

exit
"""


# ─── Verilog 重建 ────────────────────────────────────────────────────────────
def _emit_assignment_from_graph(vertex_list, stage_num, col_num, edge_index, edge_attr):
    """从图特征重建 assignment dict。"""
    node_wires = {}
    INPUT_PORTS = ["a", "b", "c"]

    for e in range(edge_index.shape[1]):
        src_idx = edge_index[0, e].item()
        dst_idx = edge_index[1, e].item()
        port_oh = edge_attr[e, 2:5].tolist()
        if port_oh[0] > 0.5:
            dst_conc_type = 0
        elif port_oh[1] > 0.5:
            dst_conc_type = 1
        else:
            dst_conc_type = 2

        src_info = vertex_list[src_idx]
        dst_info = vertex_list[dst_idx]

        for idx, info in [(src_idx, src_info), (dst_idx, dst_info)]:
            if idx not in node_wires:
                t = info[2]
                if   t == 0: node_wires[idx] = {"from": {"a": None, "b": None, "c": None}, "to": {"sum": None, "carry": None}}
                elif t == 1: node_wires[idx] = {"from": {"a": None, "b": None},             "to": {"sum": None, "carry": None}}
                elif t == 2: node_wires[idx] = {"from": None,                                "to": {"sum": None}}
                elif t == 3: node_wires[idx] = {"from": {"a": None},                         "to": {"sum": None}}

        if src_info[1] == dst_info[1]:  # same col → sum signal
            node_wires[dst_idx]["from"][INPUT_PORTS[dst_conc_type]] = src_idx
            node_wires[src_idx]["to"]["sum"] = dst_idx
        else:                            # carry
            node_wires[dst_idx]["from"][INPUT_PORTS[dst_conc_type]] = src_idx
            node_wires[src_idx]["to"]["carry"] = dst_idx

    v_src = ""
    wire_set = set()

    def _decl(name):
        nonlocal v_src
        if name is not None and name not in wire_set:
            wire_set.add(name)
            v_src += f"    wire {name};\n"

    for node_idx, info in node_wires.items():
        type_idx = vertex_list[node_idx][2]
        if type_idx == 2:  # PP
            to_sum = info["to"]["sum"]
            _decl(f"from_{node_idx}_to_{to_sum}")
            st, co, _, li = vertex_list[node_idx]
            v_src += f"    assign from_{node_idx}_to_{to_sum} = pp_{co}[{li}];\n"
        elif type_idx == 3:  # pass
            a_src = info["from"]["a"]
            a_wire = f"from_{a_src}_to_{node_idx}"
            _decl(a_wire)
            st, _, _, _ = vertex_list[node_idx]
            to_sum = info["to"]["sum"]
            if st < stage_num and to_sum is not None:
                sum_wire = f"from_{node_idx}_to_{to_sum}"
                _decl(sum_wire)
                v_src += f"    assign {sum_wire} = {a_wire};\n"
            vis_wire = f"visual_{node_idx}"
            _decl(vis_wire)
            v_src += f"    assign {vis_wire} = {a_wire};\n"
        elif type_idx == 0:  # FA
            a_w = f"from_{info['from']['a']}_to_{node_idx}"
            b_w = f"from_{info['from']['b']}_to_{node_idx}"
            c_w = f"from_{info['from']['c']}_to_{node_idx}"
            s_w = f"from_{node_idx}_to_{info['to']['sum']}"
            carry_dst = info["to"]["carry"]
            cy_w = f"from_{node_idx}_to_{carry_dst}" if carry_dst is not None else None
            for w in [a_w, b_w, c_w, s_w, cy_w]:
                _decl(w)
            if cy_w:
                v_src += f"    FA  ct32_{node_idx} (.a({a_w}), .b({b_w}), .cin({c_w}), .sum({s_w}), .cout({cy_w}));\n"
            else:
                v_src += f"    FA_no_carry ct32_{node_idx} (.a({a_w}), .b({b_w}), .cin({c_w}), .sum({s_w}));\n"
        elif type_idx == 1:  # HA
            a_w = f"from_{info['from']['a']}_to_{node_idx}"
            b_w = f"from_{info['from']['b']}_to_{node_idx}"
            s_w = f"from_{node_idx}_to_{info['to']['sum']}"
            carry_dst = info["to"]["carry"]
            cy_w = f"from_{node_idx}_to_{carry_dst}" if carry_dst is not None else None
            for w in [a_w, b_w, s_w, cy_w]:
                _decl(w)
            if cy_w:
                v_src += f"    HA  ct22_{node_idx} (.a({a_w}), .cin({b_w}), .sum({s_w}), .cout({cy_w}));\n"
            else:
                v_src += f"    HA_no_carry ct22_{node_idx} (.a({a_w}), .cin({b_w}), .sum({s_w}));\n"

    routed_wire_list = [[] for _ in range(col_num)]
    for vertex_idx, (st, co, ty, _) in enumerate(vertex_list):
        if ty == 3 and st == stage_num:
            vis = f"visual_{vertex_idx}"
            routed_wire_list[co].append(vis)
            if vis not in wire_set:
                v_src += f"    wire {vis} = 1'b0;\n"
                wire_set.add(vis)

    return {"router_src": v_src, "routed_wire_list": routed_wire_list}


def reconstruct_verilog(sample, bit_width=16, encode_type="and"):
    """从图特征重建完整 MUL Verilog 字符串。"""
    X  = sample["X"]
    ei = sample["edge_index"]
    ea = sample["edge_attr"]

    vertex_list = []
    for i in range(X.shape[0]):
        stage_idx = int(X[i, 0].item())
        col_idx   = int(X[i, 1].item())
        local_idx = int(X[i, 2].item())
        type_idx  = int(X[i, 3:7].argmax().item())
        vertex_list.append((stage_idx, col_idx, type_idx, local_idx))

    col_num   = max(v[1] for v in vertex_list) + 1
    stage_num = max((v[0] for v in vertex_list if v[0] >= 0), default=0)

    assignment = _emit_assignment_from_graph(vertex_list, stage_num, col_num, ei, ea)

    pp   = get_initial_partial_product(bit_width, encode_type)
    ct32 = np.zeros(len(pp), dtype=int)
    ct22 = np.zeros(len(pp), dtype=int)

    m = Mul.__new__(Mul)
    m.bit_width   = bit_width
    m.encode_type = encode_type
    m.initial_pp  = pp
    m.ct          = CompressorTree(pp, ct32, ct22)

    return m.emit_verilog(assignment=assignment)


# ─── Yosys + OpenROAD ───────────────────────────────────────────────────────
def run_abc_power(sample_idx, verilog_str, target_delay, work_dir):
    """在 work_dir 下运行 Yosys+ABC → OpenROAD，返回 power_mW 或 None。"""
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
        f.write(STA_SCRIPT.format(
            lef_path=LEF_PATH, lib_path=LIB_PATH,
            netlist_path=netlist_path,
        ))

    # Yosys synthesis
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

    # OpenROAD power analysis
    try:
        with open(sta_log, "w") as fout:
            subprocess.run(
                ["openroad", sta_tcl],
                stdout=fout, stderr=subprocess.STDOUT, timeout=120,
            )
    except Exception:
        return None

    # 解析功耗 (Watts → mW)
    try:
        with open(sta_log, "r") as f:
            for line in f:
                words = line.split()
                if words and words[0] == "Total":
                    # 格式: Total  Internal  Switching  Leakage  <power_W>  <pct>%
                    power_w = float(words[-2])
                    return power_w * 1000.0  # → mW
    except Exception:
        pass
    return None


# ─── 指标 ───────────────────────────────────────────────────────────────────
def _topk_recall(pred, true, k_ratios=(0.05, 0.10, 0.20)):
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


# ─── 归一化散点图（与 evaluate_proxy.py 一致）────────────────────────────────
def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12), lo, hi


def _draw_panel(ax, t, p, title):
    """x/y 各自 min-max 归一化到 [0,1]，tick 标签还原原始物理量。"""
    t_n, t_lo, t_hi = _minmax(t)
    p_n, p_lo, p_hi = _minmax(p)

    ax.scatter(t_n, p_n, alpha=0.6, c="tomato", edgecolors="k",
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

    ax.set_xlabel("True Power (mW) [normalized]", fontsize=10)
    ax.set_ylabel("ABC Power (mW) [normalized]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)


# ─── 主评估流程 ──────────────────────────────────────────────────────────────
def evaluate_abc(
    data_path,
    num_samples=500,
    workers=8,
    target_delay=2000,
    eval_filter_n=730,
    seed=42,
    fig_dir="outputs",
    keep_tmpdir=False,
):
    print(f"\n{'=' * 70}")
    print(f"  ABC 基线功耗评估  (TSMC28)")
    print(f"  数据集:           {data_path}")
    print(f"  样本数上限:       {num_samples}")
    print(f"  并发 workers:     {workers}")
    print(f"  ABC target_delay: {target_delay} ps")

    dataset = torch.load(data_path, map_location="cpu", weights_only=False)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:num_samples]
    print(f"  实际使用样本:     {len(indices)} / {len(dataset)}")

    tmpdir = tempfile.mkdtemp(prefix="eval_abc_")
    print(f"  临时目录:         {tmpdir}")

    def _worker(idx):
        sample = dataset[idx]
        try:
            verilog = reconstruct_verilog(sample)
        except Exception as e:
            return idx, None, float("nan"), f"verilog_err: {e}"
        work_dir = os.path.join(tmpdir, f"sample_{idx}")
        power_mw = run_abc_power(idx, verilog, target_delay, work_dir)
        return idx, sample["power"], power_mw, None

    results = []
    fail_count = 0
    print(f"\n  🚀 开始并行综合 (workers={workers}) ...")
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, idx): idx for idx in indices}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx, gt, pred, err = fut.result()
            done += 1
            if pred is not None and np.isfinite(pred) and gt is not None and np.isfinite(gt):
                results.append((idx, float(gt), float(pred)))
            else:
                fail_count += 1
            if done % 50 == 0 or done == len(indices):
                elapsed = time.time() - t0
                print(f"    进度: {done}/{len(indices)} | 成功: {len(results)} "
                      f"| 失败: {fail_count} | 耗时: {elapsed:.0f}s")

    if not keep_tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if len(results) < 10:
        print(f"\n  ❌ 有效结果 {len(results)} 个，太少，退出。")
        return

    print(f"\n  有效样本: {len(results)} / {len(indices)}  (失败: {fail_count})")

    idx_arr  = np.array([r[0] for r in results])
    true_arr = np.array([r[1] for r in results])
    pred_arr = np.array([r[2] for r in results])

    ns = np.array([dataset[idx]["X"].shape[0] for idx in idx_arr])

    print()
    m_mix = _metrics(pred_arr, true_arr)
    _print_metrics("混合-N (全部)", m_mix)

    m_fix = None
    if eval_filter_n is not None:
        keep = (ns == eval_filter_n)
        n_keep = int(keep.sum())
        if n_keep >= 16:
            m_fix = _metrics(pred_arr[keep], true_arr[keep])
            _print_metrics(f"固定 N={eval_filter_n}", m_fix)
        else:
            print(f"  ⚠️ N={eval_filter_n} 仅 {n_keep} 个样本，跳过")

    print(f"\n  真值 power 范围: [{true_arr.min():.4f}, {true_arr.max():.4f}] mW")
    print(f"  ABC  power 范围: [{pred_arr.min():.4f}, {pred_arr.max():.4f}] mW")
    print(f"  ABC/真值 均值比: {pred_arr.mean()/true_arr.mean():.3f}")

    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "scatter_abc_baseline.png")

    if m_fix is not None:
        keep = (ns == eval_filter_n)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        _draw_panel(axes[0], true_arr, pred_arr,
                    f"Mixed-N\nn={m_mix['n']}  τ={m_mix['tau']:+.3f}  "
                    f"R²={m_mix['r2']:.3f}  MAPE={m_mix['mape']:.1f}%")
        _draw_panel(axes[1], true_arr[keep], pred_arr[keep],
                    f"Fixed-N={eval_filter_n}\nn={m_fix['n']}  τ={m_fix['tau']:+.3f}  "
                    f"R²={m_fix['r2']:.3f}  MAPE={m_fix['mape']:.1f}%")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_panel(ax, true_arr, pred_arr,
                    f"ABC Baseline\nn={m_mix['n']}  τ={m_mix['tau']:+.3f}  "
                    f"R²={m_mix['r2']:.3f}  MAPE={m_mix['mape']:.1f}%")

    plt.suptitle("ABC Synthesis Power vs Ground Truth  (TSMC28)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📈 散点图: {fig_path}")
    print(f"\n{'=' * 70}")
    return {"mix": m_mix, "fix": m_fix}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="dataset/glitch_power_data_16bit_v2_13k_edge10.pt")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="评估样本数 (0=全部)")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发 EDA 线程数")
    parser.add_argument("--target_delay", type=int, default=2000,
                        help="ABC 综合目标延迟 (ps)")
    parser.add_argument("--eval_filter_n", type=int, default=730,
                        help="fix-N 子集节点数; 0=关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig_dir", type=str, default="outputs")
    parser.add_argument("--keep_tmpdir", action="store_true",
                        help="保留临时工作目录 (调试用)")
    args = parser.parse_args()

    filter_n = None if args.eval_filter_n <= 0 else args.eval_filter_n
    n = len(torch.load(args.data, map_location="cpu", weights_only=False)) \
        if args.num_samples <= 0 else args.num_samples

    evaluate_abc(
        data_path=args.data,
        num_samples=n,
        workers=args.workers,
        target_delay=args.target_delay,
        eval_filter_n=filter_n,
        seed=args.seed,
        fig_dir=args.fig_dir,
        keep_tmpdir=args.keep_tmpdir,
    )
