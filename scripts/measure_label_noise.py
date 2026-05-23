"""
测量 EDA 工具复现噪声：同一个 Verilog 文件原封不动送 EDA N 次，统计 power 的方差。

这是模型 MAPE 的物理下限 —— 如果噪声 MAPE ≈ 模型 MAPE，模型已经到顶，
继续训没用，应该去补特征或换更稳的 EDA 流程。

用法（在项目根目录运行）:
    python3 scripts/measure_label_noise.py

默认 3 电路 × 5 次重复 = 15 次 EDA 调用，约 15~30 分钟。
"""
import os
import sys
import uuid
import statistics
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorGraph
from run_power_sweep import VerilogEmitter, generate_legal_random_routing, evaluate_single_routing


# ========================== 配置 ==========================
BIT_WIDTH      = 16
ENCODE_TYPE    = "and"
TARGET_DELAY   = 2.0
NUM_CIRCUITS   = 3       # 测多少个不同电路
NUM_REPEATS    = 5       # 每个电路重复送 EDA 几次
MAX_WORKERS    = 8       # SSH 并发上限，参考 run_power_sweep.py 的注释
BUILD_DIR      = "build_noise_test"


# ========================== 生成电路 ==========================
def generate_circuits(n_circuits: int):
    """生成 n_circuits 个合法的随机路由电路，返回 [(circuit_id, verilog_content), ...]"""
    pp = get_initial_partial_product(BIT_WIDTH, ENCODE_TYPE)
    ct = CompressorTree.dadda(pp)
    assignment = ct.compressor_assignment_fused()
    comp_graph = CompressorGraph(pp, assignment)
    emitter = VerilogEmitter(comp_graph)

    os.makedirs(BUILD_DIR, exist_ok=True)
    circuits = []

    for cid in range(n_circuits):
        rand_connections = generate_legal_random_routing(comp_graph)
        routing_assignment = emitter.emit_assignment(rand_connections)
        mul = Mul(BIT_WIDTH, ENCODE_TYPE, ct)

        tmp_path = os.path.join(BUILD_DIR, f"circuit_{cid}_{uuid.uuid4().hex[:6]}.v")
        mul.emit_verilog(tmp_path, assignment=routing_assignment)
        with open(tmp_path, "r") as f:
            content = f.read()
        os.remove(tmp_path)

        circuits.append((cid, content))
        print(f"  [生成] 电路 {cid}: {len(content)} bytes")

    return circuits


# ========================== 并行重测 ==========================
def measure_one_circuit(cid: int, verilog_content: str, n_repeats: int):
    """对同一份 verilog_content 跑 n_repeats 次 EDA，返回 power 列表（mW）"""
    print(f"\n  [测量] 电路 {cid}: 准备 {n_repeats} 次重复评估...")
    powers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n_repeats)) as ex:
        futures = {
            ex.submit(evaluate_single_routing, idx=cid * 1000 + rep,
                      verilog_content=verilog_content,
                      bit_width=BIT_WIDTH, target_delay=TARGET_DELAY): rep
            for rep in range(n_repeats)
        }
        for fut in concurrent.futures.as_completed(futures):
            rep = futures[fut]
            try:
                res = fut.result(timeout=900)
            except Exception as e:
                print(f"    ❌ 电路 {cid} 重复 {rep}: 异常 {e}")
                continue

            if res.get("success") and not res.get("logic_failed"):
                p = res["power_mw"]
                powers.append(p)
                print(f"    ✅ 电路 {cid} 重复 {rep}: power = {p:.6f} mW")
            else:
                print(f"    ❌ 电路 {cid} 重复 {rep}: 评估失败")

    return powers


# ========================== 统计 ==========================
def report(circuit_id: int, powers: list):
    if len(powers) < 2:
        print(f"  ⚠️  电路 {circuit_id}: 有效样本不足 ({len(powers)})，无法统计")
        return None

    mean = statistics.mean(powers)
    stdev = statistics.stdev(powers)
    pmin, pmax = min(powers), max(powers)
    cv = stdev / mean * 100                          # 变异系数 %
    mape_pairwise = sum(abs(p - mean) / mean for p in powers) / len(powers) * 100

    print(f"\n  📊 电路 {circuit_id} 统计 (n={len(powers)})")
    print(f"     mean   = {mean:.6f} mW")
    print(f"     std    = {stdev:.6f} mW")
    print(f"     range  = [{pmin:.6f}, {pmax:.6f}] (Δ={pmax-pmin:.6f})")
    print(f"     CV     = {cv:.3f}%   ← 噪声变异系数")
    print(f"     MAPE   = {mape_pairwise:.3f}%   ← 等价于模型的 MAPE 指标")

    return {"mean": mean, "std": stdev, "cv": cv, "mape": mape_pairwise, "n": len(powers)}


# ========================== 主入口 ==========================
def main():
    print(f"{'='*60}")
    print(f"  EDA 标签噪声测试")
    print(f"  位宽={BIT_WIDTH} 编码={ENCODE_TYPE} 目标延迟={TARGET_DELAY}ns")
    print(f"  电路数={NUM_CIRCUITS}  每电路重复={NUM_REPEATS}  EDA并发={MAX_WORKERS}")
    print(f"{'='*60}\n")

    circuits = generate_circuits(NUM_CIRCUITS)

    all_stats = []
    for cid, content in circuits:
        powers = measure_one_circuit(cid, content, NUM_REPEATS)
        stats = report(cid, powers)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\n  ❌ 所有电路都失败了，无法给出结论")
        return

    print(f"\n{'='*60}")
    print(f"  🎯 跨电路汇总")
    print(f"{'='*60}")
    avg_cv   = statistics.mean(s["cv"]   for s in all_stats)
    avg_mape = statistics.mean(s["mape"] for s in all_stats)
    max_mape = max(s["mape"] for s in all_stats)

    print(f"     平均 CV   = {avg_cv:.3f}%")
    print(f"     平均 MAPE = {avg_mape:.3f}%   ← 模型 MAPE 的物理下限")
    print(f"     最差 MAPE = {max_mape:.3f}%")
    print()
    print(f"  📌 对比你的模型:  当前训练 MAPE ≈ 2.65%")
    if avg_mape >= 2.0:
        print(f"     → 噪声 MAPE ({avg_mape:.2f}%) 已经接近模型 MAPE (2.65%)")
        print(f"     → 模型基本到顶，继续训练收益极低，应该去补特征或降噪")
    elif avg_mape >= 1.0:
        print(f"     → 噪声 MAPE ({avg_mape:.2f}%) 占模型 MAPE 的 {avg_mape/2.65*100:.0f}%")
        print(f"     → 仍有提升空间，但天花板有限（最多到 ~{avg_mape:.2f}%）")
    else:
        print(f"     → 噪声 MAPE ({avg_mape:.2f}%) 远低于模型 MAPE (2.65%)")
        print(f"     → 模型确实有欠拟合空间，调网络/特征都可能有收益")


if __name__ == "__main__":
    main()
