import os
import sys
import random
import concurrent.futures

# 将当前目录加入系统路径，确保能够顺利 import run_power_sweep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_power_sweep import evaluate_single_routing


def evaluate_single_design(verilog_file_path, target_delay, bit_width=16):
    """
    EDA 桥梁包装器：读取 Verilog → 调用沙盒评估 → 返回 area, delay, power (mW)

    返回干净的原始数值，不加任何扰动。
    数据采集 (generate_dataset.py) 和搜索 (arith_das) 都调用此函数。
    """
    if not os.path.exists(verilog_file_path) or os.path.getsize(verilog_file_path) == 0:
        print(f"[致命错误] 本地文件 {verilog_file_path} 不存在或大小为 0！传输终止。")
        return float('inf'), float('inf'), float('inf')

    try:
        with open(verilog_file_path, "r", encoding="utf-8") as f:
            verilog_content = f.read()
    except Exception as e:
        print(f"[Error] 读取本地 Verilog 文件失败: {e}")
        return float('inf'), float('inf'), float('inf')

    dummy_idx = random.randint(1, 9999)

    result = evaluate_single_routing(
        idx=dummy_idx,
        verilog_content=verilog_content,
        bit_width=bit_width,
        target_delay=target_delay
    )

    if result.get("success"):
        if result.get("logic_failed"):
            print(f"⚠️ [ID: {dummy_idx:04d}] 综合通过，但逻辑测试 FAILED！(废弃)")
            return float('inf'), float('inf'), float('inf')

        area  = result.get("area", float('inf'))
        delay = abs(result.get("delay", float('inf')))
        power = result.get("power_mw", float('inf'))

        # ⚠️ 不再加随机扰动，返回干净原始值
        return area, delay, power

    else:
        print(f"❌ [ID: {dummy_idx:04d}] 评估失败: {result.get('log', 'Unknown error')}")
        return float('inf'), float('inf'), float('inf')


def evaluate_single_design_with_noise(verilog_file_path, target_delay, bit_width=16):
    """
    带微小扰动的版本，仅供 Pareto 搜索使用（打破平局锁死）。
    generate_dataset.py 不应调用此函数。
    """
    area, delay, power = evaluate_single_design(verilog_file_path, target_delay, bit_width)
    if area  != float('inf'): area  += random.uniform(0, 1e-5)
    if delay != float('inf'): delay += random.uniform(0, 1e-5)
    if power != float('inf'): power += random.uniform(0, 1e-5)
    return area, delay, power


def _evaluate_single_design_full(verilog_file_path, target_delay, bit_width=16):
    """完整版评估: 返回 dict 含 area/delay/power/node_powers (POC: per-node power)"""
    if not os.path.exists(verilog_file_path) or os.path.getsize(verilog_file_path) == 0:
        return {"area": float('inf'), "delay": float('inf'), "power": float('inf'), "node_powers": {}}
    try:
        with open(verilog_file_path, "r", encoding="utf-8") as f:
            verilog_content = f.read()
    except Exception:
        return {"area": float('inf'), "delay": float('inf'), "power": float('inf'), "node_powers": {}}

    dummy_idx = random.randint(1, 9999)
    result = evaluate_single_routing(
        idx=dummy_idx, verilog_content=verilog_content,
        bit_width=bit_width, target_delay=target_delay,
    )
    if result.get("success") and not result.get("logic_failed"):
        return {
            "area":  result.get("area",  float('inf')),
            "delay": abs(result.get("delay", float('inf'))),
            "power": result.get("power_mw", float('inf')),
            "node_powers": result.get("node_powers", {}),
        }
    return {"area": float('inf'), "delay": float('inf'), "power": float('inf'), "node_powers": {}}


def evaluate_batch_parallel(verilog_files, target_delay, bit_width=16, max_workers=8):
    """并行批量评估（使用干净版，不加扰动）。返回 dict 含 node_powers。"""
    results = []
    actual_workers = min(max_workers, len(verilog_files))
    print(f"开始并行评估 {len(verilog_files)} 个架构，并发数: {actual_workers}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_file = {
            executor.submit(_evaluate_single_design_full, vf, target_delay, bit_width): vf
            for vf in verilog_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            vf = future_to_file[future]
            try:
                res = future.result(timeout=900)
            except Exception as e:
                print(f"❌ 评估超时/异常: {vf} → {e}")
                res = {"area": float('inf'), "delay": float('inf'), "power": float('inf'), "node_powers": {}}
            res["file"] = vf
            results.append(res)

    return results
