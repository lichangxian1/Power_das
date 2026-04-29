import os
import sys
import random

# 将当前目录加入系统路径，确保能够顺利 import run_power_sweep
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_power_sweep import evaluate_single_routing

def evaluate_single_design(verilog_file_path, target_delay, bit_width=16):
    """
    作为包装器，直接调用 run_power_sweep.py 中已经跑通的沙盒评估函数
    返回: area, delay, power (mW)
    """
    # ==========================================
    # 1. 读取本地生成的 Verilog 文件内容
    # ==========================================
    if not os.path.exists(verilog_file_path) or os.path.getsize(verilog_file_path) == 0:
        print(f"[致命错误] 本地文件 {verilog_file_path} 不存在或大小为 0！传输终止。")
        return float('inf'), float('inf'), float('inf')

    try:
        with open(verilog_file_path, "r", encoding="utf-8") as f:
            verilog_content = f.read()
    except Exception as e:
        print(f"[Error] 读取本地 Verilog 文件失败: {e}")
        return float('inf'), float('inf'), float('inf')

    # ==========================================
    # 2. 调用 run_power_sweep 的完美验证版引擎
    # ==========================================
    dummy_idx = random.randint(1, 9999) # 随便给个 idx 用于日志打印区分
    
    # 这一步直接复用了你在 run_power_sweep.py 里写好的独立沙盒、重试机制、报错拦截
    result = evaluate_single_routing(
        idx=dummy_idx, 
        verilog_content=verilog_content, 
        bit_width=bit_width, 
        target_delay=target_delay
    )

    # ==========================================
    # 3. 解析并返回 PPA 给 ARITH-DAS 强化学习环境
    # ==========================================
    if result.get("success"):
        # 拦截逻辑错误：波形验证不通过的直接报废
        if result.get("logic_failed"):
            print(f"⚠️ [ID: {dummy_idx:04d}] 综合通过，但逻辑测试 FAILED！(废弃)")
            return float('inf'), float('inf'), float('inf')
        
        area = result.get("area", float('inf'))
        delay = abs(result.get("delay", float('inf')))
        power = result.get("power_mw", float('inf')) # 已经是 mW
        
        # 加入微小随机扰动打破 Pareto 前沿平局锁死
        if area != float('inf'): area += random.uniform(0, 1e-5)
        if delay != float('inf'): delay += random.uniform(0, 1e-5)
        if power != float('inf'): power += random.uniform(0, 1e-5)
        
        return area, delay, power
        
    else:
        # 如果穷尽了重试次数或者远端崩溃，返回无穷大惩罚
        print(f"❌ [ID: {dummy_idx:04d}] 评估失败: {result.get('log', 'Unknown error')}")
        return float('inf'), float('inf'), float('inf')

# 并行评估的接口保留（防备别处调用）
def evaluate_batch_parallel(verilog_files, target_delay, bit_width=16, max_workers=8):
    import concurrent.futures
    results = []
    print(f"开始并行评估 {len(verilog_files)} 个架构，并发数: {max_workers}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(evaluate_single_design, vf, target_delay, bit_width): vf 
            for vf in verilog_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            vf = future_to_file[future]
            area, delay, power = future.result()
            results.append({"file": vf, "area": area, "delay": delay, "power": power})
            
    return results