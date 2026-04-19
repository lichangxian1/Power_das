import concurrent.futures
import subprocess
import os
import re
import uuid
import random  # 确保导入了 random
import time    

def evaluate_single_design(verilog_file_path, target_delay):
    time.sleep(random.uniform(0.1, 3.0))
    # --- EDA 服务器配置 (目标端) ---
    eda_user = "lchangxian"
    eda_host = "202.120.39.27"
    eda_port = "16822" 
    eda_work_dir = "/home/lchangxian/arith_das_workspace" 
    
    unique_id = uuid.uuid4().hex[:8]
    design_name = f"mult_16b_{unique_id}"
    remote_verilog_name = f"{design_name}.v"

    # ==========================================
    # 步骤 1.1：先检查本地文件是不是空的
    # ==========================================
    if not os.path.exists(verilog_file_path) or os.path.getsize(verilog_file_path) == 0:
        print(f"[致命错误] 本地文件 {verilog_file_path} 不存在或大小为 0！传输终止。")
        return float('inf'), float('inf')

    # ==========================================
    # 步骤 1.2：智能提取 Verilog 内部的顶层模块名
    # ==========================================
    top_module = "unknown_module"
    try:
        with open(verilog_file_path, "r", encoding="utf-8") as f:
            # 抓取开头的模块名，比如 MUL
            match = re.search(r'module\s+([a-zA-Z0-9_]+)', f.read())
            if match:
                top_module = match.group(1) 
    except Exception as e:
        print(f"[Error] 读取本地 Verilog 文件失败: {e}")

    try:
        # ==========================================
        # 步骤 2：使用 rsync 发送文件到 EDA 服务器
        # ==========================================
        rsync_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {eda_port}",
            verilog_file_path,
            f"{eda_user}@{eda_host}:{eda_work_dir}/{remote_verilog_name}"
        ]
        subprocess.run(rsync_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ==========================================
        # 步骤 3：SSH 触发 DC 综合 【注意传入了 TOP_MODULE】
        # ==========================================
        ssh_run_cmd = [
            "ssh", "-p", eda_port,
            f"{eda_user}@{eda_host}",
            f"cd {eda_work_dir} && "
            f"env DESIGN_NAME={design_name} VERILOG_FILE={remote_verilog_name} TARGET_DELAY={target_delay} TOP_MODULE={top_module} "
            f"dc_shell -f synth.tcl"
        ]
        subprocess.run(ssh_run_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # ==========================================
        # 步骤 4：带“探针”的 SSH 远程读取与清理
        # ==========================================
        ssh_cat_cmd = [
            "ssh", "-p", eda_port,
            f"{eda_user}@{eda_host}",
            # 探针核心 Shell 脚本：
            # 1. 尝试 cat 报告（去掉了 2>/dev/null，让错误原形毕露）
            # 2. 立刻把 cat 的退出状态码存进变量 CAT_STATUS
            # 3. 无条件执行彻底的 rm -f 清理
            # 4. 最后把存下来的 CAT_STATUS 作为整个 SSH 的退出码返回给 Python
            f"cat {eda_work_dir}/{design_name}_area.rpt {eda_work_dir}/{design_name}_timing.rpt; "
            f"CAT_STATUS=$?; "
            f"rm -f {eda_work_dir}/{design_name}_*.rpt {eda_work_dir}/{design_name}.v; "
            f"exit $CAT_STATUS"
        ]
        
        # 探针捕获：启用 stderr=subprocess.PIPE 来抓取真正的报错文本
        result = subprocess.run(ssh_cat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        content = result.stdout
        error_info = result.stderr.strip()

        # --- 探针触发逻辑 ---
        if result.returncode != 0:
            print(f"[探针拦截] 设计 {design_name} 的报告读取失败！")
            print(f" -> 失败状态码: {result.returncode}")
            print(f" -> 真实死因: {error_info}")
            
            # 如果你想针对特定错误做处理，可以在这里写分支
            if "No such file or directory" in error_info:
                print(" -> 分析: DC 综合阶段可能已崩溃，未能生成报告。将按最差惩罚处理。\n")
            elif "Connection reset by peer" in error_info or "Timeout" in error_info:
                print(" -> 分析: SSH 连接被服务器防 DDoS 机制踢掉！\n")
            
            # 探针记录完毕，正常返回无穷大，让强化学习去惩罚它
            # 注意：补上打破平局的极小随机数
            return float('inf') + random.uniform(0, 1e-5), float('inf') + random.uniform(0, 1e-5)
        # ==========================================
        # 步骤 5：解析 PPA (加入随机扰动打破优先队列平局)
        # ==========================================
        area, delay = float('inf'), float('inf')
        area_match = re.search(r'Total cell area:\s+([0-9.]+)', content)
        if area_match:
            area = float(area_match.group(1)) + random.uniform(0, 1e-5)
            
        delay_match = re.search(r'data arrival time\s+([0-9.]+)', content)
        if delay_match:
            delay = float(delay_match.group(1)) + random.uniform(0, 1e-5)
            
        print(f"Parsed PPA: area = {area:.3f}, "
                f"delay = {delay:.3f} ")

        return area, delay

    except Exception as e:
        print(f"[Error] 评估 {verilog_file_path} 失败: {e}")
        return float('inf'), float('inf')

def evaluate_batch_parallel(verilog_files, target_delay, max_workers=8):
    """
    批量并行评估 Verilog 文件
    :param max_workers: 并发数。取决于你的 EDA 服务器有多少个 CPU 核心以及 DC License 数量。
    """
    results = []
    print(f"开始并行评估 {len(verilog_files)} 个架构，并发数: {max_workers}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将所有文件提交给线程池
        future_to_file = {
            executor.submit(evaluate_single_design, vf, target_delay): vf 
            for vf in verilog_files
        }
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_file):
            vf = future_to_file[future]
            area, delay = future.result()
            results.append({"file": vf, "area": area, "delay": delay})
            
    return results

# # ====== 调用示例 ======
# # 假设强化学习的当前 epoch 生成了 16 个不同的候选压缩树拓扑
# # candidate_files = ["./tmp/cand_1.v", "./tmp/cand_2.v", ..., "./tmp/cand_16.v"]
# # ppa_results = evaluate_batch_parallel(candidate_files, target_delay=1.5, max_workers=8)

# if __name__ == "__main__":
#     # 测试文件列表（复制一份假装有两个文件需要并行评估）
#     os.system("cp test_ha.v test_ha_2.v")
#     test_files = ["test_ha.v", "test_ha_2.v"]
    
#     # 设定目标延迟 1.5ns，最大并发 2
#     print("开始测试跨服务器 DC 综合...")
#     results = evaluate_batch_parallel(test_files, target_delay=1.5, max_workers=2)
    
#     for res in results:
#         print(f"文件: {res['file']} | 面积: {res['area']} | 延迟: {res['delay']}")