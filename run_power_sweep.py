import os
import re
import time
import uuid
import random
import subprocess
import concurrent.futures
import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入 ARITH-DAS 组件
from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorGraph

# ==============================================================================
# 1. 纯净版 Verilog 连线生成器 & 随机合法连线发生器
# ==============================================================================
class VerilogEmitter:
    def __init__(self, comp_graph):
        self.comp_graph = comp_graph

    @staticmethod
    def _add_node(node_id, node_type, node_wires):
        if node_id not in node_wires:
            if node_type == 0: node_wires[node_id] = {"from": {"a": None, "b": None, "c": None}, "to": {"sum": None, "carry": None}}
            elif node_type == 1: node_wires[node_id] = {"from": {"a": None, "b": None}, "to": {"sum": None, "carry": None}}
            elif node_type == 2: node_wires[node_id] = {"from": None, "to": {"sum": None}}
            elif node_type == 3: node_wires[node_id] = {"from": {"a": None}, "to": {"sum": None}}
        return node_wires

    @staticmethod
    def _declare_wire(wire_name, wire_set: set):
        if wire_name is None: return "", wire_set
        v_src = ""
        if wire_name not in wire_set:
            wire_set.add(wire_name)
            v_src += f"    wire {wire_name};\n"
        return v_src, wire_set

    def emit_assignment(self, samples_connection):
        node_wires = {}
        INPUT_PORTS = ["a", "b", "c"]
        for src_idx, dst_idx, dst_conc_type, _ in samples_connection:
            src_info = self.comp_graph.vertex_list[src_idx]
            dst_info = self.comp_graph.vertex_list[dst_idx]
            node_wires = self._add_node(src_idx, src_info[2], node_wires)
            node_wires = self._add_node(dst_idx, dst_info[2], node_wires)

            if src_info[1] == dst_info[1]:
                node_wires[dst_idx]["from"][INPUT_PORTS[dst_conc_type]] = src_idx
                node_wires[src_idx]["to"]["sum"] = dst_idx
            else:
                node_wires[dst_idx]["from"][INPUT_PORTS[dst_conc_type]] = src_idx
                node_wires[src_idx]["to"]["carry"] = dst_idx

        v_src = ""
        wire_set = set()
        for node_idx in node_wires.keys():
            type_idx = self.comp_graph.vertex_list[node_idx][2]
            if type_idx == 2:
                v, wire_set = self._declare_wire(f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}", wire_set)
                v_src += v + f"    assign from_{node_idx}_to_{node_wires[node_idx]['to']['sum']} = pp_{self.comp_graph.vertex_list[node_idx][1]}[{self.comp_graph.vertex_list[node_idx][3]}];\n"
            elif type_idx == 3:
                a_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}"
                sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}" if self.comp_graph.vertex_list[node_idx][0] < self.comp_graph.stage_num else None
                v, wire_set = self._declare_wire(a_wire, wire_set)
                v_src += v
                if sum_wire: 
                    v, wire_set = self._declare_wire(sum_wire, wire_set)
                    v_src += v + f"    assign {sum_wire} = {a_wire};\n"
                v, wire_set = self._declare_wire(f"visual_{node_idx}", wire_set)
                v_src += v + f"    assign visual_{node_idx} = {a_wire};\n"
            elif type_idx == 0:
                a_wire, b_wire, c_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}", f"from_{node_wires[node_idx]['from']['b']}_to_{node_idx}", f"from_{node_wires[node_idx]['from']['c']}_to_{node_idx}"
                sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
                carry_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['carry']}" if node_wires[node_idx]["to"]["carry"] else None
                for wire in [a_wire, b_wire, c_wire, sum_wire, carry_wire]:
                    v, wire_set = self._declare_wire(wire, wire_set)
                    v_src += v
                v_src += f"    FA{' ' if carry_wire else '_no_carry '} ct32_{node_idx} (.a({a_wire}), .b({b_wire}), .cin({c_wire}), .sum({sum_wire}){', .cout('+carry_wire+')' if carry_wire else ''});\n"
            elif type_idx == 1:
                a_wire, b_wire = f"from_{node_wires[node_idx]['from']['a']}_to_{node_idx}", f"from_{node_wires[node_idx]['from']['b']}_to_{node_idx}"
                sum_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['sum']}"
                carry_wire = f"from_{node_idx}_to_{node_wires[node_idx]['to']['carry']}" if node_wires[node_idx]["to"]["carry"] else None
                for wire in [a_wire, b_wire, sum_wire, carry_wire]:
                    v, wire_set = self._declare_wire(wire, wire_set)
                    v_src += v
                v_src += f"    HA{' ' if carry_wire else '_no_carry '} ct22_{node_idx} (.a({a_wire}), .cin({b_wire}), .sum({sum_wire}){', .cout('+carry_wire+')' if carry_wire else ''});\n"

        routed_wire_list = [[] for _ in range(self.comp_graph.col_num)]
        for vertex_idx, (stage_idx, col_idx, type_idx, _) in enumerate(self.comp_graph.vertex_list):
            if type_idx == 3 and stage_idx == self.comp_graph.stage_num:
                routed_wire_list[col_idx].append(f"visual_{vertex_idx}")
                # 🛡️ 究极防错
                if f"visual_{vertex_idx}" not in wire_set:
                    v_src += f"    wire visual_{vertex_idx} = 1'b0;\n"
                    wire_set.add(f"visual_{vertex_idx}")

        return {"router_src": v_src, "routed_wire_list": routed_wire_list}

def generate_legal_random_routing(comp_graph):
    samples_connection = []
    for s in range(comp_graph.stage_num + 1):
        for c in range(comp_graph.col_num):
            sum_mask = comp_graph.get_slice_sum_mask(s, c)
            if c == 0:
                M_a, M_b, M_c = sum_mask[0, :, :], sum_mask[1, :, :], sum_mask[2, :, :]
            else:
                carry_mask = comp_graph.get_slice_carry_mask(s, c)
                M_a = torch.cat((sum_mask[0, :, :], carry_mask[0, :, :]), dim=0)
                M_b = torch.cat((sum_mask[1, :, :], carry_mask[1, :, :]), dim=0)
                M_c = torch.cat((sum_mask[2, :, :], carry_mask[2, :, :]), dim=0)
            M = torch.cat((M_a, M_b, M_c), dim=1)
            
            Z_random = torch.rand_like(M, dtype=torch.float)
            sum_src_indices = comp_graph.slice_indice_map[(s - 1, c)]
            dst_indices = comp_graph.slice_indice_map[(s, c)]
            
            for local_src_idx, src_idx in enumerate(sum_src_indices):
                logits = Z_random[local_src_idx, :].masked_fill(~M[local_src_idx, :], -1e9)
                sample = torch.distributions.Categorical(logits=logits).sample()
                M[:, sample.item()] = False
                samples_connection.append((src_idx, dst_indices[sample.item() % len(dst_indices)], sample.item() // len(dst_indices), None))
            
            if c > 0:
                carry_src_indices = comp_graph.slice_indice_map[(s - 1, c - 1)]
                for local_src_idx, src_idx in enumerate(carry_src_indices):
                    if comp_graph.vertex_list[src_idx][2] in [2, 3]: continue
                    idx_in_M = local_src_idx + len(sum_src_indices)
                    logits = Z_random[idx_in_M, :].masked_fill(~M[idx_in_M, :], -1e9)
                    sample = torch.distributions.Categorical(logits=logits).sample()
                    M[:, sample.item()] = False
                    samples_connection.append((src_idx, dst_indices[sample.item() % len(dst_indices)], sample.item() // len(dst_indices), None))
    return samples_connection

# ==============================================================================
# 2. 跨服安全沙盒评估引擎
# ==============================================================================
EDA_USER = "lchangxian"
EDA_HOST = "202.120.39.27"
EDA_PORT = "16822"
EDA_BASE_DIR = "/home/lchangxian/sandbox/sandbox_base" 

def evaluate_single_routing(idx, verilog_content, bit_width=8, target_delay=1.5):
    # 打散初始并发洪峰
    time.sleep(random.uniform(0.5, 5.0))
    
    top_module = "MUL1"
    match = re.search(r'module\s+([a-zA-Z0-9_]+)', verilog_content)
    if match: top_module = match.group(1)

    uid = uuid.uuid4().hex[:6]
    local_v_path = f"build/{top_module}_{uid}.v"
    os.makedirs("build", exist_ok=True)
    
    with open(local_v_path, "w") as f:
        f.write(verilog_content)

    remote_sandbox = f"/home/{EDA_USER}/sandbox/sandbox_sets/sandbox_{uid}"
    
    MAX_RETRIES = 3 # License 重试机制
    
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.1, 10.0))
            ssh_setup_cmd = [
                "ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}",
                f"cp -r {EDA_BASE_DIR} {remote_sandbox} && "
                f"sed -i 's#> /dev/null 2>&1##g' {remote_sandbox}/scripts/run_xa_vcs.sh && "
                f"rm -rf {remote_sandbox}/results/* {remote_sandbox}/reports/* {remote_sandbox}/src/rtl/* {remote_sandbox}/src/tb/file_* || true"
            ]
            subprocess.run(ssh_setup_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            rsync_cmd = [
                "rsync", "-avz", "-e", f"ssh -p {EDA_PORT}", 
                local_v_path, f"{EDA_USER}@{EDA_HOST}:{remote_sandbox}/src/rtl/{top_module}_{bit_width}.v"
            ]
            subprocess.run(rsync_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ssh_run_cmd = [
                "ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}",
                f"cd {remote_sandbox} && bash scripts/run_all.sh {top_module} {bit_width} {target_delay}"
            ]
            result = subprocess.run(ssh_run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            stdout = result.stdout

            # [License 挤兑保护机制]
            if "FATAL" in stdout:
                log_cmd = ["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"tail -n 100 {remote_sandbox}/dc_synth*.log 2>/dev/null; tail -n 50 {remote_sandbox}/*.log 2>/dev/null"]
                log_res = subprocess.run(log_cmd, capture_output=True, text=True, check=False)
                
                if "DCSH-1" in log_res.stdout or "is not enabled" in log_res.stdout:
                    print(f"⚠️ [ID: {idx:04d}] DC License 被抢光 (Attempt {attempt+1}/{MAX_RETRIES})，等待 15-30 秒后重试...")
                    subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"rm -rf {remote_sandbox}"], check=False)
                    time.sleep(random.uniform(15.0, 30.0)) 
                    continue 
                else:
                    if log_res.stdout: stdout += f"\n\n--- 🕵️ 远端崩溃现场核心日志 ---\n{log_res.stdout[-2500:]}\n-----------------------------------------\n"

            # 正常清理
            subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"rm -rf {remote_sandbox}"], check=False)
            if os.path.exists(local_v_path): os.remove(local_v_path)

            # 🌟 数据解析与逻辑报错拦截
            power, area, delay = None, None, None
            logic_failed = False
            
            for line in stdout.split('\n'):
                if "总功耗" in line and "W" in line:
                    val_str = line.split(':')[1].strip().split()[0]
                    if val_str != "N/A" and val_str != "File": power = float(val_str)
                elif "芯片总面积" in line:
                    area = float(line.split(':')[1].strip().split()[0])
                elif "极限工作延迟" in line:
                    delay = float(line.split(':')[1].strip().split()[0])
                elif "FAILED:" in line: # 拦截到了仿真平台的报错打印！
                    logic_failed = True
            
            if power is not None:
                return {"id": idx, "power_mw": power * 1000, "area": area, "delay": delay, "success": True, "logic_failed": logic_failed}
            else:
                return {"id": idx, "success": False, "log": stdout}

        except Exception as e:
            subprocess.run(["ssh", "-p", EDA_PORT, f"{EDA_USER}@{EDA_HOST}", f"rm -rf {remote_sandbox}"], check=False)
            return {"id": idx, "success": False, "log": str(e)}

    # 耗尽重试次数
    if os.path.exists(local_v_path): os.remove(local_v_path)
    return {"id": idx, "success": False, "log": "Max retries exceeded due to License limits."}

# ==============================================================================
# 3. 并发主循环与绘图
# ==============================================================================
if __name__ == "__main__":
    BIT_WIDTH = 16
    ENCODE_TYPE = "booth"
    TARGET_PERIOD = 1.5
    TOTAL_SAMPLES = 1000     
    MAX_WORKERS = 50          # ⚠️ 强烈建议不要超过 10，保护 SSH 隧道！

    print(f"🚀 [Phase 1] 正在生成 {TOTAL_SAMPLES} 个纯净的随机连线 Verilog...")
    pp = get_initial_partial_product(BIT_WIDTH, ENCODE_TYPE)
    ct = CompressorTree.dadda(pp)
    assignment = ct.compressor_assignment_fused()
    comp_graph = CompressorGraph(pp, assignment)
    emitter = VerilogEmitter(comp_graph)

    verilog_contents = []
    for _ in range(TOTAL_SAMPLES):
        rand_connections = generate_legal_random_routing(comp_graph)
        routing_assignment = emitter.emit_assignment(rand_connections)
        mul_module = Mul(BIT_WIDTH, ENCODE_TYPE, ct)
        
        temp_path = f"build/tmp_gen_{uuid.uuid4().hex}.v"
        os.makedirs("build", exist_ok=True)
        mul_module.emit_verilog(temp_path, assignment=routing_assignment)
        with open(temp_path, "r") as f:
            verilog_contents.append(f.read())
        os.remove(temp_path)
        
    print(f"🚀 [Phase 2] 开始跨服沙盒并发仿真 (Max Workers: {MAX_WORKERS})...")
    results_power = []
    full_results_log = [] 
    logic_fail_count = 0  # 🌟 逻辑失败计数器
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_single_routing, i, v, BIT_WIDTH, TARGET_PERIOD): i for i, v in enumerate(verilog_contents)}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["success"]:
                # 🌟 如果逻辑验证失败，拒绝计入图表，并打印警告
                if res.get("logic_failed"):
                    logic_fail_count += 1
                    print(f"⚠️ [ID: {res['id']:04d}] 综合通过，但逻辑测试 FAILED！(作为残次品废弃)")
                else:
                    results_power.append(res["power_mw"])
                    full_results_log.append({
                        "id": res["id"],
                        "power_mw": res["power_mw"],
                        "area": res["area"],
                        "delay": res["delay"]
                    })
                    print(f"✅ [ID: {res['id']:04d}] 成功! 功耗: {res['power_mw']:.4f} mW | 面积: {res['area']} | 延迟: {res['delay']}")
            else:
                print(f"❌ [ID: {res['id']:04d}] 失败！请检查日志。")
                print(f"--- ERROR LOG ---\n{res['log']}\n-----------------")

    print(f"\n🎉 [Phase 3] 实验完成报告")
    print(f"   ▶ 生成总数: {TOTAL_SAMPLES}")
    print(f"   ▶ 有效且逻辑正确: {len(results_power)}")
    print(f"   ▶ 逻辑验证失败数: {logic_fail_count}")
    
    if len(full_results_log) > 0:
        import csv
        with open(f'routing_sweep_data_{BIT_WIDTH}bit_{ENCODE_TYPE}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "power_mw", "area", "delay"])
            writer.writeheader()
            writer.writerows(full_results_log)
        print(f"💾 所有数值数据已永久保存至: routing_sweep_data_{BIT_WIDTH}bit_{ENCODE_TYPE}.csv")
