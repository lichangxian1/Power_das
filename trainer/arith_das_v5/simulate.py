"""评估执行：远端 DC 综合直出 PPA 的并行 worker、verilator 实测误差、
结果汇总。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
import sys
import logging


from paretoset import paretoset
import numpy as np

from utils import (
    CompressorTree,
    Mul,
)


class SimulateMixin:
    """远端 DC 综合、verilator 实测误差与结果汇总。"""

    def get_full_target_delay_result(self):
        """save/export 期的 full-PPA 诊断。v5 固定单一 DC 周期（fixed_target_delay
        由构造守卫保证非 None），只评这 1 个 target_delay；synth 恒为 "dc"。"""
        from .core import CompressorRouting  # 延迟导入：避免循环依赖，且保持按类名查找（可 monkeypatch）
        build_dir = self.build_dir + "_full_ppa"
        rtl_path = os.path.join(build_dir, "MUL.v")
        full_target_delay = [self.fixed_target_delay]
        os.makedirs(build_dir, exist_ok=True)

        ct = CompressorTree(
            self.initial_pp,
            self.state["ct32"],
            self.state["ct22"],
            self.state.get("ct42"),
        )
        if self.trunc_cols > 0:               # ① full-target-delay 诊断/导出也必须带截断
            ct.trunc_cols = self.trunc_cols
            ct.trunc_bits = self._trunc_bits
        mul = Mul(self.bit_width, self.encode_type, ct)

        # Phase B：用最优设计的近似 cell 评 full-target-delay PPA（否则退化成精确）
        cell_map = self._cell_map_from_types(self.found_best_info.get("cell_types"))
        assignment = self.emit_assignment(
            self.found_best_info["connection"], cell_map=cell_map
        )
        mul.emit_verilog(
            rtl_path,
            assignment=assignment,
            extra_modules_src=self._approx_modules_src(cell_map),
        )
        # 与训练奖励同源：full-target-delay 诊断也走远端 DC 直出。
        # r2 审查 #1：远端无 yosys/openroad（已 ssh 实证），原 openroad 回退一触即崩
        # 且口径污染（openroad 数值与 DC 差 ~20×）——DC 失败的 td 直接跳过，
        # 全部失败抛给 save_experiment 的诊断隔离层（诊断可弃，训练不可崩）。
        simulated_result = []
        for td in full_target_delay:
            one = CompressorRouting._dc_simulate_one(
                self.bit_width, rtl_path, build_dir, td, 0
            )
            if one is None:
                logging.warning("[full-ppa] 远端 DC 失败 td=%s，跳过该点（不回退 openroad）", td)
                continue
            simulated_result.append(one)
        if not simulated_result:
            raise RuntimeError("full-PPA 诊断：全部 target_delay 的远端 DC 均失败")
        simulated_result = self._apply_power_proxy_to_results(
            simulated_result,
            self.found_best_info["connection"],
        )
        return simulated_result

    def get_full_target_delay_pareto(self, simulated_result, target=["delay", "power"]):
        value_0_list = [item[target[0]] for item in simulated_result]
        value_1_list = [item[target[1]] for item in simulated_result]

        points = np.asarray(list(zip(value_0_list, value_1_list)))
        pareto_indices = paretoset(points, sense=["min", "min"])
        pareto_points = points[pareto_indices]
        return pareto_points

    @staticmethod
    def parallel_simulate_worker(
        bit_width,
        encode_type,
        ct,
        rtl_path,
        build_path,
        target_delay,
        id,
        target_delay_id,
        synth,
        error_gate="analytic",
        error_gate_vectors=16_000_000,
    ):
        from .core import CompressorRouting  # 延迟导入：避免循环依赖，且保持按类名查找（可 monkeypatch）
        if synth != "dc":
            raise NotImplementedError("arith_das_v5 只支持 synth='dc'")
        # 远端 DC 直出 PPA（功耗取 DC report_power，不走 VCS/XA）
        one = CompressorRouting._dc_simulate_one(
            bit_width, rtl_path, build_path, target_delay, id
        )
        if one is not None:
            simulated_result = [one]
        else:
            # P0(codex)：远端 DC 多次重试仍失败 → **不**回退本地 ABC（量纲差 ~20×，
            # 混进同一 PPO batch 会污染梯度/best；正是断网那次的故障）。标记失败，上层踢出本批。
            logging.warning(
                f"[dc] worker {id} remote DC failed → 丢弃该样本(不混 ABC): {rtl_path}"
            )
            return {
                "result": None,
                "failed": True,
                "id": id,
                "target_delay_id": target_delay_id,
                "target_delay": target_delay,
            }
        # 误差闸门：DC/综合成功后，并行在本 worker 测 verilator 真实 MED（失败=None，不丢样本）。
        measured_error = None
        if error_gate == "verilator":
            measured_error = CompressorRouting._measure_error_verilator(
                rtl_path, build_path, error_gate_vectors
            )
        return {
            "result": simulated_result,
            "measured_error": measured_error,
            "id": id,
            "target_delay_id": target_delay_id,
            "target_delay": target_delay,
        }

    @staticmethod
    def _dc_simulate_one(bit_width, rtl_path, build_path, target_delay, worker_id):
        """远端 DC 直出 PPA，返回与本地 simulate_worker 同构的 dict（power 转为 W），
        失败返回 None。使用专用 base 副本 sandbox_base_dcpwr（默认 POWER_MODE=dc，
        跳过 v2lvs/SPICE/VCS/XA，area/delay/power 全部取自 DC 综合）。"""
        repo_root = os.path.dirname(os.path.dirname(  # 包内比原单文件多一层
            os.path.dirname(os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import run_power_sweep as rps

        # 指向 DC 直出专用 base 副本；原 sandbox_base 不受影响。
        rps.EDA_BASE_DIR = os.environ.get(
            "EDA_BASE_DIR_DC", "/home/lchangxian/sandbox/sandbox_base_dcpwr"
        )
        # evaluate_single_routing 在 cwd 下写 build/ 临时文件；切到可写目录避开 build 符号链接坑
        os.makedirs(build_path, exist_ok=True)
        try:
            os.chdir(build_path)
        except OSError:
            pass
        with open(rtl_path) as f:
            rtl_src = f.read()
        r = rps.evaluate_single_routing(
            worker_id, rtl_src, bit_width=bit_width, target_delay=target_delay
        )
        if (
            not r
            or not r.get("success")
            or r.get("area") is None
            or r.get("power_mw") is None
        ):
            return None
        # 07-11 deepk k28 事故：DC 报告偶发解析成 area=0.0/delay=0.1，0.0 过得了 None 检查
        # → 荒谬 objective 抢占 best 并冻结其后所有轮。物理下界兜底（16-bit 乘法器不可能 <10µm²）。
        if float(r["area"]) <= 10.0 or float(r["power_mw"]) <= 0.0:
            logging.warning(
                f"[dc] worker {worker_id} 异常 PPA 读数 area={r.get('area')} "
                f"power={r.get('power_mw')} → 按失败丢弃"
            )
            return None
        delay = r.get("delay")
        if delay is None:
            delay = target_delay
        # evaluate_single_routing 的 delay 约定为负（关键路径到达时间），取绝对值得正向延时
        return {
            "delay": abs(float(delay)),
            "area": float(r["area"]),
            "power": float(r["power_mw"]) / 1000.0,  # mW → W，对齐本地 simulate_worker
            "target_delay": target_delay,
            "worker_id": worker_id,
        }

    @staticmethod
    def _measure_error_verilator(rtl_path, build_path, n_vectors):
        """误差闸门：verilator MC 实测 circular-wrap 真实误差（codex 审过的接入）。
        返回 dict(med, bias, wce_mc, source="verilator") 或 None（编译/运行/解析失败）。
        - 每次用全新 obj 目录（绝对路径；_dc_simulate_one 改过 cwd 不恢复，故全部绝对化）。
        - verilator --build -j1（8 worker 同跑时避免 make 多核过订阅）。
        - 失败重试 1 次；仍失败返回 None → 上层回退解析闸门（不丢该样本，别浪费 DC）。
        WCE 只上报不当闸门（MC 尾部不收敛）。"""
        import shutil
        import subprocess

        repo_root = os.path.dirname(os.path.dirname(  # 包内比原单文件多一层
            os.path.dirname(os.path.abspath(__file__))))
        harness = os.path.join(repo_root, "verilate", "mul_err_wrap.cpp")
        rtl_abs = os.path.abspath(rtl_path)
        for attempt in range(2):
            verr = os.path.abspath(os.path.join(build_path, f"verr_{attempt}"))
            try:
                shutil.rmtree(verr, ignore_errors=True)
                os.makedirs(verr, exist_ok=True)
                obj = os.path.join(verr, "obj_dir")
                exe = os.path.join(obj, "mul_err")
                bcmd = ["verilator", "--cc", "--exe", "--build", "-j", "1", "-O3",
                        "-Wno-fatal", "--top-module", "MUL", "--Mdir", obj,
                        rtl_abs, harness, "-o", "mul_err"]
                b = subprocess.run(bcmd, cwd=verr, capture_output=True, text=True, timeout=180)
                if b.returncode != 0 or not os.path.exists(exe):
                    raise RuntimeError(f"verilator build rc={b.returncode}")
                r = subprocess.run([exe, str(int(n_vectors))], cwd=verr,
                                   capture_output=True, text=True, timeout=120)
                if r.returncode != 0:
                    raise RuntimeError(f"verilator run rc={r.returncode}")
                med = bias = wce = mred = None
                for line in r.stdout.strip().splitlines():
                    p = line.split(",")
                    if p[0] == "masked":
                        med, bias, wce = float(p[1]), float(p[2]), float(p[5])
                        if len(p) > 6:           # MRED 为 harness 新增的第 7 字段（向后兼容）
                            mred = float(p[6])
                        break
                if med is None:
                    raise RuntimeError("no masked line")
                shutil.rmtree(verr, ignore_errors=True)
                return {"med": med, "bias": bias, "wce_mc": wce,
                        "mred": mred, "source": "verilator"}
            except Exception as e:  # noqa: BLE001
                logging.warning("[errgate] verilator measure attempt %d failed (%s): %s",
                                attempt, os.path.basename(rtl_path), e)
                shutil.rmtree(verr, ignore_errors=True)
        return None

    def _apply_power_proxy_to_results(self, simulated_result, samples_connection):
        """v5 固定 power_source='eda'：只补齐审计字段（power 即 DC 实测值）。
        proxy 改写分支已随 power proxy 一并剪除；保留原函数名以少改调用点。"""
        for item in simulated_result:
            item["eda_power"] = item.get("power")
            item["proxy_power_mw"] = None
            item["power_source"] = "eda"
        return simulated_result

    def _summarize_result(self, simulated_result):
        delay = float(np.mean([item["delay"] for item in simulated_result]))
        area = float(np.mean([item["area"] for item in simulated_result]))
        power = float(np.mean([item["power"] for item in simulated_result]))
        eda_power_values = [item.get("eda_power") for item in simulated_result]
        eda_power = None
        if all(value is not None for value in eda_power_values):
            eda_power = float(np.mean(eda_power_values))
        proxy_values = [item.get("proxy_power_mw") for item in simulated_result]
        proxy_power_mw = None
        if all(value is not None for value in proxy_values):
            proxy_power_mw = float(np.mean(proxy_values))

        area_violation = 0.0
        area_feasible = True
        if self.area_budget is not None:
            area_violation = max(0.0, area - float(self.area_budget))
            area_feasible = area_violation <= 0.0

        delay_violation = 0.0
        delay_feasible = True
        if self.fixed_target_delay is not None:
            delay_violation = max(0.0, delay - float(self.fixed_target_delay))
            delay_feasible = delay_violation <= 0.0

        return {
            "delay": delay,
            "area": area,
            "power": power,
            "eda_power": eda_power,
            "proxy_power_mw": proxy_power_mw,
            "area_budget": self.area_budget,
            "fixed_target_delay": self.fixed_target_delay,
            "area_violation": area_violation,
            "delay_violation": delay_violation,
            "area_feasible": area_feasible,
            "delay_feasible": delay_feasible,
            "power_source": self.power_source,
        }
