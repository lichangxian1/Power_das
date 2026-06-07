"""
随机生成 N 个 16-bit 乘法器网表，分别用本地 abc 和远端 DC 综合，比较面积差异。

用法：
    python scripts/compare_abc_dc.py [--n 8] [--target_delay 2.0] [--workers 4] [--keep]
"""

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import get_initial_partial_product, CompressorTree, Mul
from trainer.arith_das import CompressorGraph
from run_power_sweep import VerilogEmitter, generate_legal_random_routing, evaluate_single_routing


# ── 本地 abc 综合（yosys + openroad STA） ──────────────────────────────────────

def abc_synthesize(verilog_content: str, target_delay: float, work_root: str, idx: int, keep: bool):
    """调用本地 yosys+abc+openroad 综合，返回 dict(area, delay, power) 或 None。"""
    from utils.mul import Mul as MulCls

    worker_path = os.path.join(work_root, f"abc_worker_{idx}_{uuid.uuid4().hex[:4]}")
    rtl_path = os.path.join(worker_path, "design.v")
    os.makedirs(worker_path, exist_ok=True)

    with open(rtl_path, "w") as f:
        f.write(verilog_content)

    try:
        result = MulCls.simulate_worker(
            worker_path=worker_path,
            rtl_path=rtl_path,
            target_delay=target_delay,
            worker_id=idx,
            keep_files=keep,
        )
        return {"area": result["area"], "delay": result["delay"], "power": result["power"]}
    except Exception as e:
        print(f"  [abc #{idx}] 失败: {e}")
        return None
    finally:
        if not keep:
            import shutil
            if os.path.exists(worker_path):
                shutil.rmtree(worker_path, ignore_errors=True)


# ── 远端 DC 综合 ───────────────────────────────────────────────────────────────

def dc_synthesize(verilog_content: str, target_delay: float, idx: int):
    """发送到远端 EDA 服务器运行 DC 综合，返回 dict(area, delay, power_mw) 或 None。"""
    result = evaluate_single_routing(
        idx=idx,
        verilog_content=verilog_content,
        bit_width=16,
        target_delay=target_delay,
    )
    if result.get("success") and not result.get("logic_failed"):
        return {
            "area":  result["area"],
            "delay": abs(result.get("delay", float("inf"))),
            "power": result.get("power_mw", float("inf")),
        }
    print(f"  [dc  #{idx}] 失败: {result.get('log', '')[:120]}")
    return None


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",            type=int,   default=8,   help="生成的网表数量")
    parser.add_argument("--target_delay", type=float, default=2000, help="abc 目标延迟 (ps)，DC 目标延迟 (ns) 由 run_power_sweep 独立控制")
    parser.add_argument("--workers",      type=int,   default=4,   help="并发 worker 数")
    parser.add_argument("--keep",         action="store_true",     help="保留中间文件")
    parser.add_argument("--out",          default="outputs/abc_dc_compare.json")
    args = parser.parse_args()

    # ── 1. 生成随机网表 ──
    print(f"\n{'='*60}")
    print(f"Step 1: 生成 {args.n} 个随机 16-bit 网表")
    print(f"{'='*60}")

    pp         = get_initial_partial_product(16, "and")
    ct         = CompressorTree.dadda(pp)
    assignment = ct.compressor_assignment_fused()
    comp_graph = CompressorGraph(pp, assignment)
    emitter    = VerilogEmitter(comp_graph)
    mul        = Mul(16, "and", ct)

    work_root = os.path.join(_REPO_ROOT, "outputs", "abc_dc_compare_tmp")
    os.makedirs(work_root, exist_ok=True)

    netlists = []
    for i in range(args.n):
        connections      = generate_legal_random_routing(comp_graph)
        routing_dict     = emitter.emit_assignment(connections)
        tmp_path         = os.path.join(work_root, f"netlist_{i}.v")
        mul.emit_verilog(rtl_path=tmp_path, assignment=routing_dict)
        with open(tmp_path) as f:
            netlists.append(f.read())
        print(f"  生成网表 #{i}: {tmp_path}")

    # ── 2. 本地 abc 综合（并发） ──
    print(f"\n{'='*60}")
    print(f"Step 2: 本地 abc 综合 (yosys + openroad STA)")
    print(f"{'='*60}")

    abc_results = [None] * args.n
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(abc_synthesize, nl, args.target_delay, work_root, i, args.keep): i
            for i, nl in enumerate(netlists)
        }
        for fut in concurrent.futures.as_completed(futs):
            i   = futs[fut]
            res = fut.result()
            abc_results[i] = res
            status = f"area={res['area']:.2f}  delay={res['delay']:.4f}  power={res['power']:.6f}" if res else "FAILED"
            print(f"  abc #{i}: {status}")

    # ── 3. 远端 DC 综合（并发） ──
    print(f"\n{'='*60}")
    print(f"Step 3: 远端 DC 综合 (ssh → {__import__('run_power_sweep').EDA_HOST})")
    print(f"{'='*60}")

    dc_results = [None] * args.n
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(dc_synthesize, nl, args.target_delay, i): i
            for i, nl in enumerate(netlists)
        }
        for fut in concurrent.futures.as_completed(futs):
            i   = futs[fut]
            res = fut.result()
            dc_results[i] = res
            status = f"area={res['area']:.2f}  delay={res['delay']:.4f}  power={res['power']:.4f}mW" if res else "FAILED"
            print(f"  dc  #{i}: {status}")

    # ── 4. 比较 ──
    print(f"\n{'='*60}")
    print(f"Step 4: 面积对比 (abc vs DC)")
    print(f"{'='*60}")
    print(f"{'#':>3}  {'abc area':>10}  {'dc area':>10}  {'diff':>10}  {'diff%':>8}")
    print(f"{'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    records = []
    diffs = []
    for i in range(args.n):
        a = abc_results[i]
        d = dc_results[i]
        if a and d:
            diff    = d["area"] - a["area"]
            diff_pct = diff / a["area"] * 100 if a["area"] else float("nan")
            diffs.append(diff_pct)
            print(f"{i:>3}  {a['area']:>10.2f}  {d['area']:>10.2f}  {diff:>+10.2f}  {diff_pct:>+7.2f}%")
        else:
            print(f"{i:>3}  {'N/A':>10}  {'N/A':>10}  {'—':>10}  {'—':>8}")
        records.append({
            "id":       i,
            "abc":      a,
            "dc":       d,
            "area_diff":      (d["area"] - a["area"]) if a and d else None,
            "area_diff_pct":  ((d["area"] - a["area"]) / a["area"] * 100) if a and d and a["area"] else None,
        })

    if diffs:
        import statistics
        print(f"\n  有效样本数: {len(diffs)}/{args.n}")
        print(f"  平均面积差: {statistics.mean(diffs):+.2f}%")
        print(f"  中位数面积差: {statistics.median(diffs):+.2f}%")
        print(f"  标准差: {statistics.stdev(diffs):.2f}%" if len(diffs) > 1 else "")

    # ── 5. 保存结果 ──
    out_path = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"records": records, "target_delay": args.target_delay}, f, indent=2)
    print(f"\n  结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
