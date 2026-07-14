#!/usr/bin/env python3
"""orha 家族 in-context 探针（H1 审计探针方法学，见 OUTER_CELL_SEARCH.md §3.2.6）。

问题：comp42s_orha_* standalone 表征 10.75-14.95µm²（口径 ~4× 膨胀）没过
substd 硬标准，但 ×0.262 校准估真实 ~2.8-3.9 < 5.712。本探针用整网表 DC 直接
测真实边际成本：

  构造 16-bit / k8 截断 / 分级 4:2 压缩树乘法器（exact 槽位 = FA+HA 级联，
  DC 映射原生 FA1D0/HA1D0 = 该槽位的 std 实现），把中列区（col9-20）4 输入
  槽位换成 comp42s cell（cout 线保留为 1'b0，拓扑不变），
  Δarea / n_swap = 每槽真实节省。Δ>0 即证明真实 PPA 优于该槽位的 std 实现。

变体：exact / orha_half / orha_all / or4ao_all。每变体：golden(numpy IR) 校验
（exact 变体误差必须 ≤ 截断上界）、verilator RTL==golden（100k）、
DC+XA @1.5ns（common_eval.measure_ppa，与所有 baseline 同流程）。

用法:
  python -m Appr_Comp.cellsolver.probe_orha_incontext gen   # 生成+本地校验
  python -m Appr_Comp.cellsolver.probe_orha_incontext ppa   # + 远端 DC+XA
产物: outputs/2026-07-12_orha_probe/{probe_*.v, probe_results.json}
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "outputs", "2026-07-12_orha_probe")
N, K = 16, 8               # 位宽 / 截断列数(col1..8)
SWAP_LO, SWAP_HI = 9, 20   # 换 cell 的列区间(中列区)


class Net:
    """IR: 每个 wire 一条 (op, args)；emit 出 verilog，interpret 带缓存求值。"""

    def __init__(self):
        self.ops = {}     # name -> (op, args)
        self.order = []

    def w(self, op, *args):
        name = f"w{len(self.order)}"
        self.ops[name] = (op, args)
        self.order.append(name)
        return name

    def verilog(self, name):
        op, g = self.ops[name]
        if op == "pp":
            return f"a[{g[0]}] & b[{g[1]}]"
        if op == "c0":
            return "1'b0"
        if op == "xor3":
            return f"{g[0]} ^ {g[1]} ^ {g[2]}"
        if op == "maj":
            return f"({g[0]} & {g[1]}) | ({g[2]} & ({g[0]} ^ {g[1]}))"
        if op == "xor2":
            return f"{g[0]} ^ {g[1]}"
        if op == "and2":
            return f"{g[0]} & {g[1]}"
        if op == "orha_s":
            return f"({g[0]} | {g[1]}) ^ ({g[2]} | {g[3]})"
        if op == "orha_c":
            return f"({g[0]} | {g[1]}) & ({g[2]} | {g[3]})"
        if op == "or4":
            return f"{g[0]} | {g[1]} | {g[2]} | {g[3]}"
        if op == "ao22":
            return f"({g[0]} & {g[1]}) | ({g[2]} & {g[3]})"
        raise ValueError(op)

    def eval_all(self, A, B):
        v = {}
        for name in self.order:
            op, g = self.ops[name]
            if op == "pp":
                v[name] = (((A >> g[0]) & 1) & ((B >> g[1]) & 1)).astype(np.uint8)
            elif op == "c0":
                v[name] = np.zeros(A.shape, np.uint8)
            elif op == "xor3":
                v[name] = v[g[0]] ^ v[g[1]] ^ v[g[2]]
            elif op == "maj":
                v[name] = (v[g[0]] & v[g[1]]) | (v[g[2]] & (v[g[0]] ^ v[g[1]]))
            elif op == "xor2":
                v[name] = v[g[0]] ^ v[g[1]]
            elif op == "and2":
                v[name] = v[g[0]] & v[g[1]]
            elif op == "orha_s":
                v[name] = (v[g[0]] | v[g[1]]) ^ (v[g[2]] | v[g[3]])
            elif op == "orha_c":
                v[name] = (v[g[0]] | v[g[1]]) & (v[g[2]] | v[g[3]])
            elif op == "or4":
                v[name] = v[g[0]] | v[g[1]] | v[g[2]] | v[g[3]]
            elif op == "ao22":
                v[name] = (v[g[0]] & v[g[1]]) | (v[g[2]] & v[g[3]])
        return v


def build(swap_kind=None, swap_frac=1.0):
    """分级 4:2 归约。返回 (net, final_rows[(col,wire)], n_swap)。
    swap_kind=None → 全 exact；否则中列区 4 输入槽位按 frac 换 cell。"""
    net = Net()
    cols = {c: [] for c in range(1, 2 * N + 3)}
    for i in range(N):
        for j in range(N):
            c = i + j + 1
            if c > K:
                cols[c].append(net.w("pp", i, j))
    n_seen = n_swap = 0
    stage = 0
    while any(len(v) > 2 for v in cols.values()):
        stage += 1
        assert stage < 12
        nxt = {c: [] for c in cols}
        for c in sorted(cols):
            bits = cols[c]
            while len(bits) >= 4:
                g, bits = bits[:4], bits[4:]
                if swap_kind and SWAP_LO <= c <= SWAP_HI:
                    n_seen += 1
                    if swap_frac >= 1.0 or n_seen % 2 == 1:
                        n_swap += 1
                        if swap_kind == "orha":
                            s = net.w("orha_s", *g)
                            c1 = net.w("orha_c", *g)
                        else:
                            s = net.w("or4", *g)
                            c1 = net.w("ao22", *g)
                        nxt[c].append(s)
                        nxt[c + 1].append(c1)
                        nxt[c + 1].append(net.w("c0"))  # cout≡0, 拓扑不变
                        continue
                # exact 槽位: FA(g0,g1,g2) + HA(s1,g3) → DC 映射 FA1D0+HA1D0
                s1 = net.w("xor3", g[0], g[1], g[2])
                c1 = net.w("maj", g[0], g[1], g[2])
                s = net.w("xor2", s1, g[3])
                c2 = net.w("and2", s1, g[3])
                nxt[c].append(s)
                nxt[c + 1].append(c1)
                nxt[c + 1].append(c2)
            if len(bits) == 3:
                s = net.w("xor3", *bits)
                c1 = net.w("maj", *bits)
                nxt[c].append(s)
                nxt[c + 1].append(c1)
            else:
                nxt[c].extend(bits)
        cols = nxt
    rows = [(c, w) for c in sorted(cols) for w in cols[c]]
    return net, rows, n_swap


def emit(name, swap_kind=None, swap_frac=1.0):
    net, rows, n_swap = build(swap_kind, swap_frac)
    body = "\n".join(f"    wire {w} = {net.verilog(w)};" for w in net.order)
    terms = " + ".join(f"({{16'b0, {w}}} << {c - 1})" for c, w in rows)
    v = f"""// probe {name}: 16b k{K} staged 4:2 tree, swap={swap_kind} frac={swap_frac} n_swap={n_swap}
module MUL(
    input wire clk,
    input wire [15:0] a,
    input wire [15:0] b,
    output wire [30:0] out
);
{body}
    wire [16:0] _z = 17'b0;
    wire [32:0] full = {terms};
    assign out = full[30:0];
endmodule
"""
    path = os.path.join(OUT, f"probe_{name}.v")
    with open(path, "w") as f:
        f.write(v)

    def golden(A, B):
        vals = net.eval_all(A, B)
        acc = np.zeros(A.shape, dtype=np.int64)
        for c, w in rows:
            acc += vals[w].astype(np.int64) << (c - 1)
        return acc
    return path, golden, n_swap


VARIANTS = [("exact", None, 1.0), ("orha_half", "orha", 0.5),
            ("orha_all", "orha", 1.0), ("or4ao_all", "or4ao", 1.0)]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    A = rng.integers(0, 1 << N, 400_000, dtype=np.uint32)
    B = rng.integers(0, 1 << N, 400_000, dtype=np.uint32)
    exact_p = A.astype(np.int64) * B.astype(np.int64)
    meta = {}
    for name, kind, frac in VARIANTS:
        path, golden, n_swap = emit(name, kind, frac)
        g = golden(A, B)
        err = np.abs(g - exact_p)
        med = float(err.mean())
        nz = exact_p != 0
        mred = float((err[nz] / exact_p[nz]).mean())
        if name == "exact":
            trunc_max = sum(min(c, 32 - c) << (c - 1) for c in range(1, K + 1))
            assert float(err.max()) <= trunc_max, "exact 变体超截断上界!"
        meta[name] = {"n_swap": n_swap, "med_400k": med, "mred_400k": mred, "rtl": path}
        print(f"{name:<12} n_swap={n_swap:<3} MED(400k)={med:.1f} MRED={mred:.5f}")
        vec = os.path.join(OUT, f"_vec_{name}.txt")
        with open(vec, "w") as f:
            for i in range(100_000):
                f.write(f"{A[i]} {B[i]} {int(g[i]) & 0x7FFFFFFF}\n")
        objd = os.path.join(OUT, f"_obj_{name}")
        tb = "/home/lee/Baselines/Sayadi_TCSI23_NewConfig42/tb_check.cpp"
        r = subprocess.run(
            ["verilator", "--cc", "--exe", "--build", "-j", "4", "-O3", "-Wno-fatal",
             "--top-module", "MUL", "--Mdir", objd, path, tb, "-o", "sim"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        r = subprocess.run([os.path.join(objd, "sim"), vec], capture_output=True, text=True)
        print("   " + r.stdout.strip().splitlines()[-1])
        assert "PASS" in r.stdout, name
    with open(os.path.join(OUT, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if mode != "ppa":
        return

    sys.path.insert(0, "/home/lee/Baselines")
    from common_eval import measure_ppa
    rp = os.path.join(OUT, "probe_results.json")
    results = json.load(open(rp)) if os.path.exists(rp) else {}
    for idx, (name, _k, _f) in enumerate(VARIANTS):
        if results.get(name, {}).get("success"):
            continue
        print(f"[{name}] DC+XA @1.5ns ...", flush=True)
        p = measure_ppa(open(meta[name]["rtl"]).read(), idx=idx, target_delay=1.5)
        results[name] = {**{k: p[k] for k in ("area", "power_mw", "delay", "success")},
                         "n_swap": meta[name]["n_swap"],
                         "med_400k": meta[name]["med_400k"],
                         "mred_400k": meta[name]["mred_400k"]}
        print(f"[{name}] area={p['area']} power={p['power_mw']} delay={p['delay']}")
        with open(rp, "w") as f:
            json.dump(results, f, indent=1)
    if all(results.get(n, {}).get("success") for n, _, _ in VARIANTS):
        ae = results["exact"]["area"]
        pe = results["exact"]["power_mw"]
        print("\n=== marginal analysis (exact FA+HA slot ≈ 2.856+2.184 = 5.04 LEF) ===")
        for name in ("orha_half", "orha_all", "or4ao_all"):
            r = results[name]
            n = r["n_swap"]
            da = (ae - r["area"]) / n
            dp = (pe - r["power_mw"]) / n * 1e3
            print(f"{name:<12} n={n:<3} saving/slot: area {da:+.2f} um2, power {dp:+.2f} uW"
                  f" | implied real cell area ~ {5.04 - da:.2f} um2 (substd bar 5.712)")


if __name__ == "__main__":
    main()
