"""精确真值表张量化压缩树仿真器。

数据源与 RTL 发射同源：树连线复刻 `CompressorRouting.emit_assignment` 的 connect 语义
（connection 边 + comp_graph.vertex_list），部分积接线不重刻 Booth/AND 索引公式，
直接解析 `Mul.emit_pp_encoder()` 的机器生成文本（截断常数 C 已内联其中）。

两种前向：
  eval_exact  int64/bool 无梯度，逐位精确，chunked，可对拍 verilator（同随机流下
              med/bias 必须整数级一致）。
  eval_diff   多线性形式（每 cell 输出 = Σ_pattern TT[p]·Π basis(p)），hard 0/1 输入
              下前向值精确 == 离散电路，反向给出多线性偏导（STE：相关性在前向保真，
              偏差只在梯度）。树内 float32（bit 值恒精确 0/1），列加权和转 float64
              保证 2^31 量级整数精确。

随机流复刻 verilate/mul_err_wrap.cpp：xorshift128+，SEED=12345，a=r&0xFFFF，
b=(r>>16)&0xFFFF —— 与 verilator 是 common random numbers（前缀抽样）。
误差口径同 harness：out/golden 均 mask 31 位，e 做 circular-wrap 到 [-2^30,2^30)，
MRED 只在 golden!=0 上取 |e|/golden 均值。
"""
import os
import re

import numpy as np
import torch

MASK31 = 0x7FFFFFFF
INPUT_PORTS = ["a", "b", "c", "d"]


# ---------------------------------------------------------------- 随机流
def xorshift_ab(n, seed=12345, cache_dir=None):
    """复刻 mul_err_wrap.cpp 的 xorshift128+ 流，返回 (a, b) uint16 数组。
    递推是串行的（无法向量化），用 python 大整数逐步生成；cache_dir 给定时
    结果缓存为 npz（小 N 是大 N 的前缀 —— common random numbers）。"""
    if cache_dir:
        path = os.path.join(cache_dir, f"xs128p_seed{seed}_{n}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return z["a"], z["b"]
    M64 = (1 << 64) - 1
    s0 = (0x9E3779B97F4A7C15 ^ seed) & M64
    s1 = 0xD1B54A32D192ED03
    a = np.empty(n, dtype=np.uint16)
    b = np.empty(n, dtype=np.uint16)
    for i in range(n):
        x, y = s0, s1
        s0 = y
        x ^= (x << 23) & M64
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        r = (s1 + y) & M64
        a[i] = r & 0xFFFF
        b[i] = (r >> 16) & 0xFFFF
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, a=a, b=b)
    return a, b


# ---------------------------------------------------------------- pp 解析
_PP_PATTERNS = [
    ("and", re.compile(r"^\s*assign pp_(\d+)\[(\d+)\] = a\[(\d+)\] & b\[(\d+)\];")),
    ("const", re.compile(r"^\s*assign pp_(\d+)\[(\d+)\] = 1'b([01]);")),
    ("enc", re.compile(r"^\s*assign pp_(\d+)\[(\d+)\] = refined_y_(\d+)\[(\d+)\];")),
    ("nsgn", re.compile(r"^\s*assign pp_(\d+)\[(\d+)\] = ~sgn_(\d+);")),
    ("sgn", re.compile(r"^\s*assign pp_(\d+)\[(\d+)\] = sgn_(\d+);")),
]


def parse_pp_specs(pp_src):
    """emit_pp_encoder() 文本 → {(col, idx): spec}。spec:
    ("and", xa, xb) | ("const", v) | ("enc", e, bit) | ("sgn", e) | ("nsgn", e)"""
    specs = {}
    for line in pp_src.split("\n"):
        for kind, pat in _PP_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            col, idx = int(m.group(1)), int(m.group(2))
            if kind == "and":
                specs[(col, idx)] = ("and", int(m.group(3)), int(m.group(4)))
            elif kind == "const":
                specs[(col, idx)] = ("const", int(m.group(3)))
            elif kind == "enc":
                specs[(col, idx)] = ("enc", int(m.group(3)), int(m.group(4)))
            else:
                specs[(col, idx)] = (kind, int(m.group(3)))
            break
    return specs


def compute_pp_bits(specs, a_np, b_np, bit_width, device="cpu"):
    """按 spec 批量算 pp 位 → {(col,idx): torch.float32 [N]（值恒 0/1）}。
    booth 的 refined_y/sgn 按 BoothEncoder 模块语义向量化。"""
    a = torch.from_numpy(a_np.astype(np.int64)).to(device)
    b = torch.from_numpy(b_np.astype(np.int64)).to(device)
    n = a.shape[0]
    need_enc = sorted({s[1] for s in specs.values() if s[0] in ("enc", "sgn", "nsgn")})
    refined, sgn = {}, {}
    if need_enc:
        shifted_x = a << 1  # {2'b0, a, 1'b0}
        for e in need_enc:
            ci = 2 * e + 1
            x0 = (shifted_x >> (ci - 1)) & 1
            x1 = (shifted_x >> ci) & 1
            x2 = (shifted_x >> (ci + 1)) & 1
            single = x0 ^ x1
            double = (x0 & x1 & (1 - x2)) | ((1 - x0) & (1 - x1) & x2)
            neg = x2
            bits = []
            for i in range(bit_width + 1):
                y_ext = (b >> i) & 1 if i < bit_width else torch.zeros_like(b)
                y_shift = (b >> (i - 1)) & 1 if i >= 1 else torch.zeros_like(b)
                bits.append(neg ^ ((single & y_ext) | (double & y_shift)))
            refined[e] = bits
            sgn[e] = neg
    out = {}
    for (col, idx), spec in specs.items():
        kind = spec[0]
        if kind == "and":
            v = ((a >> spec[1]) & 1) & ((b >> spec[2]) & 1)
        elif kind == "const":
            v = torch.full((n,), spec[1], dtype=torch.int64, device=device)
        elif kind == "enc":
            v = refined[spec[1]][spec[2]]
        elif kind == "sgn":
            v = sgn[spec[1]]
        else:  # nsgn
            v = 1 - sgn[spec[1]]
        out[(col, idx)] = v.to(torch.float32)
    return out


# ---------------------------------------------------------------- 精确 cell 真值表
def _lut(vals):
    return np.asarray(vals, dtype=np.int64)


def exact_luts(t, has_carry=True):
    """节点端口序索引（port a 为最高位）的精确真值表。
    ct32: idx=a*4+b*2+c；ct22: idx=a*2+b；ct42: idx=a*8+b*4+c*2+d。"""
    if t == 0:
        idx = np.arange(8)
        s = ((idx >> 2) + ((idx >> 1) & 1) + (idx & 1))
        return {"n_in": 3, "sum": _lut(s & 1),
                "carry": _lut((s >= 2).astype(int)) if has_carry else None}
    if t == 1:
        idx = np.arange(4)
        s = (idx >> 1) + (idx & 1)
        return {"n_in": 2, "sum": _lut(s & 1),
                "carry": _lut((s >= 2).astype(int)) if has_carry else None}
    if t == 4:
        # CT42 平衡树：w=a^b, x=c^d, sum=w^x, cout=(w&c)|(~w&a), carry=(w^c)&d
        idx = np.arange(16)
        a = (idx >> 3) & 1
        b = (idx >> 2) & 1
        c = (idx >> 1) & 1
        d = idx & 1
        w = a ^ b
        x = c ^ d
        return {"n_in": 4, "sum": _lut(w ^ x),
                "carry": _lut((w ^ c) & d),
                "cout": _lut((w & c) | ((1 - w) & a))}
    raise ValueError(t)


def approx_luts_from_lib(t, cell):
    """library.json / library42_native.json 条目 → 端口序索引 LUT。
    comp32/22：sum_lut/carry_lut 顺序即 product('01')，直接用。
    comp42：patterns 串（bits[:4]=a,b,c,d，5 位则 bits[-1]=cin，仅取 cin=0 行）。"""
    if t in (0, 1):
        return {"n_in": 3 if t == 0 else 2,
                "sum": _lut(cell["sum_lut"]), "carry": _lut(cell["carry_lut"])}
    assert t == 4
    pats = cell.get("patterns")
    if pats is None:
        # 原生 4 输入库（library42_native.json）：pattern_bits=4，LUT 即 product('01')
        # 顺序（a 为最高位），与 SOP 发射同约定，直接用。
        assert int(cell.get("pattern_bits", 4)) == 4
        return {"n_in": 4, "sum": _lut(cell["sum_lut"]),
                "carry": _lut(cell["carry_lut"]), "cout": _lut(cell["cout_lut"])}
    s = np.zeros(16, dtype=np.int64)
    ca = np.zeros(16, dtype=np.int64)
    co = np.zeros(16, dtype=np.int64)
    for i, bits in enumerate(pats):
        if len(bits) == 5 and bits[-1] != "0":
            continue
        j = int(bits[:4], 2)
        s[j] = int(cell["sum_lut"][i])
        ca[j] = int(cell["carry_lut"][i])
        co[j] = int(cell["cout_lut"][i])
    return {"n_in": 4, "sum": s, "carry": ca, "cout": co}


# ---------------------------------------------------------------- 树仿真
class TreeSim:
    """从 (comp_graph, connection, pp_specs) 构建的张量化仿真器。

    node_plan: 按 stage 排序的节点求值计划。每项:
      (node_idx, kind, t, col, in_refs, out_names)
      in_refs = [(src_idx, out_name), ...] 端口序；pp/visual 特殊处理。
    """

    def __init__(self, comp_graph, connection, pp_specs, device="cpu"):
        self.device = device
        self.pp_specs = pp_specs
        vlist = comp_graph.vertex_list
        self.col_num = comp_graph.col_num
        stage_num = comp_graph.stage_num

        # 复刻 emit_assignment 的 connect：node → 端口来源
        frm = {}   # node_idx -> {port: (src_idx, out_name)}
        used = set()
        node_type = {}
        for src, dst, port_t, meta in connection:
            src, dst, port_t = int(src), int(dst), int(port_t)
            meta = meta or {}
            s_st, s_c, s_t, _ = vlist[src]
            d_st, d_c, d_t, _ = vlist[dst]
            node_type[src] = s_t
            node_type[dst] = d_t
            assert s_st + 1 == d_st, "connection 必须跨相邻 stage"
            if s_c == d_c:
                out_name = meta.get("src_output", "sum")
                assert out_name == "sum"
            elif s_c + 1 == d_c:
                out_name = meta.get("src_output", "carry")
            else:
                raise ValueError(f"invalid edge {src}->{dst}")
            key = (src, out_name)
            assert key not in used, f"output {key} routed twice"
            used.add(key)
            port = INPUT_PORTS[port_t]
            frm.setdefault(dst, {})
            assert port not in frm[dst], f"input {port} of {dst} routed twice"
            frm[dst][port] = (src, out_name)

        n_ports = {0: 3, 1: 2, 2: 0, 3: 1, 4: 4}
        plan = []
        self.final_wires = [[] for _ in range(self.col_num)]
        for node in sorted(frm.keys() | {s for s, _ in used}):
            st, col, t, idx = vlist[node]
            if t == 2:
                plan.append((node, "pp", t, col, (col, idx), None))
                continue
            ports = INPUT_PORTS[: n_ports[t]]
            in_refs = [frm[node][p] for p in ports]
            has_carry = (node, "carry") in used
            plan.append((node, "cell" if t in (0, 1, 4) else "visual",
                         t, col, in_refs, has_carry))
        # 末段 visual（stage==stage_num）即 routed_wire_list
        for node in sorted(frm.keys() | {s for s, _ in used}):
            st, col, t, idx = vlist[node]
            if t == 3 and st == stage_num:
                self.final_wires[col].append(node)
        for col, ws in enumerate(self.final_wires):
            assert len(ws) <= 2, f"col {col} 剩余 {len(ws)} 根线（应 ≤2）"
        # stage 序求值（pp 在最前）
        plan.sort(key=lambda e: (vlist[e[0]][0], e[0]))
        self.plan = plan
        self.vlist = vlist

    # ------------------------------------------------------------ 精确整数模式
    def eval_exact(self, pp_bits, cell_luts):
        """pp_bits: {(col,idx): float/int tensor}; cell_luts: {node_idx: luts}（缺省精确）。
        返回 out int64 [N]（未 mask）。"""
        dev = self.device
        val = {}
        for node, kind, t, col, ref, aux in self.plan:
            if kind == "pp":
                val[(node, "sum")] = pp_bits[ref].to(torch.int64)
            elif kind == "visual":
                (src, oname), = ref
                val[(node, "sum")] = val[(src, oname)]
            else:
                luts = cell_luts.get(node) or exact_luts(t, has_carry=True)
                ins = [val[r] for r in ref]
                idx = ins[0]
                for x in ins[1:]:
                    idx = (idx << 1) | x
                sum_t = torch.from_numpy(luts["sum"]).to(dev)[idx]
                val[(node, "sum")] = sum_t
                if aux or t == 4:  # has_carry；ct42 恒有 carry+cout
                    val[(node, "carry")] = torch.from_numpy(luts["carry"]).to(dev)[idx]
                if t == 4:
                    val[(node, "cout")] = torch.from_numpy(luts["cout"]).to(dev)[idx]
        n = next(iter(val.values())).shape[0]
        out = torch.zeros(n, dtype=torch.int64, device=dev)
        for col, ws in enumerate(self.final_wires):
            for w in ws:
                out = out + (val[(w, "sum")] << col)
        return out

    # ------------------------------------------------------------ 可微模式
    def eval_diff(self, pp_bits, sel):
        """sel: {node_idx: (tt_stack [K,2^n,n_out], w [K])}，w 由 solver 提供
        （hard one-hot + STE）。非 slot 节点用精确 TT 的多线性形式。
        返回 out float64 [N]（可反传到各 w）。"""
        dev = self.device
        val = {}
        cache_exact = {}
        for node, kind, t, col, ref, aux in self.plan:
            if kind == "pp":
                val[(node, "sum")] = pp_bits[ref]  # float32 0/1
                continue
            if kind == "visual":
                (src, oname), = ref
                val[(node, "sum")] = val[(src, oname)]
                continue
            ins = [val[r] for r in ref]
            # basis P: [2^n, N]，port0 为最高位
            P = torch.ones(1, ins[0].shape[0], dtype=torch.float32, device=dev)
            for x in ins:
                P = torch.stack([P * (1 - x), P * x], dim=1).reshape(-1, P.shape[1])
            if node in sel:
                tt_stack, w = sel[node]  # [K, 2^n, n_out], [K]
                mixed = torch.einsum("k,kpo->po", w, tt_stack)  # [2^n, n_out]
            else:
                if (t, aux) not in cache_exact:
                    luts = exact_luts(t, has_carry=True)
                    outs = [luts["sum"], luts["carry"]] + (
                        [luts["cout"]] if t == 4 else [])
                    cache_exact[(t, aux)] = torch.from_numpy(
                        np.stack(outs, axis=1)).to(torch.float32).to(dev)
                mixed = cache_exact[(t, aux)]
            o = mixed.T @ P  # [n_out, N]
            val[(node, "sum")] = o[0]
            if o.shape[0] > 1:
                val[(node, "carry")] = o[1]
            if t == 4:
                val[(node, "cout")] = o[2]
        n = pp_bits[next(iter(pp_bits))].shape[0]
        out = torch.zeros(n, dtype=torch.float64, device=dev)
        for col, ws in enumerate(self.final_wires):
            for w_ in ws:
                out = out + val[(w_, "sum")].to(torch.float64) * float(2 ** col)
        return out


# ---------------------------------------------------------------- 误差口径
def wrap31_int(e):
    HALF, FULL = 1 << 30, 1 << 31
    e = torch.where(e > HALF, e - FULL, e)
    e = torch.where(e < -HALF, e + FULL, e)
    return e


def error_stats(out, a_np, b_np):
    """整数模式统计，公式逐条对齐 mul_err_wrap.cpp。"""
    golden = (torch.from_numpy(a_np.astype(np.int64)).to(out.device)
              * torch.from_numpy(b_np.astype(np.int64)).to(out.device)) & MASK31
    e = wrap31_int((out & MASK31) - golden)
    ae = e.abs()
    n = out.shape[0]
    med = ae.sum().item() / n
    bias = e.sum().item() / n
    nz = golden != 0
    mred = (ae[nz].to(torch.float64) / golden[nz].to(torch.float64)).sum().item() / \
        int(nz.sum().item())
    return {"med": med, "bias": bias, "mred": mred,
            "wce_mc": int(ae.max().item()), "n": n}


def diff_mred(out, a_np, b_np):
    """可微 MRED（wrap 用 where 的直通梯度；golden==0 剔除）。"""
    dev = out.device
    golden = (torch.from_numpy(a_np.astype(np.int64)).to(dev)
              * torch.from_numpy(b_np.astype(np.int64)).to(dev)) & MASK31
    gf = golden.to(torch.float64)
    e = torch.remainder(out, float(1 << 31)) - gf
    HALF, FULL = float(1 << 30), float(1 << 31)
    e = torch.where(e > HALF, e - FULL, e)
    e = torch.where(e < -HALF, e + FULL, e)
    nz = golden != 0
    return (e.abs()[nz] / gf[nz]).mean()
