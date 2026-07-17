"""k 截断档：截断常数 C* 的设置与解析误差（MED/MRED 闭式）计算，
per-k TruncProfile 的捕获/恢复/激活。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import logging


import numpy as np


class TruncationMixin:
    """k 截断档：截断常数 C* 与 MED/MRED 闭式解析误差。"""

    def _setup_truncation(self):
        """Phase C ①：算截断 [0,k) 的校正常数 C（用低列槽位的常数 1 位表示）+ 误差量。

        E[Δ] = Σ_{c<k} 0.25·pp[c]·2^c   （AND, P=1/4；截断丢失值期望，恒正→负偏置）
        C = round(E[Δ])（trunc_correct='bias'），贪心用 col<k 的槽位表示（列 c 有 pp[c] 个槽、权重 2^c）
        Δmax = Σ_{c<k} pp[c]·2^c，WCE_trunc = max(C, Δmax−C)。
        error_metric=mred 时 C 改取 argmin E[|C−Δ|/p]（全宽 MC）：round(E[Δ]) 是 MED 最优但对
        MRED 过校正（小积整积被截、输出≈C、(C−p)/p 被 1/p 加权重锤），C* 比 E[Δ] 小 2–10×。"""
        k = self.trunc_cols
        if not (0 <= k <= len(self.initial_pp)):
            raise ValueError(
                f"trunc_cols={k} 越界，应在 [0, {len(self.initial_pp)}]（截断列数）"
            )
        pp = [int(x) for x in self.initial_pp]
        e_delta = sum(0.25 * pp[c] * (1 << c) for c in range(k))
        delta_max = sum(pp[c] * (1 << c) for c in range(k))
        c_target = int(round(e_delta)) if self.trunc_correct == "bias" else 0
        # mred 口径：常数改取 MRED 最优 C*（Δ 只依赖 a/b 低 k 位，但 1/p 权重需要全宽乘积，
        # 故单独全宽 MC；确定性 seed，整个训练只算一次——_trunc_bits 缓存保证）。
        if (getattr(self, "error_metric", "med") == "mred"
                and self.trunc_correct == "bias" and k > 0 and c_target > 0):
            rng_f = np.random.default_rng(1)
            W = int(self.bit_width)
            Nf = 1_000_000
            af = rng_f.integers(0, 1 << W, size=Nf, dtype=np.int64)
            bf = rng_f.integers(0, 1 << W, size=Nf, dtype=np.int64)
            dl = np.zeros(Nf, dtype=np.int64)
            for i in range(min(k, W)):
                ai = (af >> i) & 1
                for j in range(min(k - i, W)):
                    dl += (ai & ((bf >> j) & 1)) << (i + j)
            pm = af * bf
            nz = pm > 0
            pw = pm[nz].astype(np.float64)
            dw = dl[nz].astype(np.float64)

            def _mred_of(cc):
                return float(np.mean(np.abs(cc - dw) / pw))

            grid = np.linspace(0.0, 1.2 * c_target, 81)
            i0 = int(np.argmin([_mred_of(cc) for cc in grid]))
            step = grid[1] - grid[0]
            fine = np.linspace(max(0.0, grid[i0] - step), grid[i0] + step, 41)
            vals = [_mred_of(cc) for cc in fine]
            j0 = int(np.argmin(vals))
            c_star = int(round(fine[j0]))
            logging.info(
                "[trunc-mred] C*=%d (MED口径 C0=%d)  模型MRED: C0=%.3e → C*=%.3e",
                c_star, c_target, _mred_of(float(c_target)), vals[j0],
            )
            c_target = c_star
            self._trunc_model_mred = float(vals[j0])  # 外环 MRED-slack 过滤的 floor
        bits, remaining = {}, c_target
        for c in range(k - 1, -1, -1):          # 高列→低列贪心填常数 1
            w = 1 << c
            m = min(pp[c], remaining // w)
            if m > 0:
                bits[c] = m
                remaining -= m * w
        c_actual = c_target - remaining          # 实际可表示的 C（余量通常 0）
        # 截断残差 MED = E[|C−Δ|]，Δ = Σ_{i+j<k} 2^(i+j)·a_i·b_j（被丢的低列加权值）。
        # PP 共享 a/b 位 → 各列高度强相关，独立卷积会低估约 30%（codex review 指出，
        # k=8 独立=142 vs 真实=200）。故用 MC 直接采 a,b 算 Δ，捕捉相关性。均匀输入口径，
        # 仍是一阶估计（⑤ 接 SAIF 真实逐位概率后再精化）；零 bias≠零 MED 必须显式建模。
        rng = np.random.default_rng(0)
        N = 200000
        av = rng.integers(0, 1 << k, size=N, dtype=np.int64)
        bv = rng.integers(0, 1 << k, size=N, dtype=np.int64)
        delta = np.zeros(N, dtype=np.int64)
        for i in range(k):
            ai = (av >> i) & 1
            for j in range(k - i):
                delta += (ai & ((bv >> j) & 1)) << (i + j)
        trunc_med = float(np.abs(c_actual - delta).mean())
        self._trunc_bits = bits
        self._trunc_const = c_actual
        self._trunc_delta = e_delta
        self._trunc_wce = max(c_actual, delta_max - c_actual)
        self._trunc_med = trunc_med
        logging.info(
            "[trunc] cols<%d const-driven; E[Δ]=%.2f C=%d(target %d) Δmax=%d "
            "MED_trunc=%.2f WCE_trunc=%d bits=%s",
            k, e_delta, c_actual, c_target, delta_max, trunc_med, self._trunc_wce, bits,
        )
        # 跨-k 归一 error_scale（仅 error_as_metric）：让 med/error_scale≈O(1) 对所有 k，
        # 从而单一 error_weight 跨 k 行为一致（见 error_scale_mode 注释）。
        if self.error_as_metric and self.error_scale_mode != "fixed" and k > 0:
            if self.error_scale_mode == "pow2k":
                self.error_scale = float(1 << (k - 1))
            elif self.error_scale_mode == "sqrt2k":
                # √k·2^(k-1)：截断 MED floor ∝ std(Δ) ∝ √k·2^(k-1)，比 pow2k 跨 k 更平
                # （med/scale 跨 k 仅 ~1.3× vs pow2k ~3.7×）。
                self.error_scale = (float(k) ** 0.5) * float(1 << (k - 1))
            elif self.error_scale_mode == "floor":
                self.error_scale = max(float(self._trunc_med), 1.0)
            else:
                raise ValueError(f"未知 error_scale_mode={self.error_scale_mode!r}")
            logging.info(
                "[trunc] error_scale_mode=%s -> error_scale=%.2f "
                "(归一后 floor/scale=%.3f)",
                self.error_scale_mode, self.error_scale, trunc_med / self.error_scale,
            )

    def _analytic_error(self, type_choices):
        """从采样类型闭式估计 (med_lsb, abs_bias_lsb, nmed, wce_lsb)。

        med_lsb = Σ wae·2^col   —— MED 的保守上界（输出 LSB 绝对单位，跨位宽稳定）
        abs_bias_lsb = |Σ bias·2^col|  —— 带符号误差的绝对值（抓正负抵消）
        nmed = med_lsb / maxprod  —— 标准 NMED，仅用于上报
        wce_lsb = Σ maxe·2^col  —— ④ WCE 可加上界（最坏情况输出误差，LSB；与传播无关恒成立）
        bias/wae 为 P=1/4 一阶估计；maxe 与概率无关（逐 cell 最坏），故 WCE 上界精确。"""
        items = []
        for node_idx, (t, k) in type_choices.items():
            if k == 0:
                continue
            col = self.comp_graph.vertex_list[int(node_idx)][1]
            items.append((col, t, k))
        wae_total, bias_total, wce_total = self._error_totals_from_cols(items)
        maxprod = float((2 ** self.bit_width - 1) ** 2)
        return wae_total, abs(bias_total), wae_total / maxprod, wce_total

    def _error_totals_from_cols(self, items):
        """闭式误差核算（单一事实源）：items = [(col, t, k)]，k≥1（非 exact）。
        返回 (med_lsb, signed_bias_lsb, wce_lsb)，均含截断项。
        Phase C ①：截断的确定性误差。−E[Δ]+C 为净偏置（bias 项会驱动 cell 抵消残差）；
        MED_trunc=E[|C−Δ|] 进 MED 上界（三角不等式：MED_total ≤ MED_trunc + Σ wae·2^col，
        否则纯截断设计解析 MED=0 会骗过 med_budget）；WCE_trunc 进尾部上界（与 ④ 同口径）。"""
        bias_total = 0.0
        wae_total = 0.0
        wce_total = 0.0
        for col, t, k in items:
            _head, table = self._type_head_and_table(t)
            entry = table[k]
            w = float(1 << col)
            bias_total += entry["bias"] * w
            wae_total += entry["wae"] * w
            wce_total += entry.get("maxe", 0.0) * w
        if self.trunc_cols > 0:
            bias_total += (-self._trunc_delta + self._trunc_const)
            wae_total += self._trunc_med
            wce_total += self._trunc_wce
        return wae_total, bias_total, wce_total

    def _capture_trunc_profile(self):
        return {
            "bits": dict(self._trunc_bits),
            "const": self._trunc_const,
            "delta": self._trunc_delta,
            "wce": self._trunc_wce,
            "med": self._trunc_med,
            "model_mred": self._trunc_model_mred,
            # 双评审发现 #5 修复：error_as_metric+pow2k/sqrt2k/floor 模式下
            # _setup_truncation 会按 k 改 error_scale——必须随档存取，否则切回
            # 缓存 k 时 error_scale 是"上一个新算 k"的（访问历史依赖）
            "error_scale": self.error_scale,
        }

    def _restore_trunc_profile(self, p):
        self._trunc_bits = dict(p["bits"])
        self._trunc_const = p["const"]
        self._trunc_delta = p["delta"]
        self._trunc_wce = p["wce"]
        self._trunc_med = p["med"]
        self._trunc_model_mred = p["model_mred"]
        if "error_scale" in p:
            self.error_scale = p["error_scale"]

    def _activate_trunc_profile(self, k):
        """M0 k 线程化：把全局截断状态切到深度 k。TruncProfile 逐 k lazy 缓存
        （C*/floor 的确定性 MC 每个 k 只算一次，秒级）。k 不改树结构——截断列压缩器
        照常实例化、DC 常数传播扫掉，k 只换 (常数 PP, C*, 资格窗, 误差 floor)。"""
        k = int(k)
        if not hasattr(self, "_trunc_profiles"):
            self._trunc_profiles = {}
            if self.trunc_cols > 0 and self._trunc_bits:
                self._trunc_profiles[self.trunc_cols] = self._capture_trunc_profile()
        if k in self._trunc_profiles:
            if self.trunc_cols != k or not (k == 0 or self._trunc_bits):
                self.trunc_cols = k
                self._restore_trunc_profile(self._trunc_profiles[k])
            return
        self.trunc_cols = k
        if k == 0:
            self._trunc_bits, self._trunc_const = {}, 0
            self._trunc_delta = self._trunc_wce = self._trunc_med = 0.0
            self._trunc_model_mred = None
        else:
            self._trunc_bits = {}
            self._trunc_model_mred = None
            self._setup_truncation()
        self._trunc_profiles[k] = self._capture_trunc_profile()
