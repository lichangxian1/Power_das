"""近似 cell 类型：库文件加载/解析 + 逐 cell 类型采样
（independent / cardinality 两种 sampler）及其 log-prob。

本模块是 CompressorRouting 的一个切面（mixin）：方法从原单文件
arith_das_v5.py 逐行原样搬运，通过 core.CompressorRouting 多继承拼装，
self 上的属性均在 core.__init__ 中定义。"""
import os
import sys
import json
import re

import torch


class CellTypeMixin:
    """近似 cell 类型库加载与逐 cell 类型采样（含 log-prob）。"""

    def _load_approx_types(self, sel_path, lib_path):
        """从 selected_compressors.json + library.json 构建类型表（index 0 = exact）。"""
        import itertools

        sel = json.load(open(self._resolve_path(sel_path)))["selected"]
        lib = json.load(open(self._resolve_path(lib_path)))["cells"]
        # 别名顺序：直接从 selected 文件按 type 字段读取（exact 永远在 index 0，其余按文件
        # 出现顺序）。不再写死 pos/neg 的 6+4 槽位，使菜单大小可变（v3 lean/dense A/B 对照）。
        # 向后兼容旧 selected_compressors.json（同样筛出 exact+6/exact+4、index0=exact）。
        def _ordered(ctype):
            ks = [k for k, v in sel.items() if v.get("type") == ctype]
            ex = [k for k in ks if sel[k].get("group") == "exact"]
            ap = [k for k in ks if sel[k].get("group") != "exact"]
            return ex + ap
        self.type_table_32 = [dict(sel[k]) for k in _ordered("32")]
        self.type_table_22 = [dict(sel[k]) for k in _ordered("22")]
        assert self.type_table_32[0]["group"] == "exact", "T32[0] 必须是 exact"
        assert self.type_table_22[0]["group"] == "exact", "T22[0] 必须是 exact"
        if self.use_ct42:
            self.type_table_42 = self._load_approx42_table(sel)
            assert self.type_table_42[0]["group"] == "exact", "T42[0] 必须是 exact"
        else:
            self.type_table_42 = []

        # 预生成每个近似 cell 的可综合 SOP module（LUT 取自 library.json）
        APPR = os.path.join(self._REPO_ROOT, "Appr_Comp")
        if APPR not in sys.path:
            sys.path.insert(0, APPR)
        from gen_verilog import emit_module

        pat3 = ["".join(p) for p in itertools.product("01", repeat=3)]
        pat2 = ["".join(p) for p in itertools.product("01", repeat=2)]
        for entry in self.type_table_32 + self.type_table_22:
            name = entry["name"]
            if entry["group"] == "exact":
                continue  # exact 用内置 FA/HA，无需追加 module
            cell = lib[name]
            if entry["type"] == "32":
                src = emit_module(name, ["a", "b", "cin"], pat3,
                                  cell["sum_lut"], cell["carry_lut"],
                                  f"{name} bias={entry['bias']:+.3f}")
            else:
                src = emit_module(name, ["a", "cin"], pat2,
                                  cell["sum_lut"], cell["carry_lut"],
                                  f"{name} bias={entry['bias']:+.3f}")
            self.approx_module_src_by_name[name] = src
        if self.use_ct42:
            self._load_approx42_modules()

    def _extract_verilog_modules(self, rtl_path, names):
        src = open(self._resolve_path(rtl_path)).read()
        modules = {}
        for name in names:
            pat = (
                r"(?ms)^module\s+"
                + re.escape(name)
                + r"\b.*?^endmodule\s*\n?"
            )
            m = re.search(pat, src)
            if m is None:
                raise ValueError(f"cannot find Verilog module {name} in {rtl_path}")
            modules[name] = m.group(0)
            if not modules[name].endswith("\n"):
                modules[name] += "\n"
        return modules

    def _ct42_entry_with_cin0_metrics(self, name, cell):
        entry = {
            "name": name,
            "type": "42",
            "group": cell.get("group", "P"),
            "alias": cell.get("alias", name),
            "builder": cell.get("builder"),
            "area": cell.get("area"),
            "power_mw": cell.get("power_mw"),
            "delay_ns": cell.get("delay_ns"),
            # M2 恒零标志随条目走（_zero_entry_of 靠它挑真恒零 cell）
            "const_zero": bool(cell.get("const_zero", False)),
        }
        patterns = cell.get("patterns")
        if patterns and "sum_lut" in cell and "carry_lut" in cell and "cout_lut" in cell:
            bias = 0.0
            wae = 0.0
            maxe = 0.0
            er = 0.0
            for i, bits in enumerate(patterns):
                if bits[-1] != "0":
                    continue
                a, b, c, d = [int(x) for x in bits[:4]]
                approx = (
                    int(cell["sum_lut"][i])
                    + 2 * int(cell["carry_lut"][i])
                    + 2 * int(cell["cout_lut"][i])
                )
                exact = a + b + c + d
                err = approx - exact
                prob = 1.0
                for bit in (a, b, c, d):
                    prob *= 0.25 if bit else 0.75
                bias += prob * err
                wae += prob * abs(err)
                maxe = max(maxe, abs(err))
                if err != 0:
                    er += prob
            entry.update({"bias": bias, "wae": wae, "er": er, "maxe": maxe})
        else:
            entry.update(
                {
                    "bias": cell.get("bias", cell.get("weighted_signed_error", 0.0)),
                    "wae": cell.get("wae", cell.get("weighted_absolute_error", 0.0)),
                    "er": cell.get("er", cell.get("error_rate", 0.0)),
                    "maxe": cell.get("maxe", cell.get("max_error", 0.0)),
                }
            )
        if cell.get("is_exact") or entry["group"] == "exact":
            entry["group"] = "exact"
            entry["bias"] = 0.0
            entry["wae"] = 0.0
            entry["er"] = 0.0
            entry["maxe"] = 0.0
        return entry

    def _load_approx42_table(self, selected):
        lib_path = self._resolve_path(self.approx42_library_path)
        lib42_full = json.load(open(lib_path))
        lib42 = lib42_full["cells"]
        # 原生 4 输入 cell（无 cin 端口，gen_comp42_native.py 产出）：发射时不接 .cin
        self._ct42_native4_names = {
            n for n, c in lib42.items() if c.get("pattern_bits") == 4
        }
        # exact CT42 锚点 PPA（外环 area_save_frac 打分需要）：
        # selected_compressors42_native.json 存 meta.anchor_area；
        # library42_native.json 存 meta.anchors.CT42_BAL.area。
        meta = lib42_full.get("meta") or {}
        self._ct42_exact_area = meta.get("anchor_area")
        if self._ct42_exact_area is None:
            self._ct42_exact_area = (
                (meta.get("anchors") or {}).get("CT42_BAL") or {}
            ).get("area")
        selected42_keys = [k for k, v in selected.items() if v.get("type") == "42"]
        if selected42_keys:
            exact_keys = [
                k for k in selected42_keys
                if selected[k].get("group") == "exact"
            ]
            approx_keys = [
                k for k in selected42_keys
                if selected[k].get("group") != "exact"
            ]
            table = []
            for key in exact_keys + approx_keys:
                name = selected[key]["name"]
                cell = dict(lib42.get(name, {}))
                cell.update(selected[key])
                table.append(self._ct42_entry_with_cin0_metrics(name, cell))
            return table

        entries = [
            self._ct42_entry_with_cin0_metrics(name, cell)
            for name, cell in lib42.items()
        ]
        exact = [e for e in entries if e.get("group") == "exact"]
        approx = [e for e in entries if e.get("group") != "exact"]
        approx.sort(key=lambda e: (float(e["wae"]), abs(float(e["bias"])), float(e["maxe"]), e["name"]))
        if not exact:
            exact = [{
                "name": "CT42",
                "type": "42",
                "group": "exact",
                "bias": 0.0,
                "wae": 0.0,
                "er": 0.0,
                "maxe": 0.0,
                "area": getattr(self, "_ct42_exact_area", None),
            }]
        max_types = self.approx42_max_types
        if max_types is not None:
            max_types = int(max_types)
            if max_types < 1:
                raise ValueError("approx42_max_types must be >= 1 or None")
            approx = approx[: max(0, max_types - 1)]
        return [exact[0]] + approx

    def _load_approx42_modules(self):
        names = [
            e["name"] for e in self.type_table_42
            if e.get("group") != "exact"
        ]
        if not names:
            return
        self.approx_module_src_by_name.update(
            self._extract_verilog_modules(self.approx42_rtl_path, names)
        )

    def sample_cell_types(self):
        """对压缩器节点采样 cell 类型。

        返回 (cell_map, type_choices, type_log_prob, type_sample_info)。
        cell_map: {node_idx -> module名}（仅非 exact）。
        type_choices: 旧独立模式记录全部节点；cardinality 模式只记录非 exact 节点。
        type_sample_info: PPO 重算 log_prob 所需的采样口径元数据。
        """
        if self.approx_cardinality_sampler:
            return self._sample_cell_types_cardinality()
        return self._sample_cell_types_independent()

    def _sample_cell_types_independent(self):
        """旧行为：每个可压缩器槽独立采 exact/approx。"""
        cell_map, type_choices = {}, {}
        total_log_prob = 0.0
        if not self.use_approx_types:
            return cell_map, type_choices, total_log_prob, {"mode": "none"}
        emb = self._node_emb
        for node_idx, info in enumerate(self.comp_graph.vertex_list):
            _, c, t, _ = info
            if t == 0:
                head, table = self.type_head_32, self.type_table_32
            elif t == 1:
                head, table = self.type_head_22, self.type_table_22
            elif t == 4 and self.use_ct42:
                head, table = self.type_head_42, self.type_table_42
            else:
                continue
            logits = self._masked_type_logits(head(emb[node_idx]), c)
            dist = torch.distributions.Categorical(logits=logits)
            sample = dist.sample()
            total_log_prob += dist.log_prob(sample).item()
            k = sample.item()
            type_choices[node_idx] = (t, k)
            if k != 0:
                cell_map[node_idx] = table[k]["name"]
        return cell_map, type_choices, total_log_prob, {"mode": "independent"}

    def _approx_col_upper(self):
        upper = self.approx_max_col
        if self.approx_col_window is not None:
            upper = min(upper, self.trunc_cols + self.approx_col_window)
        return upper

    def _is_approx_col_allowed(self, col):
        return self.trunc_cols <= col < self._approx_col_upper()

    def _eligible_type_nodes(self):
        nodes = []
        for node_idx, info in enumerate(self.comp_graph.vertex_list):
            _, c, t, _ = info
            if t in (0, 1, 4) and self._is_approx_col_allowed(c):
                _head, table = self._type_head_and_table(t)
                if len(table) <= 1:
                    continue
                nodes.append(node_idx)
        return nodes

    def _type_head_and_table(self, t):
        if t == 0:
            return self.type_head_32, self.type_table_32
        if t == 1:
            return self.type_head_22, self.type_table_22
        if t == 4 and self.use_ct42:
            return self.type_head_42, self.type_table_42
        raise ValueError(f"unknown compressor type {t}")

    def _node_type_logits(self, node_idx):
        _, c, t, _ = self.comp_graph.vertex_list[node_idx]
        head, _table = self._type_head_and_table(t)
        return self._masked_type_logits(head(self._node_emb[node_idx]), c)

    def _cardinality_dist(self, n_eligible):
        choices = torch.tensor(
            self.approx_cardinality_choices, device=self.device, dtype=torch.long
        )
        mask = choices <= int(n_eligible)
        logits = self.approx_cardinality_logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def _eligible_node_scores(self, eligible_nodes):
        scores = []
        for node_idx in eligible_nodes:
            logits = self._node_type_logits(node_idx)
            # Score slots by approximate-vs-exact odds; K itself is sampled separately.
            scores.append(torch.logsumexp(logits[1:], dim=0) - logits[0])
        return torch.stack(scores)

    def _sample_cell_types_cardinality(self):
        """方案 B：先采 n_approx，再无放回采 slot，最后在非 exact cell 中采具体类型。"""
        cell_map, type_choices = {}, {}
        total_log_prob = 0.0
        if not self.use_approx_types:
            return cell_map, type_choices, total_log_prob, {"mode": "none"}

        eligible_nodes = self._eligible_type_nodes()
        if not eligible_nodes:
            return (
                cell_map,
                type_choices,
                total_log_prob,
                {"mode": "cardinality", "cardinality_choice_idx": 0, "selected_order": []},
            )

        k_dist = self._cardinality_dist(len(eligible_nodes))
        k_sample = k_dist.sample()
        total_log_prob += k_dist.log_prob(k_sample).item()
        k_choice_idx = int(k_sample.item())
        n_approx = int(self.approx_cardinality_choices[k_choice_idx])

        selected_order = []
        if n_approx > 0:
            scores = self._eligible_node_scores(eligible_nodes)
            remaining = torch.ones(
                len(eligible_nodes), device=self.device, dtype=torch.bool
            )
            for _ in range(n_approx):
                node_dist = torch.distributions.Categorical(
                    logits=scores.masked_fill(~remaining, -1e9)
                )
                pos = node_dist.sample()
                total_log_prob += node_dist.log_prob(pos).item()
                pos_i = int(pos.item())
                remaining[pos_i] = False
                node_idx = eligible_nodes[pos_i]
                selected_order.append(node_idx)

                _, _c, t, _ = self.comp_graph.vertex_list[node_idx]
                logits = self._node_type_logits(node_idx)
                cell_dist = torch.distributions.Categorical(logits=logits[1:])
                cell_sample = cell_dist.sample()
                total_log_prob += cell_dist.log_prob(cell_sample).item()
                k = int(cell_sample.item()) + 1
                _head, table = self._type_head_and_table(t)
                type_choices[node_idx] = (t, k)
                cell_map[node_idx] = table[k]["name"]

        return (
            cell_map,
            type_choices,
            total_log_prob,
            {
                "mode": "cardinality",
                "cardinality_choice_idx": k_choice_idx,
                "selected_order": selected_order,
            },
        )

    def _sampled_cell_type(self, cell_types, node_idx):
        if node_idx in cell_types:
            return cell_types[node_idx]
        return cell_types.get(str(node_idx))

    def _independent_cell_type_log_prob(self, cell_types):
        new_log_prob = torch.zeros((), device=self.device)
        for node_idx, (t, k) in cell_types.items():
            node_idx = int(node_idx)
            logits = self._node_type_logits(node_idx)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = new_log_prob + dist.log_prob(
                torch.tensor(int(k), device=self.device)
            )
        return new_log_prob

    def _cardinality_cell_type_log_prob(self, cell_types, type_sample_info):
        eligible_nodes = self._eligible_type_nodes()
        k_choice_idx = int(type_sample_info.get("cardinality_choice_idx", 0))
        k_dist = self._cardinality_dist(len(eligible_nodes))
        new_log_prob = k_dist.log_prob(
            torch.tensor(k_choice_idx, device=self.device)
        )

        selected_order = [int(x) for x in type_sample_info.get("selected_order", [])]
        if not selected_order:
            return new_log_prob

        pos_by_node = {node_idx: pos for pos, node_idx in enumerate(eligible_nodes)}
        scores = self._eligible_node_scores(eligible_nodes)
        remaining = torch.ones(len(eligible_nodes), device=self.device, dtype=torch.bool)
        for node_idx in selected_order:
            pos_i = pos_by_node[node_idx]
            node_dist = torch.distributions.Categorical(
                logits=scores.masked_fill(~remaining, -1e9)
            )
            new_log_prob = new_log_prob + node_dist.log_prob(
                torch.tensor(pos_i, device=self.device)
            )
            remaining[pos_i] = False

            tk = self._sampled_cell_type(cell_types, node_idx)
            if tk is None:
                raise ValueError(f"missing sampled cell type for node {node_idx}")
            _t, k = tk
            logits = self._node_type_logits(node_idx)
            cell_dist = torch.distributions.Categorical(logits=logits[1:])
            new_log_prob = new_log_prob + cell_dist.log_prob(
                torch.tensor(int(k) - 1, device=self.device)
            )
        return new_log_prob

    def _cell_type_log_prob(self, sample_info):
        type_sample_info = sample_info.get("cell_type_info") or {}
        mode = type_sample_info.get("mode")
        if mode == "outer":
            return None  # 外环模式：类型非采样所得，不进 PPO ratio
        if mode == "cardinality":
            return self._cardinality_cell_type_log_prob(
                sample_info.get("cell_types") or {}, type_sample_info
            )
        cell_types = sample_info.get("cell_types") or {}
        if cell_types:
            return self._independent_cell_type_log_prob(cell_types)
        return None

    def _approx_modules_src(self, cell_map):
        if not cell_map:
            return ""
        used = sorted(set(cell_map.values()))
        body = "".join(self.approx_module_src_by_name[n] for n in used)
        return "\n// ===== approximate compressor cells =====\n" + body

    def _cell_map_from_types(self, cell_types):
        """从 {node_idx:(t,k)} 复原 {node_idx:module名}（k=0/exact 不收）。
        要求 comp_graph 与采样时同序（同一 assignment 重建即一致）。"""
        cell_map = {}
        for node_idx, tk in (cell_types or {}).items():
            t, k = tk
            if k != 0:
                _head, table = self._type_head_and_table(t)
                cell_map[int(node_idx)] = table[k]["name"]
        return cell_map

    def _masked_type_logits(self, logits, col):
        """col 落在 [trunc_cols, upper) 外时只留 exact(index 0)，其余置 -1e9。
        upper = approx_max_col；若设了 approx_col_window，则 upper=min(approx_max_col,
        trunc_cols+window)——把可近似列收窄到截断边界上方的窗口（高列 cell 误差∝2^col 几乎
        永远不划算，集中探索廉价低列）。截断列被常数驱动、cell 会被 DC 删掉，也不放。
        用 masked_fill（非 in-place，autograd 安全）。"""
        if not self._is_approx_col_allowed(col):
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[0] = False
            logits = logits.masked_fill(mask, -1e9)
        return logits
