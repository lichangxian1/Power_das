# 阶段3 Phase B 设计方案（待审核）

给 ARITH-DAS 搜索加「压缩器类型头」：让搜索在布线之外，**为每个压缩器槽（ct32/ct22 节点）选择 cell 类型**
（exact + 12 个近似 cell），并在 reward 里加入闭式误差项（NMED/偏置），使搜索在 PPA 与精度之间权衡，
靠正/负偏置在列权重下相互抵消。

> 本文件只是设计，**尚未改任何搜索代码**。审核通过后再按 §10 顺序实现。

---

## 0. 关键前提（已读源码核实，2026-06-18）

1. **两条 RTL 发射路是分开的。** 搜索用的是
   `CompressorRouting.emit_assignment()` → `_declare_ct32/_declare_ct22`
   （[trainer/arith_das.py](../trainer/arith_das.py) 1087/1115/1139 行），里面**硬编码** `FA`/`HA` 实例。
   我在 Phase A 改的 `compressor_tree.emit_verilog_fused_assignment(cell_policy=...)`
   **不在搜索路径上**（那是 `mul.emit_verilog` 传 list-assignment 时走的路；搜索传的是 dict）。
   → **Phase B 的 cell 注入点是 `_declare_ct32/22`，不是 compressor_tree。**

2. **搜索的 PPA 路径不跑功能测试。** `mul.simulate` → `simulate_worker`
   （[utils/mul.py](../utils/mul.py) 340–400）只做 `yosys` 综合 + `openroad` STA，
   解析 wns/area/power，**没有任何 testbench、没有逻辑判废**。
   近似 cell 是行为级 Verilog module（SOP），yosys 会照常综合到工艺库 → **搜索侧 PPA 天然支持近似 cell，
   零改动、无判废**。`send_eda.py` 的 `APPROX_EVAL` 拦路石只服务远端 sandbox 数据集/sweep，与搜索无关。

3. **module 定义注入已就绪。** `Mul.emit_verilog(..., extra_modules_src=...)`（Phase A 加）在 dict 路径也会
   把近似 module 追加到 RTL 末尾（[utils/mul.py](../utils/mul.py):279）。可直接复用
   `Appr_Comp/gen_verilog.emit_module` 生成（与 `approx_mul.py` 同款）。

4. **端口对齐已验证。** 近似 comp32 端口 `.a/.b/.cin/.sum/.cout` 同 `FA`；comp22 `.a/.cin/.sum/.cout` 同 `HA`。
   `_declare_ct32/22` 现有的实例化串可原样套用，只换 cell 名。

5. **搜索是两层。** 外层 `reset/transition/get_action_mask` 决定每列 FA/HA 的**数量** `ct32[c]/ct22[c]`；
   内层 GNN（`get_Z_mat/sample_from_logits/emit_assignment`）决定**布线**。
   Phase B 在内层再加一个**类型维度**：每个已存在的压缩器节点额外采样一个 cell 类型。

---

## 1. 改动点清单（逐函数，带行号锚点）

| # | 位置 | 改动 | 默认关时行为 |
|---|------|------|--------------|
| 1 | `MultiChannelResGCN.__init__/forward` (156/191) | 加两个类型头 `fc_type32: in_dim→\|T32\|`、`fc_type22: in_dim→\|T22\|`（仅当 `num_type32/22>0` 创建）；`forward` 末尾**额外返回节点嵌入 `x`** | 不创建头，但要改 forward 返回值→见 §回归 |
| 2 | `CompressorRouting.__init__` (422) | 读 `selected_compressors.json` 建类型表 `T32/T22`（name/bias/wae/area…）；新增配置存字段；构 `node→col` 缓存 | `use_approx_types=False` 时全部跳过 |
| 3 | 新 `CompressorRouting.sample_cell_types(x_emb)` | 对每个 ct32/ct22 节点采样 cell 类型，返回 `cell_map{node_idx→module名}`、`type_choices{node_idx→k}`、`type_log_prob` | 不调用 |
| 4 | `get_Z_mat` (1198) | 额外返回（或缓存）节点嵌入 `x_emb`，供类型头复用 | 多返回一个值，更新调用点 |
| 5 | `get_samples` (1298) | 采样后调 `sample_cell_types`；把 `type_log_prob` 累加进 `overall_log_prob`；`cell_map` 传给 `emit_assignment`；`extra_modules_src` 传给 `emit_verilog`；`cell_types` 存进 `sample_info` | cell_map=None |
| 6 | `emit_assignment(samples_connection, cell_map=None)` (1139) | 透传 `cell_map` 给 `_declare_ct32/22` | cell_map=None |
| 7 | `_declare_ct32/_declare_ct22` (1087/1115) | `cell = cell_map.get(node_idx) or "FA"/"HA"`；用 `cell` 替换硬编码实例名 | 取默认 FA/HA，串完全不变 |
| 8 | `_summarize_result/get_objective` (1368/1408) | 从 `sample_info["cell_types"]` 算 `bias_total/wae_total → NMED_analytic/bias_norm`，按模式加进 objective | 不加误差项 |
| 9 | 新 `get_error_loss(type_logits, ...)`（可选，推荐） | 可微误差期望 surrogate，仿 `get_delay_loss` 风格；`run_episode` 里像 `l_delay` 一样加 | 不启用 |
| 10 | `get_ppo_loss` (1486) | 连接 log_prob 之后，用当前类型头重算 `type_log_prob`（同 col-mask）加进 `new_log_prob` | 跳过类型分支 |
| 11 | `run_episode` (1629) / 配置解析 | 新 loss 项、新配置开关接线 | 全默认关 |

---

## 2. 类型动作空间

来自 `Appr_Comp/selected_compressors.json`（12 代表 + exact）：

- **T32（3:2，7 类）**：`[exact, apx_pos_1, apx_pos_2, apx_pos_3, apx_neg_1, apx_neg_2, apx_neg_3]`
- **T22（2:2，5 类）**：`[exact, apx_pos_1, apx_pos_2, apx_neg_1, apx_neg_2]`

索引 0 恒为 exact。每类带 `(module名, bias=weighted_signed_error, wae=weighted_absolute_error, area, dyn_w, leak_w, tmax)`。

**只在低位列近似**：配置 `approx_max_col`，对 `col_idx >= approx_max_col` 的节点把类型 logits 掩成只剩 exact
（强制精确）。高位列主导精度，绝不近似。`_declare_ct32` 的 `FA_no_carry`（仅最后一列 `col_num-1`）天然落在
非近似区，无需 `_no_carry` 近似变体。

---

## 3. 采样与 log_prob（`sample_cell_types`）

```
for node_idx, (s, c, t, idx) in 枚举 vertex_list 中 t∈{0,1}:
    head   = fc_type32 if t==0 else fc_type22
    logits = head(x_emb[node_idx])                 # [|T|]
    if c >= approx_max_col:  logits[1:] = -1e9      # 强制 exact
    dist   = Categorical(logits=logits)
    k      = dist.sample();  type_log_prob += dist.log_prob(k)
    cell_map[node_idx] = T[t][k].module             # k==0 -> None(精确)
```
- exact（k=0）→ `cell_map` 不写该节点（落默认 FA/HA）。
- `type_log_prob` 与布线的 `overall_log_prob` **相加**，统一进 PPO ratio。
- 类型采样**独立于布线**（不互相掩码），实现简单、梯度清晰。

---

## 4. RTL 发射

- `_declare_ct32`：`cell = (cell_map or {}).get(node_idx) or "FA"`，把
  `FA {inst} (.a..)` 改成 `{cell} {inst} (.a..)`；`FA_no_carry` 分支保持（近似区不会到那）。
- `_declare_ct22` 同理，默认 `HA`。
- `get_samples` 里：`used = {非None cell}`，`extra = approx_modules_src(used)`（复用 gen_verilog.emit_module），
  `mul.emit_verilog(rtl_path, assignment=assignment, extra_modules_src=extra)`。
- 综合：yosys 把近似 module 与 FA/HA 一样综合到工艺库，**STA 自动给出真实 PPA**。

---

## 5. 误差 reward

逐 sample 从采样到的 `cell_map` 闭式计算（`maxprod=(2^bw−1)^2`）：

- `bias_total = Σ_node bias(cell)·2^col`  （带符号 → 抓正负抵消）
- `wae_total  = Σ_node wae(cell)·2^col`   （E[Σ|e_local|·2^col]，MED 的保守上界）
- `NMED_analytic = wae_total / maxprod`，`bias_norm = |bias_total| / maxprod`

**两种接法（二选一，推荐 A）：**

- **A. 约束式（仿现有 `area_budget`）** ——推荐，PPA 仍是主目标：
  ```
  objective += nmed_violation_weight * max(0, NMED_analytic - nmed_budget)
  objective += bias_weight * bias_norm        # 始终压偏置(抵消是核心卖点)
  ```
- **B. 加权式**：`objective += λ_nmed*NMED_analytic + λ_bias*bias_norm`。

无论 A/B，误差都折进标量 `objective`，PPO 的 `A=-objective` 通过 §3 的 `type_log_prob` 自然训练类型头，
**不强制需要可微项**。

**可选可微 surrogate（推荐叠加，降方差）`get_error_loss`：**
对每节点 `p=softmax(type_logits)`，`E_bias=Σ_k p_k·bias_k`、`E_wae=Σ_k p_k·wae_k`，
`l_err = λ_bias·(Σ E_bias·2^col / maxprod)^2 + λ_wae·relu(Σ E_wae·2^col/maxprod − budget)`。
仿 `get_delay_loss` 在 `run_episode` 里像 `l_delay` 一样加权累加。离散类型多、REINFORCE 方差大，
这个可微项直接塑形类型分布、收敛更稳。

> 偏置/wae 用 P(1)=1/4（仅第一级 PP 精确，深层列有偏）。这是**一阶解析估计**；真实 ER/MED/NMED/WCE
> 由 `approx_mul.py` 穷举/采样事后校验（已能跑）。后续可换逐 stage 真实输入概率细化。

---

## 6. PPO 接入

`get_ppo_loss` 的 `new_log_prob` 在累加完连接项后，用**当前**类型头对 `sample_info["cell_types"]`
重算类型 log_prob（同 `approx_max_col` 掩码）并相加，使 ratio 与采样时口径一致。需要 `get_Z_mat`
回传的 `x_emb`（§1#4）。

---

## 7. 新增配置开关（全部默认关/None）

```
use_approx_types: bool = False          # 总开关
approx_lib_path: str = "Appr_Comp/selected_compressors.json"
approx_max_col: int = 6                 # 只有 col<此 可近似
approx_types_32 / approx_types_22: list = None   # None=用 selected 全集
# 误差 reward（约束式 A）
nmed_budget: float = None
nmed_violation_weight: float = 2.0
bias_weight: float = 0.0
# 可微 surrogate（可选）
use_error_loss: bool = False
error_loss_weight: float = 0.0
bias_loss_weight: float = 0.0
# PPA 评测：搜索侧用本地 yosys+openroad，近似 cell 原生支持，无需额外开关
```

---

## 8. 回归保证（默认关 = 字节级一致）

- `use_approx_types=False` 时：类型头不创建、`sample_cell_types` 不调、`cell_map=None`、误差项/误差 loss 不加、
  PPO 类型分支跳过 → 发射的 router_src 与改前**逐字符相同**。
- **唯一侵入点**：`MultiChannelResGCN.forward` 多返回一个 `x`，必须同步改 `get_Z_mat` 的解包
  （`out_a,out_b,out_c,out_sum,out_carry,x_emb = self.gcn.forward(...)`）。这是仅有的、即使关开关也变的一行。
- **回归测试**：开关关时跑 1 episode，`diff` 新旧 `MUL-0.v`；应为空。

---

## 9. 风险与已知近似

1. 误差解析用 P=1/4，深层列偏乐观/悲观 → 靠 `approx_mul.py` 事后穷举校验；可作为 reward 的一阶项接受。
2. **delay 可微 loss（`get_delay_loss`）仍用 exact FA/HA 时延常数**；近似 cell 一般更快，故是保守（偏悲观）估计。
   第一版不改 delay_loss；若要精确，可在 §3 后用类型分布加权 `tmax`（见 `delay_constants.py`），列为后续。
3. **power proxy 不认近似 cell**（proxy 用工艺库 FA/HA cell）。→ 近似搜索建议 `power_source="eda"`
   （本地 yosys+openroad 真实功耗）。proxy 支持近似 cell 列为后续。
4. 类型与布线独立采样：忽略「某 cell 对特定布线更优」的耦合，换取实现简单；先验证有效再考虑联合。

---

## 10. 实现顺序与验证（小步、每步可回归）

1. **基建**：GCN 类型头 + forward 回传 x + get_Z_mat 解包；`sample_cell_types`；`emit_assignment/_declare_*`
   透传 cell_map。→ 开关关跑 1 ep，`diff` RTL 为空（回归过）。
2. **发射通路**：开开关、`approx_max_col` 给小值，生成 1 个近似 MUL RTL，`yosys` 综合通过、出 PPA。
   用 `approx_mul.py` 对同一 cell_map 穷举 8-bit 验证误差与解析一致。
3. **reward**：接 §5-A 约束式 + bias 项进 objective；TB 记录 `NMED_analytic/bias_norm/area/power`。
4. **训练**：接 §6 PPO 类型 log_prob（+ 可选 §5 可微 loss）；短跑观察 objective、NMED、面积/功耗曲线。
5. **评估**：最优解导出 → `approx_mul.py` 穷举（小 bit）或采样（大 bit）报 ER/MED/NMED/WCE，对比 exact 基线 PPA。

---

## 11. 待你拍板的决策点

- **D1 误差接法**：约束式 A（`nmed_budget`，PPA 主目标）还是加权式 B？（推荐 A）
- **D2 可微 surrogate**：是否叠加 `get_error_loss`（推荐叠，降方差）还是纯 PPO？
- **D3 近似范围 `approx_max_col`**：先固定小值（如 6）还是设成可调超参？（推荐可调，默认 6）
- **D4 功耗来源**：近似搜索固定 `power_source="eda"`？（推荐是；proxy 暂不认近似 cell）
- **D5 delay_loss**：第一版保持 exact 时延常数（保守）可接受？（推荐可接受，精确化列后续）

---

## 12. 实现状态（2026-06-18 已完成 / 全套按推荐 D1=A,D2=开,D3=可调默认6,D4=eda,D5=保守）

全部落在 [trainer/arith_das.py](../trainer/arith_das.py)，**默认全关，关时行为字节级一致**（已测）。

| 改动 | 位置 | 状态 |
|---|---|---|
| GCN 加 `embed()` + `forward(return_embedding=)` + `embedding_dim` | `MultiChannelResGCN` | ✅ |
| `__init__` 加配置开关 + 类型表加载 + 类型头 + 优化器并入头参数 | `CompressorRouting.__init__` | ✅ |
| `_load_approx_types/_resolve_path/sample_cell_types/_approx_modules_src/_analytic_error` | 新方法 | ✅ |
| `get_Z_mat` 按开关回传并缓存 `self._node_emb` | `get_Z_mat` | ✅ |
| `_declare_ct32/22` + `emit_assignment` 透传 `cell_map`（默认 FA/HA 串不变） | 发射 | ✅ |
| `get_samples` 采样类型→注入近似 module→存 `cell_types` | `get_samples` | ✅ |
| 约束式误差项进 `get_objective(cell_types=)` | reward | ✅ |
| `get_error_loss`（D2 可微 surrogate）+ `run_episode` 按 `use_error_loss` 接入 | 训练 | ✅ |
| `get_ppo_loss` 重算类型 log_prob 并入 `new_log_prob` | PPO | ✅ |
| **持久化缺口修复**：`found_best_info` 存 `cell_types`；`export_best_candidate` 与 `get_full_target_delay_result` 用 `_cell_map_from_types` 复原 cell_map → emit 带近似 cell（否则导出/评估退化成纯 FA/HA） | best/export | ✅ |

**已验证**：(1) 关开关 emit 三路径(default/None/{})逐字符一致、`sample_cell_types` 不耗 RNG、ppo+optim.step 正常；
(2) 开开关 RTL well-formed（实例化的近似 cell 全有 module 定义、端口与 FA/HA 对齐、近似只在 col<max_col）；
(3) `get_error_loss` 与 PPO 类型分支都能把梯度回传到两个类型头；(4) 误差项使 objective 单调增（非负惩罚）。

**现成 config**：[configs/config_groups/mul_8_and_approx.yaml](../configs/config_groups/mul_8_and_approx.yaml)、
[configs/config_groups/mul_16_and_approx.yaml](../configs/config_groups/mul_16_and_approx.yaml)（均已 OmegaConf 加载+构建验证）。
跑法：`python pipeline.py +config_groups=mul_16_and_approx`。

**开启示例**（加到某 config 的 `trainer.kwargs`）：
```yaml
    use_approx_types: true
    approx_lib_path: Appr_Comp/selected_compressors.json   # 相对仓库根（pipeline 会 chdir）
    approx_library_path: Appr_Comp/library.json
    approx_max_col: 6
    # 误差约束（D1=A），LSB 绝对单位（med=Σwae·2^col，MED 上界）。NMED=med/maxprod 仅上报。
    med_budget: 8.0                  # 容忍的 MED 上限（输出 LSB），超出才罚
    med_violation_weight: 0.1
    bias_weight: 0.05
    # D2 可微 surrogate（随时可关）
    use_error_loss: true
    error_loss_weight: 0.1
    bias_loss_weight: 0.05
    # D4：近似搜索建议 eda 功耗（proxy 暂不认近似 cell）
    power_source: eda
```

---

## 13. 缺陷自查（2026-06-18，已修 3 项 + 2 项已知限制）

| # | 缺陷 | 严重度 | 处置 |
|---|------|--------|------|
| 1 | **误差项归一化失效**：med/bias 除以 maxprod=4^n，8bit NMED~1e-4、**16bit~1e-9**，远小于 PPA O(1)；`nmed_budget=1e-3` 16bit 永不触发，bias² 梯度消失（3.7e-17）→ 误差/抵消激励形同虚设 | **高** | ✅ 改用 **LSB 绝对单位** med=Σwae·2^col、bias=\|Σbias·2^col\|（跨位宽稳定、梯度 O(1)）；bias 用 L1 非 L2；配置键 `nmed_*`→`med_*`(LSB)。修后 objective 误差项 2.7（vs PPA~5）、grad\|sum\| 65.7 |
| 2 | **类型头未进 checkpoint**：`save_experiment` 只存 `gcn.state_dict()`，类型头是 self 上独立 Linear → resume/重载丢失已学类型策略 | 中 | ✅ `use_approx_types` 时额外存 `type_heads.pth`（最优设计本身经 best_info.cell_types 仍可复原，此修针对续训/推理重载） |
| 3 | **masking 用 in-place** `forced[0]=logits[0]`（3 处），autograd 语义含糊 | 低 | ✅ 统一 `_masked_type_logits`（masked_fill，非 in-place） |
| 4 | **area_budget + 近似 混用时** `_candidate_rank` 在 budget 模式按 (可行性,越界,power) 排名、**不含误差项** → 导出的 best 选择忽略 MED（仅影响 best 选择，不影响 PPO 梯度）。当前两个 config 都不用 area_budget，不触发 | 已知限制 | ⏭ 未改；若要 budget+近似，再把 MED 纳入 budget 模式排名 |
| 5 | 类型头按节点 Python 循环（sample/error/ppo 各 O(节点数) 个小 Linear）。16bit 210 节点，但搜索是 **EDA-bound**（每样本综合数秒），此开销可忽略 | 已知限制 | ⏭ 未改；如需可按 arity 批量化 head 前向 |

已验证（das env）：修后全套回归——关开关字节级一致、开开关 RTL well-formed、误差项有量级、error_loss/PPO 梯度非消失回传类型头、export round-trip 带近似 cell、type_heads.pth 可存读、两 config OmegaConf 构建通过。

**尚未在本机做的**（无 yosys/openroad/iverilog 于当前 shell）：真实综合 PPA、功能仿真测真误差。
应在带 EDA 工具的 pipeline 环境跑：开开关短训几 episode 看 yosys 是否综合通过、objective/NMED/area 曲线；
最优解用 `approx_mul.py`（或对发射 RTL 做功能仿真）穷举校验 ER/MED/NMED/WCE。
