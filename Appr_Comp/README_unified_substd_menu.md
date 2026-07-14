# 统一 substd 菜单（一份 json 管 22/32/42）

**文件**：`Appr_Comp/selected_compressors_all_substd.json`（`gen_unified_substd_menu.py` 生成）
**准则**：菜单里每个近似 cell 的 PPA 都优于原生标准单元（substd 硬标准或 in-context 探针）。

## 内容（exact 锚点 + 全 substd）
| 类型 | exact 锚点 | 近似 cell（全部 PPA<std） |
|---|---|---|
| T22 (2:2) | comp22_94 | e4, 50, a0, 00, f0, 55（standalone<HA1D0=2.184）|
| T32 (3:2) | comp32_e994 | fa50, 5500, aa00, 5555, ff00, ff55（standalone<FA1D0=2.856）|
| T42 (4:2) | CT42 | comp42s: orha_n/z/p, or4ao_n, or4oa_p, or4_n, thru_n |

T42 的 or4_n(5.376)/thru_n(0.336) 过 standalone 硬标准；orha_n/z/p、or4ao/or4oa
过 **in-context 探针**（整网表 DC 实测真实 ~1.2-1.7µm² « 2xFA1D0=5.712，
见 `outputs/2026-07-12_orha_probe/PROBE_ORHA.md`）。
**排除**：原 12 个 comp42n（standalone 10.9-17.1µm²，全在 std 之上，无探针证据）。

## 文件结构（一文件三用）
- `selected`：训练菜单本体（22/32/42 都在此选，按 `type` 字段分流）
- `cells`：42 的 LUT 源（7 个 comp42s 完整真值表，供 native-4 免-cin 识别）
- `meta.anchor_area`=17.304：exact CT42 锚点面积（外环 area 打分用）

## 训练怎么配（4 个路径）
```bash
--approx_lib_path        Appr_Comp/selected_compressors_all_substd.json   # 22/32/42 菜单
--approx_library_path    Appr_Comp/library.json                          # 22/32 LUT backing
--use_ct42                                                               # 开 T42
--approx42_library_path  Appr_Comp/selected_compressors_all_substd.json   # 42 LUT+锚点(同一文件)
--approx42_rtl_path      Appr_Comp/rtl/comp42n_lib.v                     # 42 结构化 RTL backing
```
`library.json` 与 `comp42n_lib.v` 是**组件库/RTL backing**（不是菜单），架构使然必须保留：
22/32 的 Verilog 从 LUT 当场 emit、42 的 Verilog 按名从 comp42n_lib.v 抽取。

## 关键行为差异（vs 旧的分离式 42 菜单）
菜单含 `type==42` 条目 → loader 走 `_load_approx42_table` **Branch A（显式菜单）**：
- ✅ **不再受 `--approx42_max_types` 截断**（旧的 20-cell 截断坑自动消失，可不传该参数）
- ⚠ 必须含一个 `group==exact` 的 CT42 条目（本菜单已含 `comp42_exact`）

## 验证
`python scripts/smoke_unified_substd_menu.py`（EDA-free）：A loader 三表 / B 19 个近似
module 可发射(42 免 cin) / C 整树 emit+verilator 编译过 / D Branch A 无视 max_types。已全过。

## 重新生成
`python -m Appr_Comp.gen_unified_substd_menu`（幂等；改 substd 选型或 comp42s 后重跑）。
