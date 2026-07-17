# arith_das_v5 包结构与阅读指南

原来的单文件 `trainer/arith_das_v5.py`（4761 行）已拆成本包。拆分是**纯代码搬运**：
每个方法逐行原样搬入对应模块，已用 ast 逐方法对拍验证语法树一致、4 个 smoke
脚本对拍输出一致，行为与拆分前完全相同。对外 import 路径不变：

```python
from trainer.arith_das_v5 import CompressorRouting, CompressorGraph   # 照旧可用
from trainer import CompressorRoutingGeneArchV5                        # 照旧可用
```

## 拆分方式：mixin（混入类）

`CompressorRouting` 原来是一个 4300 行的巨型类。现在按主题切成 10 个
**mixin**——所谓 mixin 就是"只装一组方法的部分类"，每个文件放一个主题的方法，
最后在 [core.py](core.py) 里用多重继承拼回唯一的 `CompressorRouting`：

```python
class CompressorRouting(CellTypeMixin, OuterSearchMixin, ..., EnvironmentMixin):
    def __init__(...):   # 全部超参数和 self.xxx 属性都在这里定义
```

读代码时记住两条：
- **所有 `self.xxx` 属性都在 `core.py` 的 `__init__` 里定义**，各 mixin 只读写它们；
- mixin 之间通过 `self.方法名()` 互相调用，跨文件跳转直接搜方法名即可。

## 文件地图

| 文件 | 职责 | 行数 |
|---|---|---|
| [core.py](core.py) | `CompressorRouting` 类本体：`__init__`（全部超参/状态）+ 通用小工具 | 590 |
| [training.py](training.py) | ★ **训练机制**：`run_experiment`/`run_episode` 主循环、objective(reward)、advantage 分组归一、PPO loss、日志 | 479 |
| [sampling.py](sampling.py) | 内环采样：GCN 前向 + 按动作掩码逐步采样布线、组装样本（`get_samples`） | 337 |
| [networks.py](networks.py) | GCN 策略网络（多通道残差 GCN） | 175 |
| [compressor_graph.py](compressor_graph.py) | 连接矩阵 → GCN 输入图的翻译 | 266 |
| [environment.py](environment.py) | 动作空间环境：`reset`/`transition`/`get_action_mask`/legalize | 375 |
| [cell_types.py](cell_types.py) | 近似 cell 类型库加载 + 逐 cell 类型采样（cardinality sampler） | 458 |
| [outer_search.py](outer_search.py) | 外环 cell 配置搜索：变异算子/杂交/bandit 骰子/预算过滤/预筛门/greedy 求解器 | 804 |
| [truncation.py](truncation.py) | k 截断档：截断常数 C*、解析误差（MED/MRED 闭式） | 208 |
| [rtl_emit.py](rtl_emit.py) | 连接矩阵 + cell 映射 → Verilog 乘法器 RTL | 383 |
| [simulate.py](simulate.py) | 远端 DC 综合 worker、verilator 实测误差、结果汇总 | 284 |
| [pareto_front.py](pareto_front.py) | v5 非支配档案：mred 分箱准入/亲代采样/代表解/前沿导入导出 | 349 |
| [persistence.py](persistence.py) | 策略权重加载、实验目录保存、PPA 诊断、最优解导出 | 266 |

## 训练机制阅读路线（从这里开始读）

一次训练 run 的主线调用链，括号里是"文件 :: 方法"：

```
run_experiment (training.py)              # 最外层 for episode 循环
└─ run_episode (training.py)              # 一个 episode = 采样一批 → 更新一次策略
   ├─ _v5_begin_episode (pareto_front.py) # 从前沿档案采一个亲代状态 + 外环变异 cell 配置
   ├─ reset (environment.py)              # 重置压缩树状态
   ├─ get_samples (sampling.py)           # ★ 内环：采 batch_size 个设计并评估
   │  ├─ sample_from_logits (sampling.py) #   GCN 前向 → 逐步采样布线动作
   │  ├─ sample_cell_types (cell_types.py)#   给每个压缩器采近似类型
   │  ├─ emit_assignment (rtl_emit.py)    #   翻译成 Verilog
   │  ├─ parallel_simulate_worker (simulate.py) # 远端 DC 出 PPA + verilator 实测误差
   │  └─ get_objective (training.py)      #   PPA+误差 → 标量 objective（越小越好）
   ├─ update_found_best_info (training.py)# 维护当前最优设计
   ├─ get_ppo_loss (training.py)          # ★ advantage 组内归一 + PPO clip loss
   │      （num_epochs 次：backward → clip_grad → optim.step）
   ├─ _v5_admit_samples (pareto_front.py) # 样本按 (mred分箱, area, power) 进非支配档案
   └─ log_episode (training.py)           # tensorboard/日志
```

想理解 reward 怎么算，读 `training.py::get_objective`；
想理解策略怎么更新，读 `training.py::get_ppo_loss` 和 `run_episode` 的 epoch 循环；
想理解样本从哪来，读 `sampling.py::get_samples`。

## 与原单文件的两处有意差异（其余逐字一致）

1. `core.py` 的 `_REPO_ROOT`：文件比原来深一层目录，多剥一层 `dirname`，
   运行时值不变（仍是仓库根）。
2. `simulate.py` 的 `get_full_target_delay_result` / `parallel_simulate_worker`
   体首各加一行 `from .core import CompressorRouting`：原文件在方法体内按类名
   调用同伴静态方法，包内直接顶层 import 会和 core 循环导入，故改为调用时延迟
   导入；仍按类名查找，monkeypatch 语义不变。

## 历史追溯

拆分前的完整单文件历史：`git log -- trainer/arith_das_v5.py`（拆分提交之前）。
更早的全量实现在 `trainer/arith_das.py`（v5 不可达路径未搬入本包，构造函数里
有显式守卫，误开相应开关会直接 ValueError）。
