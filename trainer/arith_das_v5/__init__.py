"""ARITH-DAS v5 训练器（trainer/arith_das.py 的 v5 精简迁移版）。

来源：trainer/arith_das.py（v5 架构，commit 8c59826 时点）。本文件只保留
scripts/train_dc.py --pareto_v5 训练闭环实际可达的代码路径，可达部分逐行照搬
原实现（v5 流程下功能与原文件完全一致），剪掉以下确定不可达的遗留部分：

  - power proxy（功耗代理模型）：train_dc 固定 power_source="eda"、
    use_power_proxy=False，proxy 分支不可达；
  - get_delay_loss / DELAY_CONSTANT（可微延迟 surrogate）：所有 config 均
    use_delay_loss=false，且 v5 固定单一 DC 周期、delay_weight=0；
  - get_error_loss 及 _bias/_wae/_maxe 张量缓存（可微误差 surrogate）：
    v5 要求 error_metric=mred，train_dc 对 mred 强制 use_error_loss=False；
  - area_budget 面积预算约束模式：train_dc 固定 area_budget=None；
  - 本地 openroad/ABC 综合主路径：train_dc 固定 synth="dc"（DC 失败后的
    openroad 回退调用保留，与原实现一致——该回退在远端环境的问题见
    v5 r2 审查清单，修复待批，此处不擅改）；
  - gomil 温启动、get_pool_objectives、get_masked_logits 等仓库级零引用代码。

被剪掉的功能在构造函数里有显式守卫：传入相应开关会直接 ValueError 报错并
提示改用 trainer.arith_das（旧全量实现），不会静默改变行为。

术语速记：PPO=近端策略优化（策略梯度 RL 算法）；PPA=power/performance/area
（功耗/时序/面积）；MED=平均误差距离（绝对误差期望，LSB 单位）；MRED=平均
相对误差；DC=Synopsys Design Compiler（远端综合直出 PPA）；Dadda 树=一种
经典压缩树乘法器结构；4:2 compressor（CT42）=一次压 4 个部分积位的压缩单元。
"""
from .compressor_graph import CompressorGraph
from .core import CompressorRouting
from .networks import (
    ConfigurableGCN,
    MultiChannelResGCN,
    MultiChannelResGCNBlock,
)

__all__ = [
    "CompressorGraph",
    "CompressorRouting",
    "ConfigurableGCN",
    "MultiChannelResGCN",
    "MultiChannelResGCNBlock",
]
