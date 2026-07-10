# diffam 式 cell 维度梯度求解器（OUTER_CELL_SEARCH.md §3.2 步骤⑤的架构原型）。
# sim.py    精确 TT 张量化压缩树仿真器（整数模式对拍 verilator；多线性模式可微）
# solver.py 每 slot logits + hard 前向/STE 反向 + MRED hinge + 面积项
# demo_solve.py 端到端：对拍验证 → 求解 → verilator 终验 → 对比 GA cell 包
