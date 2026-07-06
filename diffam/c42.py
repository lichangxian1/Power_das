"""小型近似 4:2 compressor 库的可微代理模型。

真实候选单元在 c42_inst.data 里，是离散真值表。这个文件给它们包一层
PyTorch surrogate，让后面的乘法器搜索可以用梯度下降：

1. C42Feature 学 selector -> error pattern / area。
2. C42Behavior 学 input bits + selector -> carry / sum。
3. AM 训练时只保留 C42Behavior.par 可训练。前向里它被 round 到 5 个库
   索引之一，反向用 STE 更新连续参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import json
import torch
from torch.utils.data import Dataset
import util
import c42_inst


class C42FeatureDataset(Dataset):
    def __len__(self):
        return len(c42_inst.data)

    def __getitem__(self, idx):
        i = idx % len(c42_inst.data)

        # 目标 feature：16 项局部误差表 + 归一化面积。
        # 对每个 4-bit 输入 pattern，精确输出值是 popcount(input)。
        # compressor 输出值是 2 * carry + sum。
        t = torch.Tensor(c42_inst.data[i]["map"])
        t = t[:, 0] * 2 + t[:, 1] - torch.Tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4])
        t = torch.concat([t, torch.Tensor([c42_inst.data[i]["area"] / 7.0])])

        # selector 归一化到 [0, 1]：e1=0/4, ..., e8=4/4。
        return torch.Tensor([i / 4]), t


class C42Feature(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.mlp = util.MLP(1, 16 + 1)

    def forward(self, x):
        y = self.mlp(x)
        return y


class C42BehaviorDataset(Dataset):
    def __len__(self):
        # 5 种 compressor * 16 种 4-bit 输入 pattern。
        return len(c42_inst.data) * 16

    def __getitem__(self, idx):
        i = int(idx / 16)
        j = idx % 16

        # 输入 = 4 个 Boolean compressor 输入 + 归一化类型 id。
        input = torch.tensor([int(j / 8), int((j % 8) / 4), int((j % 4) / 2), int(j % 2), i / 4]).to(dtype=torch.float)

        # 目标 = 所选真值表里的 [carry, sum]。
        target = torch.tensor([c42_inst.data[i]["map"][j][0], c42_inst.data[i]["map"][j][1]]).to(dtype=torch.float)

        return input, target


class C42Behavior(nn.Module):
    def __init__(self, feature) -> None:
        super().__init__()
        # 可训练 compressor selector。AM 训练会给每个 compressor slot
        # deepcopy 一份，并主要优化这个标量。
        self.par = nn.Parameter(torch.ones(1))
        self.feature = feature
        self.mlp = util.MLP(4 + 16, 64, 2)

    def forward(self, x, split=False):
        assert len(x.shape) == 2

        if x.shape[-1] == 4 + 1:
            # c42 监督训练阶段：输入里显式带 selector。
            is_fixed = False
        elif x.shape[-1] == 4:
            # AM 模式：使用该模块自己的可训练 selector 参数。
            is_fixed = True
            x = torch.concat((x, self.par.broadcast_to([x.shape[0], 1])), dim=1)
        else:
            raise NotImplementedError

        t = x[:, 4:]
        # self.t0 = t
        # self.t0.retain_grad()
        if is_fixed:
            # 把 selector 量化到 {0, .25, .5, .75, 1} 之一。
            # round_pass() 保持离散前向行为，同时允许梯度更新 self.par。
            t = util.round_pass((t * 4)).clamp(0, 4) / 4
        # self.t1 = t
        # self.t1.retain_grad()
        self.t = t = self.feature(t)
        # self.t.retain_grad()

        # feature 网络提供所选 compressor 类型的可微描述。
        # 前 16 个输出是局部误差表，作为 carry/sum 行为网络的条件特征。
        err_pattern = t[:, 0:16]  # + torch.ones_like(t[:,0:16]) * torch.Tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]).cuda()

        y = self.mlp(torch.concat((x[:, 0:4], err_pattern), dim=1))

        # y = torch.sigmoid(y)

        if is_fixed:
            # 真实电路里的 carry/sum 是 bit，所以 AM 模式下也要 round。
            y = util.round_pass(y)

        # if is_fixed:
        #     _y = torch.empty_like(y)
        #     _y[:, 0:2] = util.round_pass(y[:, 0:2].clamp(0, 1))
        #     _y[:, 2] = y[:, 2]
        #     y = _y

        if split:
            # split=True 方便 AM.forward() 分别处理 carry、sum 和 area：
            # carry/sum 参与连线，area 参与面积累加。
            return {"c": y[:, 0:1], "s": y[:, 1:2], "a": t[:, 16:]}
        else:
            return torch.concat((y, t[:, 16:]), dim=1)
