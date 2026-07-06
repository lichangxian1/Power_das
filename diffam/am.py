"""由可微 4:2 compressor 组成的 8-bit 近似乘法器。

这个文件里的乘法器架构是固定的。训练不会改连线，只会改每个 slot 里的
C42Behavior.par，从而选择 c42_inst.data 中哪一种近似 compressor。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from copy import deepcopy
import torch
from torch.utils.data import Dataset
import util
import c42
import c42_inst


def gpy(a, b):
    """调试 get_py() 用的小工具：输入两个标量。"""
    a = torch.Tensor([a]).to(dtype=torch.uint8)
    b = torch.Tensor([b]).to(dtype=torch.uint8)
    return get_py(a, b)


def get_py(a, b):
    """构造 8x8 unsigned 部分积矩阵的低 8 列。

    返回：
        p: shape [N, 8, 8]。p[row, col] 是移位后的部分积 bit。
        y: 保留下来的低 8 列的精确数值，还没有经过 compressor 近似。

    高列没有表示出来；这份代码搜索的是截断后的 8-bit product model。
    """
    assert a.dtype == torch.uint8
    assert b.dtype == torch.uint8

    a = a.unsqueeze(dim=1)
    b = b.unsqueeze(dim=1)

    a0 = a % 2
    a1 = (a % 4) >> 1
    a2 = (a % 8) >> 2
    a3 = (a % 16) >> 3
    a4 = (a % 32) >> 4
    a5 = (a % 64) >> 5
    a6 = (a % 128) >> 6
    a7 = (a) >> 7

    b0 = b % 2
    b1 = (b % 4) >> 1
    b2 = (b % 8) >> 2
    b3 = (b % 16) >> 3
    b4 = (b % 32) >> 4
    b5 = (b % 64) >> 5
    b6 = (b % 128) >> 6
    b7 = (b) >> 7

    zzzzzzz = torch.zeros_like(b7)

    # 每个 pK 是由 bK & a* 生成的一行移位部分积。
    # 这里只保留 8 列，所以 p7 只剩最低的那个有效 bit。
    p0 = torch.concat([b0 & a0, b0 & a1, b0 & a2, b0 & a3, b0 & a4, b0 & a5, b0 & a6, b0 & a7], dim=1)
    p1 = torch.concat([zzzzzzz, b1 & a0, b1 & a1, b1 & a2, b1 & a3, b1 & a4, b1 & a5, b1 & a6], dim=1)
    p2 = torch.concat([zzzzzzz, zzzzzzz, b2 & a0, b2 & a1, b2 & a2, b2 & a3, b2 & a4, b2 & a5], dim=1)
    p3 = torch.concat([zzzzzzz, zzzzzzz, zzzzzzz, b3 & a0, b3 & a1, b3 & a2, b3 & a3, b3 & a4], dim=1)
    p4 = torch.concat([zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, b4 & a0, b4 & a1, b4 & a2, b4 & a3], dim=1)
    p5 = torch.concat([zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, b5 & a0, b5 & a1, b5 & a2], dim=1)
    p6 = torch.concat([zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, b6 & a0, b6 & a1], dim=1)
    p7 = torch.concat([zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, zzzzzzz, b7 & a0], dim=1)

    p = torch.concat([p0, p1, p2, p3, p4, p5, p6, p7], dim=1).view((-1, 8, 8)).to(dtype=torch.float)

    # 保留列的精确值。它主要用于调试；
    # AMDataset 后面会把标签替换成 all-e1 reference multiplier 的输出。
    y = p.sum(dim=1)
    y = y[:, 7] * 128 + y[:, 6] * 64 + y[:, 5] * 32 + y[:, 4] * 16 + y[:, 3] * 8 + y[:, 2] * 4 + y[:, 1] * 2 + y[:, 0] * 1
    return p, y


class AMDataset(Dataset):
    def __init__(self, a, b):
        # 从所有输入 pair 出发，先生成部分积。
        self.p, self.y = get_py(a, b)

        # 构造 reference multiplier：把每个 compressor selector 强制为 0，
        # 也就是选择 c42_inst.data[0] == e1。
        feature = c42.C42Feature()
        behavior = c42.C42Behavior(feature)
        behavior.load_state_dict(torch.load("c42.pt")["model"])
        behavior.requires_grad_(False)
        model = AM(behavior)
        model.cuda()
        model.set_all_par_zero()
        self.ref = model

        # 训练标签不是原始 a*b，而是固定架构下所有 compressor slot 都用 e1
        # 时的输出。
        self.y = self.ref(self.p.cuda())[0].cpu()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        return self.p[idx], self.y[idx]


class AM(nn.Module):
    def __init__(self, c: c42.C42Behavior) -> None:
        super().__init__()
        # 三层固定 compressor。每个 slot 都是独立 deepcopy，
        # 因此每个 slot 都有自己的可训练 selector par。
        self.ln0 = nn.ModuleList([deepcopy(c) for _ in range(8)])
        self.ln1 = nn.ModuleList([deepcopy(c) for _ in range(8)])
        self.ln2 = nn.ModuleList([deepcopy(c) for _ in range(8)])

    def set_all_par_zero(self):
        """强制所有 slot 选择库里的第一个 cell，即 e1。"""
        for c in [*self.ln0, *self.ln1, *self.ln2]:
            c.par.data = torch.zeros_like(c.par.data)

    # def kv_loss(self):
    #     y = 0
    #     for c in [*self.ln0, *self.ln1, *self.ln2]:
    #         y += c.kv_loss()
    #     return y

    def pf(self):
        self.print_features()

    # def print_features(self):
    #     def f(ln: list[c42.C42Behavior], start, end):
    #         lst = []
    #         for i in range(start, end, -1):
    #             lst.append(ln[i].check_features())
    #         print(lst)

    #     f(self.ln0, 7, -1)
    #     f(self.ln1, 7, 3)
    #     f(self.ln2, 7, -1)

    def print(self, sel=False):
        def f(ln: list[c42.C42Behavior], start, end):
            lst = []
            for i in range(start, end, -1):
                t = float(ln[i].par.clamp(0, 1))
                if sel:
                    # 打印原始 selector 数值，方便观察收敛。
                    lst.append(t)
                else:
                    # 把 selector 映射成最接近的离散 compressor 名字。
                    idx = int(round(t * 4))
                    if idx >= len(c42_inst.data) or idx < 0:
                        label = "_" * len(c42_inst.data[0]["name"])
                    else:
                        label = c42_inst.data[idx]["name"]
                    lst.append(label)
            print(lst)

        f(self.ln0, 7, -1)
        f(self.ln1, 7, 3)
        f(self.ln2, 7, -1)

    def forward(self, p):
        # 第 0 层：按列压缩部分积 row 0..3。
        self.t0 = t0 = []
        for c42 in self.ln0:
            # t0[0]['c']: carry, t0[0]['s']: sum, t0[0]['a']: area
            u = c42(p[:, 0:4, len(t0)], True)
            u['c'].requires_grad_(True).retain_grad()
            u['s'].requires_grad_(True).retain_grad()
            u['a'].requires_grad_(True).retain_grad()
            t0.append(u)

        # 第 1 层：按列压缩部分积 row 4..7。
        self.t1 = t1 = []
        for c42 in self.ln1:
            u = c42(p[:, 4:8, len(t1)], True)
            u['c'].requires_grad_(True).retain_grad()
            u['s'].requires_grad_(True).retain_grad()
            u['a'].requires_grad_(True).retain_grad()
            t1.append(u)

        zzzzzzzzzz = torch.zeros_like(t0[0]["s"])
        self.t2 = t2 = []
        # 第 2 层：固定 reduction tree。第 i-1 列的 carry 进入第 i 列。
        # 不足 4 个输入的列用 0 补齐。
        t2.append(self.ln2[0](torch.concat((t0[0]["s"], zzzzzzzzzz, zzzzzzzzzz, zzzzzzzzzz), dim=1), True))
        t2.append(self.ln2[1](torch.concat((t0[1]["s"], zzzzzzzzzz, zzzzzzzzzz, zzzzzzzzzz), dim=1), True))
        t2.append(self.ln2[2](torch.concat((t0[2]["s"], t0[1]["c"], zzzzzzzzzz, zzzzzzzzzz), dim=1), True))
        t2.append(self.ln2[3](torch.concat((t0[3]["s"], t0[2]["c"], zzzzzzzzzz, zzzzzzzzzz), dim=1), True))

        t2.append(self.ln2[4](torch.concat((t0[4]["s"], t0[3]["c"], t1[4]["s"], zzzzzzzzzz), dim=1), True))
        t2.append(self.ln2[5](torch.concat((t0[5]["s"], t0[4]["c"], t1[5]["s"], zzzzzzzzzz), dim=1), True))
        t2.append(self.ln2[6](torch.concat((t0[6]["s"], t0[5]["c"], t1[6]["s"], t1[5]["c"]), dim=1), True))
        t2.append(self.ln2[7](torch.concat((t0[7]["s"], t0[6]["c"], t1[7]["s"], t1[6]["c"]), dim=1), True))

        for u in self.t2:
            u['c'].requires_grad_(True).retain_grad()
            u['s'].requires_grad_(True).retain_grad()
            u['a'].requires_grad_(True).retain_grad()


        sum = zzzzzzzzzz.clone()
        # 第 i 列 compressor 的 sum 权重是 2^i，carry 权重是 2^(i+1)，
        # 所以这里写成 (s + 2*c) * 2^i。
        for i, x in enumerate(t2):
            sum += (x["s"] + x["c"] * 2) * 2**i

        # 保留的最高列上，第 0/1 层还有最终 carry 需要补进来。
        sum += 2**7 * (t0[7]["c"] + t1[7]["c"])

        area = zzzzzzzzzz.clone()
        # 面积目标是所有被选 compressor 的 area proxy 之和。
        for c in [*t0, *t1, *t2]:
            area += c["a"]

        return sum[:, 0], area[:, 0]
