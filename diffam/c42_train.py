"""训练可微 4:2 compressor surrogate。

这个脚本通常先于 am_train.py 运行。流程是：

1. 先训练 C42Feature，让它复现每个库单元的 error pattern 和 area。
2. 冻结 feature 网络，再训练 C42Behavior，让它复现真值表里的 carry/sum。

保存出来的 c42_new.pt 可以作为后续 AM 训练使用的 behavior model checkpoint。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import util
import c42

util.seed_all(10032)


def train_feature():
    """拟合 selector -> [16 项局部误差, area]。"""
    model = c42.C42Feature()
    model.cuda()

    dataset = c42.C42FeatureDataset()
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 1)

    for ep in range(2000):
        for input, target in train_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            # 对 5 种 compressor 做 full-batch 回归。
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        if ep % 400 == 0:
            print(loss.data)
            # for input, target in train_loader:
            #     input = input.cuda()
            #     target = target.cuda()
            #     output = model(input)
            #     ok = (output[:, 0:3] < target[:, 0:3] + 0.2) & (output[:, 0:3] > target[:, 0:3] - 0.2)
            #     ok[:, 2] = (output[:, 2] < target[:, 2] + 0.1) & (output[:, 2] > target[:, 2] - 0.1)
            #     ok = ok[:, 0] & ok[:, 1] & ok[:, 2]
            # print(loss.data, ok.sum() / torch.ones_like(ok).sum())
    return model


def train(feature):
    """拟合 [4 个输入 bit, selector] -> [carry, sum]。"""
    model = c42.C42Behavior(feature)
    model.cuda()

    dataset = c42.C42BehaviorDataset()
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 1)

    for ep in range(2400):
        for input, target in train_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            # area 由冻结的 feature model 产生；这一阶段只监督 Boolean 行为。
            loss = criterion(output[:, 0:2], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        if ep % 400 == 0:
            for input, target in train_loader:
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                ok = (output[:, 0:2] < target[:, 0:2] + 0.1) & (output[:, 0:2] > target[:, 0:2] - 0.1)
                # ok[:, 2] = (output[:, 2] < target[:, 2] + 0.1) & (output[:, 2] > target[:, 2] - 0.1)
                ok = ok[:, 0] & ok[:, 1]  # & ok[:, 2]
            print(loss.data, ok.sum() / torch.ones_like(ok).sum())
    return model


# 阶段 1：学习所有候选 compressor 的紧凑 feature。
feature = train_feature()
feature.requires_grad_(False)

# 阶段 2：学习由 feature 条件控制的可微 carry/sum 预测器。
model = train(feature)
torch.save({"model": model.state_dict()}, "c42_new.pt")
