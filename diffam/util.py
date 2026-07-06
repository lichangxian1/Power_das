import os
import random
import glob
import json
import numpy as np
import torch
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn


def seed_all(seed):
    """固定 Python、NumPy、PyTorch 的随机种子，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def onehot(i: int, total: int):
    lst = [0 for _ in range(total)]
    lst[i] = 1.0
    return lst


class AverageMeter:
    def __init__(self, fmt="%.6f"):
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def round_pass(x):
    """前向传播做 round，反向传播把梯度当作 identity 传回去。

    这是 compressor selector 用到的 straight-through estimator：
    电路前向看到的是离散选择，但 autograd 仍然可以更新背后的连续参数。
    """
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MLP(nn.Module):
    def __init__(self, *nlst) -> None:
        super().__init__()

        # nlst 是紧凑的层宽描述，例如 MLP(4, 64, 2)。
        n_input = nlst[0]
        n_output = nlst[-1]
        n_hidden_lst = nlst[1:-1]

        layers = []
        last_n = n_input
        for n in n_hidden_lst:
            layers.append(nn.Linear(last_n, n))
            layers.append(nn.ReLU())
            last_n = n

        layers.append(nn.Linear(last_n, n_output))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
