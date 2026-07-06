"""训练固定 8-bit 近似乘法器里的 compressor 选择。

架构来自 am.AM，这里不搜索连线。这个脚本只更新每个 C42Behavior.par
selector，在输出误差和所选 compressor 的总 area proxy 之间做权衡。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import util
import c42
import am
from am import AMDataset, AM
from tqdm import tqdm
from torch.distributions.normal import Normal


def tgpy(a, b):
    """调试工具：比较保留列的原始 target 和当前 AM 输出。"""
    t = float(am.gpy(a, b)[1])
    s = float(model(am.gpy(a, b)[0].cuda())[0].data)
    print("target", t)
    print("output", s)


# torch.autograd.set_detect_anomaly(True)

util.seed_all(10032)


# cd = torch.load("am_cd_resnet18.pt")
# dataset = AMDataset(cd["a"].to(dtype=torch.uint8).cpu(), cd["b"].to(dtype=torch.uint8).cpu())

Normal(0, 1).sample((32,))

N = 10240 * 32

# 下面的正态采样数据集只是保留的实验入口。
# 后面会被 exhaustive 256x256 sweep 覆盖。
ds_std = 16
ds_a = Normal(0, ds_std).sample((N,)).round().abs().clamp(0, 255).to(dtype=torch.uint8)
ds_b = Normal(0, ds_std).sample((N,)).round().clamp(-128, 127).add(128).to(dtype=torch.uint8)


# ds_a = torch.randint(0, 255, (N,)).round().abs().clamp(0, 255).to(dtype=torch.uint8)
# ds_b = torch.randint(0, 255, (N,)).round().abs().clamp(0, 255).to(dtype=torch.uint8)

# ds_a = (torch.ones(N)*33).to(dtype=torch.uint8)
# ds_b = (torch.ones(N) * 255).to(dtype=torch.uint8)

# exhaustive unsigned 8-bit 输入空间：全部 65,536 个 pair。
ds_a = torch.linspace(0, 255, 256).unsqueeze(0).repeat([256, 1]).flatten().to(dtype=torch.uint8)
ds_b = torch.linspace(0, 255, 256).unsqueeze(1).repeat([1, 256]).flatten().to(dtype=torch.uint8)

# dataset = AMDataset(torch.randint(0, 255, (N,)).to(dtype=torch.uint8), torch.randint(0, 255, (N,)).to(dtype=torch.uint8))
dataset = AMDataset(ds_a, ds_b)

train_loader = DataLoader(dataset, batch_size=65536, shuffle=True)
val_loader = DataLoader(dataset, batch_size=80, shuffle=False)


feature = c42.C42Feature()
behavior = c42.C42Behavior(feature)

# 加载已经训练好的可微 compressor surrogate。
behavior.load_state_dict(torch.load("c42.pt")["model"])

# 冻结 surrogate 本身。AM 训练应该只移动 AM(...) 内部 deepcopy 出来的
# per-slot par selector。
behavior.feature.requires_grad_(False)
behavior.mlp.requires_grad_(False)

# breakpoint()

model = AM(behavior)

model.cuda()

criterion = nn.CrossEntropyLoss()
criterion_0 = nn.MSELoss()
criterion_1 = nn.MSELoss()
# Adam 会拿到 AM 的所有参数，但 feature/mlp 已在上面冻结；
# 真正有用的可训练变量是 24 个 compressor selector 标量。
optimizer = torch.optim.Adam(model.parameters(), 0.1, weight_decay=0.2)
# optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.1)

# optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 32, 1.0)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=5, eta_min=0)

brk = False

for ep in range(100):
    model.train()

    loss_meter = util.AverageMeter()

    bar = tqdm(train_loader, desc=f"{ep}", ascii=" 123456789-", ncols=80)
    for i, (input, target) in enumerate(bar):
        input = input.cuda()
        target = target.cuda()
        # breakpoint()

        sum, area = model(input)
        # breakpoint()
        # 269.03-269.04
        # 拟合 all-e1 reference 输出，同时最小化面积。
        # 缩放系数用于让数值很大的 MSE 和数值较小的 area proxy 落在可比范围。
        loss_0 = criterion_0(sum, target)
        loss_1 = criterion_1(area, torch.zeros_like(area))

        # loss = loss_0 * 1e-3
        loss = loss_0 * 1e-3 + loss_1 * 1e-1

        optimizer.zero_grad()
        if brk:
            breakpoint()
        loss.backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()

        loss_meter.update(loss.item(), input.size(0))
        bar.set_postfix_str(f"{loss_meter.avg:>.6f}")

    # 打印每个可见 slot 最终对应的离散 compressor。
    model.print()
