import torch
data = torch.load("dataset/glitch_power_data_16bit.pt")
print(f"数据量: {len(data)}")
print(f"单样本张量维度 - X: {data[0]['X'].shape}, P: {data[0]['P'].shape}")
print(f"标签值 - 功耗: {data[0]['power']} mW, 延迟: {data[0]['delay']} ns")