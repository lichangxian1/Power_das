# convert_to_sparse.py
import torch
from tqdm import tqdm

data = torch.load("dataset/glitch_power_data_16bit_v2.pt", map_location="cpu")
for item in tqdm(data):
    P = item.pop("P")  # 移除旧的稠密 P
    edge_index = P.nonzero(as_tuple=False).t().contiguous()  # [2, E]
    item["edge_index"] = edge_index

torch.save(data, "dataset/glitch_power_data_16bit_v3.pt")
print("转换完成，新格式: edge_index [2, E]")