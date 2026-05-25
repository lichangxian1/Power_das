import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.data import DataLoader

# 从你现有的文件中导入类和函数
from train_proxy import ArithDataset, custom_collate
from proxy_mlp import ArithProxyMLP

def evaluate_model(data_path="dataset/glitch_power_data_16bit.pt", 
                   ckpt_path="dataset/glitch_power_proxy_gnn_B.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")
    
    # 1. 加载模型 Checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 动态获取模型参数，兼容不同的预训练配置
    node_feature_dim = ckpt.get("node_feature_dim", 7)
    hidden_dim = ckpt.get("hidden_dim", 64)
    num_pfp_layers = ckpt.get("num_pfp_layers", 2)
    dropout = ckpt.get("dropout", 0.0) # 推理阶段 dropout 无实际影响，但需对齐网络结构
    power_mean = ckpt.get("power_mean", 0.0)
    power_std = ckpt.get("power_std", 1.0)
    
    # 2. 实例化并加载代理模型
    model = ArithProxyMLP(
        node_feature_dim=node_feature_dim, 
        hidden_dim=hidden_dim,
        num_pfp_layers=num_pfp_layers,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ 模型加载成功: {num_pfp_layers}-PFP, hidden={hidden_dim}")
    
    # 3. 加载数据集 
    # 为了评估，我们提取 20% 的数据作为测试集 (使用固定种子确保可复现)
    full_dataset = ArithDataset(data_path)
    val_size = max(int(0.2 * len(full_dataset)), 20)
    train_size = len(full_dataset) - val_size
    
    _, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=custom_collate)

    all_pred_power = []
    all_true_power = []

    print(f"🚀 开始评估 (评估样本数: {val_size})...")

    # 4. 进行推理
    with torch.no_grad():
        # 注意: 这里的解包对齐 train_proxy.py 的 custom_collate 返回值
        for X, P, mask, power_norm, power_raw in val_loader:
            X, P, mask = X.to(device), P.to(device), mask.to(device)
            
            # 模型现在只返回归一化的功耗 pred
            pred_power_norm = model(X, P, mask)
            
            # 反归一化，恢复到真实的物理量纲
            pred_power_raw = pred_power_norm * power_std + power_mean
            
            all_pred_power.extend(pred_power_raw.cpu().numpy())
            all_true_power.extend(power_raw.numpy())

    all_pred_power = np.array(all_pred_power)
    all_true_power = np.array(all_true_power)

    # 5. 计算评价指标
    # MAPE (平均绝对百分比误差) - 加上 1e-8 防止除以 0
    mape = np.mean(np.abs((all_true_power - all_pred_power) / (np.abs(all_true_power) + 1e-8))) * 100
    # Kendall's Tau 排序相关系数
    tau, pval = stats.kendalltau(all_pred_power, all_true_power)
    if np.isnan(tau):
        tau = 0.0
    # R² 决定系数 (Pearson相关系数的平方)
    pearson_r, _ = stats.pearsonr(all_pred_power, all_true_power)
    r2 = pearson_r ** 2 if not np.isnan(pearson_r) else 0.0

    print("\n" + "="*50)
    print(f"📊 最终评估结果:")
    print(f"   ► Kendall's Tau: {tau:+.4f} (p={pval:.4f})")
    print(f"   ► R² 决定系数  : {r2:.4f}")
    print(f"   ► MAPE 百分比误差: {mape:.2f}%")
    print("="*50)

    # 6. 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(all_true_power, all_pred_power, alpha=0.6, color='dodgerblue', edgecolors='k', s=40)
    
    # 画出 y=x 的完美对角线
    min_val = min(np.min(all_true_power), np.min(all_pred_power))
    max_val = max(np.max(all_true_power), np.max(all_pred_power))
    # 稍微向外扩展一点作图边界
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', lw=2, label="Perfect Prediction (y=x)")
    
    plt.title(f"Glitch Power Proxy Evaluation\nTau={tau:.3f} | R²={r2:.3f} | MAPE={mape:.2f}%", pad=15)
    plt.xlabel("True Glitch Power (from PTPX)")
    plt.ylabel("Predicted Glitch Power (from K-PFP MLP)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片
    save_fig_path = "dataset/power_scatter_eval.png"
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 散点图已成功保存至: {save_fig_path}")

if __name__ == "__main__":
    evaluate_model()