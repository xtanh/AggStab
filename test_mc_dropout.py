"""
验证MC Dropout不确定性估计
使用已有的proagg_mlp_v1模型
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

from src.models.ProAgg import ProAggMLPV1
from src.config.utils import load_yaml_config
from src.ln.lightning_data import ProAggDataModule


def predict_with_uncertainty(model, batch_dict, n_samples=10):
    """
    MC Dropout: 多次前向传播估计不确定性
    """
    model.train()  # 保持train模式，dropout保持激活

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(batch_dict)
            samples.append(output['score'])

    samples = torch.stack(samples, dim=0)  # (n_samples, B, 1)

    mean_score = samples.mean(dim=0)       # (B, 1)
    variance = samples.var(dim=0)          # (B, 1)

    model.eval()  # 恢复eval模式

    return {
        'score': mean_score,
        'uncertainty': variance,
        'samples': samples
    }


def main():
    # 加载配置
    cfg = load_yaml_config('configs/default.yaml')

    # 加载数据（只用validation set）
    data_module = ProAggDataModule(cfg)
    data_module.setup('fit')
    val_loader = data_module.val_dataloader()

    # 加载模型
    model = ProAggMLPV1(cfg)

    # 加载你之前训练好的v1 checkpoint
    # 请替换为你的实际checkpoint路径
    checkpoint_path = input("请输入v1模型的checkpoint路径: ").strip()

    if not os.path.exists(checkpoint_path):
        print(f"错误: checkpoint不存在: {checkpoint_path}")
        print("提示: 在results/lightning_logs/version_X/checkpoints/目录下查找")
        return

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        # lightning checkpoint
        model_state = {}
        for k, v in state_dict['state_dict'].items():
            # 移除'model.'前缀
            if k.startswith('model.'):
                k = k[6:]
            model_state[k] = v
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    model.to('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    print(f"模型加载到: {device}")

    # 收集结果
    all_scores_mc = []
    all_uncertainties = []
    all_scores_std = []
    all_targets = []

    print("\n开始MC Dropout推理 (n_samples=10)...")
    for batch in tqdm(val_loader):
        # 移动batch到device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 标准推理（eval模式，dropout关闭）
        model.eval()
        with torch.no_grad():
            std_output = model(batch)

        # MC Dropout推理
        mc_output = predict_with_uncertainty(model, batch, n_samples=10)

        all_scores_std.append(std_output['score'].cpu())
        all_scores_mc.append(mc_output['score'].cpu())
        all_uncertainties.append(mc_output['uncertainty'].cpu())
        all_targets.append(batch['target'].cpu())

    # 合并结果
    scores_std = torch.cat(all_scores_std, dim=0).squeeze().numpy()
    scores_mc = torch.cat(all_scores_mc, dim=0).squeeze().numpy()
    uncertainties = torch.cat(all_uncertainties, dim=0).squeeze().numpy()
    targets = torch.cat(all_targets, dim=0).squeeze().numpy()

    # 分析结果
    print("\n" + "="*60)
    print("MC Dropout 验证结果")
    print("="*60)

    print(f"\n不确定性统计:")
    print(f"  Mean:   {uncertainties.mean():.4f}")
    print(f"  Median: {np.median(uncertainties):.4f}")
    print(f"  Max:    {uncertainties.max():.4f}")
    print(f"  Min:    {uncertainties.min():.4f}")

    # 按不确定性分组看误差
    n_bins = 5
    sorted_indices = np.argsort(uncertainties)
    bin_size = len(sorted_indices) // n_bins

    print(f"\n按不确定性分组分析 (分{n_bins}组):")
    print("-" * 60)
    print(f"{'不确定性范围':<20} {'样本数':<10} {'MAE':<10}")
    print("-" * 60)

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)

        bin_indices = sorted_indices[start:end]
        bin_uncertainty = uncertainties[bin_indices]
        bin_mae = np.abs(scores_mc[bin_indices] - targets[bin_indices]).mean()

        print(f"[{bin_uncertainty.min():.4f}, {bin_uncertainty.max():.4f}]"
              f"{'':<6} {len(bin_indices):<10} {bin_mae:<10.4f}")

    # 计算相关系数
    from scipy.stats import spearmanr, pearsonr

    sp_std, _ = spearmanr(scores_std, targets)
    sp_mc, _ = spearmanr(scores_mc, targets)

    print(f"\n相关性对比:")
    print(f"  Standard (eval mode):  Spearman = {sp_std:.4f}")
    print(f"  MC Dropout (mean):     Spearman = {sp_mc:.4f}")

    # 高不确定性样本分析
    high_uncertainty_threshold = np.percentile(uncertainties, 90)
    high_unc_mask = uncertainties > high_uncertainty_threshold

    print(f"\n高不确定性样本分析 (top 10%, threshold={high_uncertainty_threshold:.4f}):")
    print(f"  数量: {high_unc_mask.sum()}")
    print(f"  平均|预测-真实|: {np.abs(scores_mc[high_unc_mask] - targets[high_unc_mask]).mean():.4f}")
    print(f"  平均不确定性: {uncertainties[high_unc_mask].mean():.4f}")

    # 低不确定性样本分析
    low_unc_mask = uncertainties < np.percentile(uncertainties, 10)
    print(f"\n低不确定性样本分析 (bottom 10%):")
    print(f"  平均|预测-真实|: {np.abs(scores_mc[low_unc_mask] - targets[low_unc_mask]).mean():.4f}")
    print(f"  平均不确定性: {uncertainties[low_unc_mask].mean():.4f}")

    print("\n" + "="*60)
    print("结论:")
    print("  - 如果高不确定性样本的误差 > 低不确定性样本的误差")
    print("    说明MC Dropout有效，可以用来筛选可靠预测")
    print("="*60)


if __name__ == "__main__":
    main()
