# ProAgg 最佳模型记录

## 当前最佳

**模型**: `proagg_mlp_v13`
**测试分数**: test_spearman=0.7494, test_pearson=0.7338
**验证分数**: val_spearman=0.7613
**实验版本**: version_25
**Checkpoint**: `results/lightning_logs/version_25/checkpoints/best_epoch=08_val_spearman=0.7613.ckpt`

## 配置详情

### 模型架构 (proagg_mlp_v13)
- Attention Pooling: 1280→128→1 + Dropout 0.3
- MLP: 1280→1024→256→1 + Dropout 0.7
- LayerNorm + GELU

### 训练配置
- Learning Rate: 1e-4 (head), 5e-5 (LoRA)
- Weight Decay: 0.3 (强L2正则)
- Gradient Clip: 0.5
- Ranking Loss: weight=1.0, margin=0.3
- MSE Loss: weight=1.0
- Batch Size: 32
- LoRA: r=4, alpha=8, dropout=0.2

### 关键改进
1. Ranking Loss (直接优化排序)
2. Attention层Dropout 0.3
3. MLP Dropout 0.7
4. Weight Decay 0.3

## 历史对比

| 版本 | Test Spearman | 关键改进 |
|------|---------------|----------|
| v4   | 0.7430        | Baseline (Attention + 宽网络) |
| v10  | 0.7473        | + Ranking Loss |
| v12  | 0.7490        | Ranking weight=1.0, margin=0.3 |
| v13  | 0.7494        | + 更强正则化 (dropout 0.7, wd 0.3) |

## 备注
- 模型存在轻微过拟合 (train loss仍可下降)
- 0.749可能是当前架构天花板
- 用于DPO reward模型应足够
