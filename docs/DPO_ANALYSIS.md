# DPO 实验分析报告

> 更新日期：2026-03-09

## 一、实验概述

使用 DPO（Direct Preference Optimization）微调 ProteinMPNN，以 ProAgg（SaProt + MLP 聚集预测器）作为 reward 信号，目标是让 ProteinMPNN 生成更低聚集倾向的蛋白质序列。

### 数据规模

| 集合 | Backbone 数 | Preference Pairs |
|------|-------------|-----------------|
| Train | 4,744 | 28,848 |
| Validation | 657 | 3,948 |
| Test | 1,350 | — (仅评估) |

### 超参数（v1 基线运行）

| 参数 | 值 |
|------|----|
| num_samples | 12 |
| sampling temperature | 0.5 |
| DPO β | 0.1 |
| lr | 1e-5 |
| batch_size | 32 (梯度累积) |
| epochs | 10 |
| score_gap_delta | 0.0 (不过滤) |
| eval temperature | 0.1 (bug：与训练不匹配) |

---

## 二、v1 基线结果

### 2.1 训练曲线

| Epoch | Train Loss | Train Acc | Train Margin | Val Loss | Val Acc | Val Margin |
|-------|-----------|-----------|-------------|----------|---------|-----------|
| 1 | 0.6617 | 76.5% | 0.69 | 0.6022 | 76.8% | 2.16 |
| 5 | 0.5151 | 78.8% | 5.15 | 0.4981 | 78.3% | 5.97 |
| 10 | 0.4639 | 80.9% | 7.38 | 0.4526 | 80.8% | **8.22** |

训练表面正常：loss 下降、accuracy 上升。但 reward margin 从 0.69 增长到 8.22（12倍），是**过度优化**的信号。

### 2.2 Test 评估结果（1,350 backbones）

| 指标 | Original MPNN | DPO MPNN | 变化 |
|------|--------------|----------|------|
| ProAgg mean | -0.1082 | -0.1115 | **-0.0033 (变差)** |
| ProAgg max | +0.1307 | -0.0367 | **-0.1675 (大幅变差)** |
| MPNN log-prob | -1.0085 | -0.4448 | +0.5637 (提升) |
| Sequence diversity | 0.3079 | 0.0998 | **-0.2082 (崩溃)** |
| Backbones improved (mean) | — | 508/1350 | 37.6% |
| Backbones improved (max) | — | 177/1350 | **13.1%** |

**结论**：DPO 训练指标良好，但 test 上 ProAgg score 反而下降，diversity 崩溃。

---

## 三、问题根因分析

### 问题 1（P0 Critical）：`compute_seq_log_prob` 中的随机噪声

**位置**：`src/dpo/dpo_train.py:100`

```python
# BUG: 每次调用都用新的随机 randn，导致 log-prob 不可复现
randn = torch.randn(chain_M.shape, device=device)
log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
```

ProteinMPNN 用 `randn` 控制自回归解码顺序。DPO loss 计算需要 4 次 `compute_seq_log_prob`（policy_w, policy_l, ref_w, ref_l），每次 randn 不同，导致：

- `log_pi_θ(y_w)` 和 `log_pi_ref(y_w)` 用的是不同解码顺序，噪声不能相消
- DPO loss 每次计算结果不同，梯度方向随机
- 模型实质上在拟合随机噪声，而非真实的偏好信号

**修复**：在 `dpo_loss()` 中生成一次共享的固定 `randn`（zeros），传给全部 4 次调用。

---

### 问题 2（P1）：Reward Hacking / 过度优化

reward margin epoch 1→10：`0.69 → 8.22`（12倍增长）。

DPO 成功学会让 `log π_θ(y_w) - log π_θ(y_l)` 尽可能大，但这与"**采样新序列的 ProAgg 分数更好**"之间没有直接联系。

- DPO 改变了模型对已知 winner/loser 序列的**评分排序**
- 不保证从模型**采样**出的新序列 ProAgg 更优
- MPNN log-prob 提升（-1.008 → -0.445）：模型对自己的输出更自信，但不代表质量更好

**修复**：增大 β（KL 惩罚），从 0.1 提升到 0.5，限制 policy 偏离 reference 的程度。

---

### 问题 3（P1 Bug）：训练/评估 Temperature 不匹配

| 阶段 | Temperature |
|------|-------------|
| 生成训练对（`sample_and_score.py`）| **T=0.5** |
| 评估（`evaluate.py` 默认值）| **T=0.1** |

`run_dpo_pipeline.sh` 的评估步骤没有传 `--temperature` 参数，使用了 evaluate.py 的默认值 T=0.1。

T=0.1 接近贪心解码，分布极度尖锐：
- DPO 后的模型分布已经比原始模型更 peaked
- 在 T=0.1 下，64/12 个采样几乎完全相同
- diversity 0.3079 → 0.0998，实质上只在评估 1 条序列
- ProAgg max ≈ ProAgg mean，失去了多样性采样找好序列的能力

**修复**：评��时使用与训练相同的 T=0.5。

---

### 问题 4（P2）：Preference Pair 信号过弱

```bash
NUM_SAMPLES=12          # 每个 backbone 只采 12 条序列
score_gap_delta=0.0     # 不过滤任何 pair
```

- N=12 → 最多 6 个 rank-aligned pairs/backbone
- delta=0.0 → 包含 winner/loser ProAgg 分差极小的噪声对
- 噪声对让 DPO 学到错误的偏好信号

**修复**：在训练时按 `score_gap_delta` 过滤现有 pairs（不需要重新采样）。

---

### 问题 5（P3 方法局限）：结构 token 固定近似

所有采样序列都使用原始蛋白质的结构 token（来自 data.csv），即 ProAgg 认为所有变体的结构都与原始蛋白完全相同。这是 fixed-backbone design 的标准近似，但降低了 reward 信号的准确性。

---

## 四、修复方案与 v2 配置

### 代码修改

1. **`src/dpo/dpo_train.py`**：
   - `compute_seq_log_prob` 接受 `randn` 参数
   - `dpo_loss` 中生成一次 `torch.zeros` 的固定 randn，传给全部 4 次调用
   - 添加 `--score_gap_delta` 参数，在加载已有 pairs 时过滤
   - β 默认值从 0.1 改为 0.5

2. **`src/dpo/evaluate.py`**：
   - 默认 temperature 从 0.1 改为 0.5

3. **`scripts/run_dpo_pipeline.sh`**：
   - 评估步骤显式传 `--temperature 0.5` 和 `--beta 0.5`

### v2 快速验证运行命令（使用现有 pairs）

```bash
# Step 1: 重新训练（使用现有 pairs，固定代码问题）
python src/dpo/dpo_train.py \
    --pairs_path data/dpo/train_pairs.pt \
    --val_pairs_path data/dpo/val_pairs.pt \
    --output_dir results/dpo_v2/ \
    --device cuda:3 \
    --epochs 10 \
    --lr 1e-5 \
    --beta 0.5 \
    --batch_size 32 \
    --patience 3 \
    --score_gap_delta 0.05

# Step 2: 评估（temperature 与训练对齐）
python src/dpo/evaluate.py \
    --pdb_dir data/dpo/representative_pdbs/test \
    --dpo_mpnn_ckpt results/dpo_v2/mpnn_dpo_best.pt \
    --proagg_ckpt <your_proagg_ckpt> \
    --proagg_config configs/default.yaml \
    --output results/dpo_v2/eval_results.json \
    --num_samples 12 \
    --temperature 0.5 \
    --device cuda:3

# Step 3: 生成分析报告
python scripts/analyze_dpo_eval.py \
    --eval_json results/dpo_v2/eval_results.json \
    --output_dir results/dpo_v2/
```

---

## 五、预期改善

| 修复 | 预期效果 |
|------|---------|
| 固定 randn | DPO 训练信号一致，梯度有意义；reward margin 增长更稳定 |
| β 0.1→0.5 | 限制 reward hacking，margin 不会无限增长 |
| T 0.1→0.5 评估 | diversity 恢复，ProAgg max 能真正反映模型质量 |
| score_gap_delta=0.05 | 过滤噪声 pair，偏好信号更清晰 |

---

## 六、未来改进方向（不在本次快速验证范围内）

1. **Online DPO**：迭代更新 preference pairs（用当前 policy 采样 → 重新打分配对 → 再训练），避免 off-policy 问题
2. **增大 num_samples**：从 12 增加到 32-64，得到更大 score 区间和更可靠的 pairs
3. **温度退火**：训练初期用高 T 保留多样性，后期降温精细化
4. **RLHF / PPO**：直接用 ProAgg 作为在线 reward，而非 DPO 的离线 preference
