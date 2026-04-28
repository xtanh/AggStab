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

---

## 七、当前瓶颈复盘（基于 v1/v2 结果）

> 本节为 2026-03-10 补充：结合 `results/dpo/` 与 `results/dpo_v2/` 的实际评测产物，总结目前最主要的瓶颈与下一步实验优先级。

### 7.1 现象：`ProAgg mean` 变好，但 `ProAgg max@N` 变差

以 `results/dpo_v2/dpo_eval_report.txt` 为例（评估温度与训练对齐为 T=0.5）：

- `ProAgg mean`：orig=-0.2633 → dpo=-0.1232（+0.1401），60.1% backbones 提升
- `ProAgg max@N`：orig=0.1142 → dpo=-0.0815（-0.1957），仅 7.6% backbones 提升
- `Diversity`：orig=0.5175 → dpo=0.0434（-0.4741）

这说明 DPO 训练把“平均水平”推上去了，但“采样后挑最优”的能力（max@N）显著下降。

### 7.2 根因候选（优先级从高到低）

**根因 A（P0）：训练目标与使用方式不匹配（mean vs. max-of-N）**

实际使用通常是：对同一 backbone 采样 N 条序列，取 `ProAgg max` 作为最终候选（best-of-N）。但 DPO 训练的监督来自离线 preference pairs 的排序约束（winner/loser），它优化的是“对固定候选的相对偏好”，并不直接优化“采样分布的尾部质量（tail）”。

当 policy 分布变得更尖、更集中时：
- `ProAgg mean` 可能上升（因为集中到一类“平均更安全”的序列）
- 但 tail 消失 → `ProAgg max@N` 会明显变差

**根因 B（P0）：分布变尖导致多样性崩溃（mode collapse / near-collapse）**

v2 的 diversity 大幅降低，且 `n_unique` 并非恒为 1，而是出现“很多序列只差少量位点”的现象（Hamming 距离很小）。这会显著降低 best-of-N 的收益：即便 N 不小，等效探索空间也很小。

**根因 C（P1）：`beta` 在当前实现中的含义易被误解**

在本 repo 的 `src/dpo/dpo_train.py` 实现中，`beta` 出现在：

> `loss = -logsigmoid(beta * (log_ratio_w - log_ratio_l))`

因此 **beta 越大，偏好信号的梯度越强**（对 winner/loser 的拉开越激进）。它并不是一个“显式 KL 系数”。

如果把 beta 当成“更保守/更强 KL”而单调增大，可能会更容易把 policy 推向更尖的分布，进一步损害 diversity 与 max@N。

### 7.3 当前采样策略/温度/N/beta 是否合理？

以 `scripts/run_dpo_pipeline.sh` 的默认设置为准：

- 训练对生成：`N=12`，`T=0.5`
- 评估：`N=12`，`T=0.5`（已对齐训练温度）
- 训练：`beta=0.5`，并在训练时对 pairs 做 `score_gap_delta=0.05` 过滤

**结论（现阶段）：用于“快速验证 pipeline 是否能跑通”是合理的，但用于追求 `ProAgg max@N` 并不理想。**

原因与建议如下：

1) **N=12：偏小，且会限制 pair 的信息量与 score gap 上限**
   - N=12 时，每个 backbone 最多 6 个 rank-aligned pairs，且 winner/loser 之间 gap 往往不够大；
   - 若目标是 max@N（best-of-N），通常应把 N 增加到 32 或 64，才能让 tail 有机会出现并产生可学习的偏好差异。

2) **T=0.5：对 MPNN 采样探索是合理的默认值**
   - T 太低（如 0.1）会显著降低 diversity，使评估退化为近似贪心，best-of-N 失效；
   - 但 T 太高会引入明显低可设计性序列（logprob 变差），需要在 `T=0.3~0.8` 之间做 sweep，并结合 `n_unique`/diversity 与 `MPNN logprob` 一起看。

3) **beta=0.5：需要回到“实现语义”重新判断是否合理**
   - 在当前实现里，beta 主要是偏好梯度强度旋钮；
   - 若观察到 diversity 明显下降且 max@N 变差，优先尝试 **降低 beta**（例如 0.01、0.05、0.1）来减弱过度排序拉开带来的分布变尖。

### 7.4 下一步实验优先级（建议从这里开始）

**优先级 1：把 checkpoint 选择指标对齐到 max@N 与 diversity**

仅用离线 `val_loss/val_accuracy` 保存 best checkpoint，可能会系统性偏向“分布更尖、排序更强”的模型。建议在训练中增加轻量的在线评估：
- 每个 epoch 在一小撮 valid backbones 上采样（例如 50 个）
- 记录 `ProAgg max@N`、`ProAgg mean`、`diversity@N`、`MPNN logprob`
- 用 `max@N`（或 `max@N - λ * collapse_penalty`）挑 best ckpt

**优先级 2：beta sweep + N sweep（最直接验证“分布变尖”假说）**

固定其它不变：
- beta: {0.01, 0.05, 0.1, 0.2, 0.5}
- N（评估用）: {12, 32, 64}

观察：`max@N` 是否随 beta 下降回升、是否随 N 增加回升，以及 diversity 的变化。

**优先级 3：online DPO（刷新 pairs，缓解 off-policy）**

每 1-2 个 epoch 用当前 policy 在一部分 backbones 上重新采样与打分，更新 pairs（可只更新一部分以控成本），通常会比长期使用旧 pairs 更稳。
