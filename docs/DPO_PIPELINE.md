# DPO Pipeline: 用 ProAgg 引导 ProteinMPNN 生成低聚集倾向序列

## 概述

本 pipeline 将训练好的蛋白质聚集倾向预测模型 (ProAgg, 基于 SaProt) 与逆折叠模型 (ProteinMPNN) 结合,
通过 Direct Preference Optimization (DPO) 微调 ProteinMPNN, 使其生成更不容易聚集的蛋白质序列.

采用 rank-aligned pairing 策略 (参考 ProtAlign, ICLR 2026) 构建偏好对,
通过 cluster-based 数据划分保证训练/验证/测试集之间无数据泄露.

```
1. Cluster 代表挑选
   data.csv (13853 proteins, 6754 clusters)
       |
       v
   每个 cluster 随机选 1 个 PDB (seed=42)
       |
       v
   train/ (4744)  valid/ (657)  test/ (1350)

2. DPO Pipeline
   Backbone PDB
       |
       v
   ProteinMPNN 采样 12 条序列 (T=0.5)
       |
       v
   AA 序列 + 原始结构 token -> SA 序列
       |
       v
   SaProt/ProAgg 对每条序列打分
       |
       v
   Rank-aligned pairing: 第 i 名 vs 第 (N/2+i) 名
       |
       v
   DPO Loss 微调 ProteinMPNN (梯度累积, batch_size=32)
       |
       v
   Val 集选 best checkpoint -> Test 集评估
```

---

## 前置条件

- **Conda 环境**: `conda activate SaProt`
- **ProAgg checkpoint** (SaProt + LoRA), 例如:
  `results/lightning_logs/version_14/checkpoints/best_epoch=08_val_spearman=0.7548.ckpt`
- **原始数据**: `data/rocklin/rawdata/data.csv` (含 SA 序列和 cluster 信息)
- **PDB 结构文件**: `data/rocklin/all_AF_rank0_pdbs/`
- **ProteinMPNN 权重**: 自动使用默认权重 `/home/xy_th/ProteinMPNN/vanilla_model_weights/v_48_020.pt`
- **ProAgg 模型选择**：在 `configs/default.yaml` 中通过 `model.model_name` 切换（默认 `proagg_mlp_v1`）

---

## 完整流程

### Step 0: 挑选 Cluster 代表

**脚本**: `scripts/select_cluster_representatives.py`

```bash
python scripts/select_cluster_representatives.py \
    --data_csv data/rocklin/rawdata/data.csv \
    --pdb_dir data/rocklin/all_AF_rank0_pdbs \
    --output_dir data/dpo/representative_pdbs \
    --seed 42
```

从每个 cluster 中随机选 1 个蛋白质, 按 train/valid/test 划分输出到子目录 (PDB 以 symlink 引用).

**输出**:

| 目录 / 文件 | 内容 |
|------------|------|
| `data/dpo/representative_pdbs/train/` | 4744 个训练集代表 PDB |
| `data/dpo/representative_pdbs/valid/` | 657 个验证集代表 PDB |
| `data/dpo/representative_pdbs/test/` | 1350 个测试集代表 PDB |
| `data/dpo/representative_pdbs/representatives.csv` | 完整代表列表 |

---

### Step 1 & 2: 采样 + ProAgg 打分 + 构造偏好对

**脚本**: `src/dpo/sample_and_score.py`

分别在 train 和 valid 集上生成偏好对:

```bash
# Train pairs
python src/dpo/sample_and_score.py \
    --pdb_dir data/dpo/representative_pdbs/train \
    --proagg_ckpt <ProAgg checkpoint> \
    --proagg_config configs/default.yaml \
    --output data/dpo/train_pairs.pt \
    --num_samples 12 \
    --temperature 0.5 \
    --device cuda:3

# Val pairs
python src/dpo/sample_and_score.py \
    --pdb_dir data/dpo/representative_pdbs/valid \
    --proagg_ckpt <ProAgg checkpoint> \
    --proagg_config configs/default.yaml \
    --output data/dpo/val_pairs.pt \
    --num_samples 12 \
    --temperature 0.5 \
    --device cuda:3
```

**参数说明**:

| 参数 | 含义 | 当前值 |
|------|------|--------|
| `--num_samples` | 每个 backbone 采样的序列数 | 12 |
| `--temperature` | Rollout 温度 (高温促进多样性) | 0.5 |
| `--score_gap_delta` | 最小分数差阈值 (过滤噪声 pair) | 0.0 |
| `--data_csv` | 原始数据 CSV (提取结构 token) | `data/rocklin/rawdata/data.csv` |
| `--max_pdbs` | 最多处理的 PDB 数, -1 为全部 | -1 |

**Pair 构建策略** (Rank-aligned pairing):
- 对 N 个序列按 ProAgg 分数降序排列
- 第 i 名与第 (N/2 + i) 名配对 -> 每个 backbone 最多 N/2 对
- 12 个序列 -> 6 对/backbone

**重要提示：采样策略与最终使用方式对齐**

本项目下游常见使用方式是：对同一 backbone 采样 N 条序列，取 `ProAgg max@N` 作为候选（best-of-N）。
因此需要保证：

- `--temperature` 不要过低，否则采样接近贪心，多样性下降，best-of-N 失效。
- `--num_samples` 不要过小，否则会同时限制：
  1) 每个 backbone 的 pairs 数量与可学习的 score gap；
  2) `ProAgg max@N` 的上限（尾部探索不充分）。

若目标是提升 `ProAgg max@N`，建议把 `--num_samples` 提到 32 或 64，并对 `--temperature` 做 0.3~0.8 的 sweep。

**输出**:

| 文件 | 内容 |
|------|------|
| `data/dpo/train_pairs.pt` | 训练偏好对 (~28K 对) |
| `data/dpo/val_pairs.pt` | 验证偏好对 (~3.9K 对) |
| `*_baseline.json` | 每个 PDB 的所有序列和分数 |

---

### Step 3: DPO 微调 ProteinMPNN

**脚本**: `src/dpo/dpo_train.py`

```bash
python src/dpo/dpo_train.py \
    --pairs_path data/dpo/train_pairs.pt \
    --val_pairs_path data/dpo/val_pairs.pt \
    --output_dir results/dpo/ \
    --device cuda:3 \
    --epochs 10 \
    --lr 1e-5 \
    --beta 0.1 \
    --batch_size 32 \
    --patience 3
```

**参数说明**:

| 参数 | 含义 | 当前值 | 参考 (ProtAlign) |
|------|------|--------|-----------------|
| `--lr` | 学习率 | 1e-5 | 5e-6 |
| `--beta` | 偏好信号强度缩放（实现里直接乘在 margin 上） | 0.1 | 0.5 |
| `--batch_size` | 等效 batch size (梯度累积) | 32 | 64 |
| `--patience` | Early stopping patience | 3 | - |
| `--grad_clip` | 梯度裁剪 | 1.0 | - |

**DPO Loss** (内含 KL 散度约束):

$$\mathcal{L} = -\log\sigma\Big(\beta \big[(\log\pi_\theta(y_w|x) - \log\pi_\text{ref}(y_w|x)) - (\log\pi_\theta(y_l|x) - \log\pi_\text{ref}(y_l|x))\big]\Big)$$

β 在本实现中主要是“偏好 margin 的缩放系数”，**越大代表偏好梯度越强**，更容易把 policy 推得更尖、更集中。
如果观察到 diversity 崩溃或 `ProAgg max@N` 下降，优先尝试减小 β（例如 0.01~0.1），并结合在线评估指标选 checkpoint（详见 `docs/DPO_ANALYSIS.md`）。

**输出**:

| 文件 | 内容 |
|------|------|
| `results/dpo/mpnn_dpo_best.pt` | Val 上最优的 checkpoint |
| `results/dpo/mpnn_dpo_epochN.pt` | 定期保存的 checkpoint |
| `results/dpo/training_history.json` | 训练记录 |

**关键指标**:

- `train_loss` / `val_loss`: 应持续下降 (初始 ~0.693)
- `train_accuracy` / `val_accuracy`: 偏好对正确率, 应上升
- `reward_margin`: winner 和 loser 的 log-ratio 差, 应增大

---

### Step 4: 评估对比

**脚本**: `src/dpo/evaluate.py`

```bash
python src/dpo/evaluate.py \
    --pdb_dir data/dpo/representative_pdbs/test \
    --dpo_mpnn_ckpt results/dpo/mpnn_dpo_best.pt \
    --proagg_ckpt <ProAgg checkpoint> \
    --output results/dpo/eval_results.json \
    --device cuda:3
```

**评估指标**:

| 指标 | 含义 | DPO 后期望 |
|------|------|-----------|
| `proagg_mean` | 采样序列的 ProAgg 平均分 | 应提升 (更不容易聚集) |
| `proagg_max` | 最优序列的 ProAgg 分数 | 应提升 |
| `mpnn_logprob_mean` | ProteinMPNN 序列 log-prob | 应相近 (逆折叠能力保持) |
| `diversity` | 采样序列间的 Hamming 距离 | 会略降 (可接受) |

---

## 一键运行

```bash
bash scripts/run_dpo_pipeline.sh \
    results/lightning_logs/version_14/checkpoints/best_epoch=08_val_spearman=0.7548.ckpt \
    cuda:3
```

依次执行 Step 1 -> Step 2 -> Step 3 -> Step 4, 所有结果保存在 `results/dpo/`.

---

## 代码结构

```
scripts/
    select_cluster_representatives.py  # Step 0: Cluster 代表挑选
    run_dpo_pipeline.sh                # 一键运行脚本
src/
    mpnn/
        mpnn_wrapper.py     # ProteinMPNN 封装 (加载/采样/log-prob 计算)
    dpo/
        sample_and_score.py # Step 1-2: 采样 + SaProt 打分 + rank-aligned pairing
        dpo_train.py        # Step 3: DPO 微调 (梯度累积 + val early stopping)
        evaluate.py         # Step 4: 评估对比
configs/
    default.yaml            # ProAgg/SaProt 配置
data/
    dpo/
        representative_pdbs/
            train/          # 训练集代表 PDB (symlink)
            valid/          # 验证集代表 PDB (symlink)
            test/           # 测试集代表 PDB (symlink)
        train_pairs.pt      # 训练偏好对
        val_pairs.pt        # 验证偏好对
```

---

## 后续优化方向

1. **提升 ProAgg 预测准确率**: 更好的 reward signal 直接提升 DPO 效果
2. **调节 β**: 在 val 集上对比不同 β (0.05, 0.1, 0.2, 0.5) 的效果
3. **Score gap 过滤**: 设置 `--score_gap_delta` 过滤噪声 pair
4. **Semi-online 训练**: 迭代 rollout→train, 避免 off-policy 问题
5. **多目标优化**: 同时考虑聚集倾向 + designability (参考 ProtAlign)
6. **降低学习率**: 如训练不稳定, 可降至 5e-6
