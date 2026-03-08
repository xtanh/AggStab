# DPO Pipeline: 用 ProAgg 引导 ProteinMPNN 生成低聚集倾向序列

## 概述

本 pipeline 将训练好的蛋白质聚集倾向预测模型 (ProAgg) 与逆折叠模型 (ProteinMPNN) 结合,
通过 Direct Preference Optimization (DPO) 微调 ProteinMPNN, 使其生成更不容易聚集的蛋白质序列.

```
Backbone PDB
    |
    v
ProteinMPNN 采样 N 条序列
    |
    v
ProAgg 对每条序列打分 (聚集倾向)
    |
    v
高分 vs 低分 --> 构造偏好对 (winner, loser)
    |
    v
DPO Loss 微调 ProteinMPNN
    |
    v
微调后的 ProteinMPNN --> 生成低聚集序列
```

---

## 前置条件

- **Conda 环境**: `conda activate surface`
- **ProAgg 训练好的 checkpoint**, 例如:
  `results/lightning_logs/version_1/checkpoints/best_epoch=13_val_spearman=0.6674.ckpt`
- **PDB backbone 结构文件**, 已有:
  `data/rocklin/all_AF_rank0_pdbs/` (19505 个 AlphaFold 预测结构)
- **ProteinMPNN 权重** (自动使用默认权重 `/home/xy_th/ProteinMPNN/vanilla_model_weights/v_48_020.pt`)

---

## 完整流程

### Step 1: 采样 + ProAgg 打分 + 构造偏好对

**脚本**: `src/dpo/sample_and_score.py`

```bash
python -u src/dpo/sample_and_score.py \
    --pdb_dir data/rocklin/all_AF_rank0_pdbs \
    --proagg_ckpt results/lightning_logs/version_1/checkpoints/best_epoch=13_val_spearman=0.6674.ckpt \
    --output data/dpo_pairs.pt \
    --num_samples 64 \
    --temperature 0.2 \
    --max_pdbs 20 \
    --max_pairs_per_pdb 256 \
    --device cuda:3
```

**参数说明**:

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--pdb_dir` | PDB backbone 文件目录 | `data/rocklin/all_AF_rank0_pdbs` |
| `--proagg_ckpt` | ProAgg 训练好的 checkpoint | 见上 |
| `--num_samples` | 每个 backbone 采样的序列数 | 64 |
| `--temperature` | 采样温度, 越高越多样 | 0.2 (建议 0.1~0.3) |
| `--max_pdbs` | 最多处理多少个 PDB, -1 表示全部 | 快速验证: 20, 正式: 200~500 |
| `--max_pairs_per_pdb` | 每个 backbone 最多保留多少对 | 256 |
| `--top_ratio` | 取 top/bottom 多少比例构造对 | 0.25 |

**输出**:

| 文件 | 内容 |
|------|------|
| `data/dpo_pairs.pt` | 偏好对数据 (PyTorch 格式), 包含 seq_winner, seq_loser, ProAgg scores, PDB 路径 |
| `data/dpo_pairs_baseline.json` | Baseline 结果 (每个 PDB 的所有序列和分数, 可用于分析) |

**运行时间参考**: 20 个 PDB ~ 4 分钟 (GPU)

---

### Step 2: DPO 微调 ProteinMPNN

**脚本**: `src/dpo/dpo_train.py`

```bash
python -u src/dpo/dpo_train.py \
    --pairs_path data/dpo_pairs.pt \
    --output_dir results/dpo/ \
    --device cuda:3 \
    --epochs 10 \
    --lr 1e-5 \
    --beta 0.1 \
    --log_every 50 \
    --save_every 5
```

**参数说明**:

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--pairs_path` | Step 1 输出的偏好对文件 | `data/dpo_pairs.pt` |
| `--epochs` | 训练轮数 | 快速验证: 3, 正式: 10~20 |
| `--lr` | 学习率 | 1e-5 |
| `--beta` | DPO 温度参数, 越大越保守 | 0.1 (建议 0.05~0.5) |
| `--grad_clip` | 梯度裁剪 | 1.0 |
| `--max_pairs` | 使用的最大偏好对数, -1 表示全部 | 快速验证: 200, 正式: -1 |

**输出**:

| 文件 | 内容 |
|------|------|
| `results/dpo/mpnn_dpo_epochN.pt` | 微调后的 ProteinMPNN checkpoint |
| `results/dpo/training_history.json` | 训练记录 (loss, accuracy, reward_margin) |

**关键指标** (训练时观察):

- `loss`: 应持续下降 (初始 ~0.693)
- `accuracy`: 模型在偏好对上的正确率, 应上升到 ~1.0
- `reward_margin`: winner 和 loser 的 log-ratio 差, 应持续增大

**运行时间参考**: 200 对 x 3 epoch ~ 3 分钟 (GPU)

---

### Step 3: 评估对比

**脚本**: `src/dpo/evaluate.py`

```bash
python -u src/dpo/evaluate.py \
    --pdb_dir data/rocklin/all_AF_rank0_pdbs \
    --dpo_mpnn_ckpt results/dpo/mpnn_dpo_epoch10.pt \
    --proagg_ckpt results/lightning_logs/version_1/checkpoints/best_epoch=13_val_spearman=0.6674.ckpt \
    --output results/dpo/eval_results.json \
    --num_samples 64 \
    --temperature 0.2 \
    --max_pdbs 10 \
    --device cuda:3
```

**输出**:

| 文件 | 内容 |
|------|------|
| `results/dpo/eval_results.json` | 对比结果 (原始 vs DPO, 每个 PDB 的详细指标) |

**评估指标**:

| 指标 | 含义 | DPO 后期望 |
|------|------|-----------|
| `proagg_mean` | 采样序列的 ProAgg 平均分 | 应提升 (更不容易聚集) |
| `proagg_max` | 最优序列的 ProAgg 分数 | 应提升 |
| `mpnn_logprob_mean` | ProteinMPNN 自身的序列 log-prob | 会略高 (更集中) |
| `diversity` | 采样序列间的平均 Hamming 距离 | 会略降 (可接受) |

**运行时间参考**: 10 个 PDB ~ 5 分钟 (评估两个模型)

---

## 一键运行

```bash
bash scripts/run_dpo_pipeline.sh \
    data/rocklin/all_AF_rank0_pdbs \
    results/lightning_logs/version_1/checkpoints/best_epoch=13_val_spearman=0.6674.ckpt \
    cuda:3
```

依次执行 Step 1 -> Step 2 -> Step 3, 所有结果保存在 `results/dpo/`.

---

## 验证结果参考

在 20 个 PDB 上用 200 对偏好对训练 3 epoch 后的效果 (10 个 PDB 评估):

| 指标 | 原始 ProteinMPNN | DPO 微调后 | 变化 |
|------|-----------------|-----------|------|
| ProAgg 均分 | -0.32 | +0.23 | **+0.55** |
| ProAgg 最高分 | +0.30 | +0.40 | **+0.10** |
| 序列多样性 | 0.37 | 0.33 | -0.04 |

---

## 代码结构

```
src/
  mpnn/
    mpnn_wrapper.py     # ProteinMPNN 封装 (加载/采样/log-prob 计算)
  dpo/
    sample_and_score.py # Step 1: 采样 + ProAgg 打分 + 构造偏好对
    dpo_train.py        # Step 2: DPO 微调训练
    evaluate.py         # Step 3: 评估对比
configs/
    dpo.yaml            # DPO 配置文件
scripts/
    run_dpo_pipeline.sh # 一键运行脚本
```

---

## 后续优化方向

1. **提升 ProAgg 预测准确率**: 更好的 reward signal 直接提升 DPO 效果
2. **选择有代表性的 PDB 进行微调**: 覆盖更多拓扑结构, 提升泛化性
3. **调节超参数**: temperature, beta, 数据量等
4. **多目标优化**: 同时考虑聚集倾向 + designability (参考 ProtAlign)
