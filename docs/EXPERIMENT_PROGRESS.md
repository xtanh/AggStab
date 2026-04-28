# 实验进度与结论沉淀（持续更新）

> 目的：把每次实验的**命令 / 关键超参 / 产物路径 / 结果摘要 / 结论与下一步**固化在一个地方，避免遗忘与重复踩坑。  
> 约定：之后每次你让我分析实验结果时，我都会同步把结论追加到本文件。

---

## 快速索引

- DPO v2 系统排查：`docs/DPO_V2_REPORT.md`
- DPO 总体分析（含 v1/v2 对比与超参建议）：`docs/DPO_ANALYSIS.md`
- DPO pipeline 文档：`docs/DPO_PIPELINE.md`
- 病态分析脚本（同聚/组成崩溃 + pairs 偏置）：`scripts/analyze_dpo_v2_pathology.py`
- Mini pipeline（小规模快速跑通/复现问题）：`scripts/run_dpo_mini_pipeline.sh`

---

## 当前核心假设（2026-03-10）

1) `results/dpo_v2/` 的主要失败模式是 **composition collapse（低复杂度/同聚/极端带电序列）**，导致 `proagg_max@N` 系统性变差、diversity 崩溃。  
2) 偏好对（pairs）存在轻微但一致的 “charged↑ / hydrophobic↓” 偏置，DPO 在较强偏好优化强度（例如较大 `beta`）且缺少反塌缩约束时会放大该偏置。  
3) `src/dpo/dpo_train.py` 里的 `beta` 在实现语义上是 **偏好梯度强度缩放**，不是显式 KL 系数；过大更可能促发分布变尖/塌缩。

---

## 源数据约束（来自 source paper）

基于 `docs/source_data.pdf` 对应论文《Global Analysis of Aggregation Determinants in Small Protein Domains》方法部分，可确定以下与 DPO 子集选择直接相关的事实：

1) **作者用于机器学习的数据不是“所有库混合”**
   - 论文方法明确写到：机器学习只合并了 **Metagenomic libraries**，**没有并入 Protein Family library**。
   - 原因有两个：
     - library-specific effects 难以校正；
     - Protein Family library 在高温聚集上的动态范围较小，不利于建模。

2) **当前本地 `data/rocklin/rawdata/data.csv` 看起来已经是过滤后的 metagenomic ML 数据**
   - 行数 `13853`
   - cluster 数 `6754`
   - 长度范围 `38-71 aa`，中位数 `68 aa`
   - 列包含：`log2_fold_change_{4,50,75}_clip`、`cluster`、`split`
   - 这与论文方法部分“过滤后用于 ML 的小型 globular domains”一致。

3) **论文在进入 ML 前已经做过两层重要过滤**
   - 移除：未在所有 control replicates 中定量到的蛋白
   - 移除：`radius_of_gyration / sqrt(length) > 1.8` 的蛋白（非紧凑 globular proteins）

4) **训练/验证/测试 split 是按 cluster 做的**
   - 论文使用 `MMseqs2`，最小序列一致性阈值 `0.3`
   - 然后按 cluster 分配 train / valid / test，以测试泛化

**对我们的启示**

- 你现在 DPO 用的数据源本身已经是作者认为“适合泛化建模”的那一部分，不需要再去追求把更多异质库混进来。
- 但这不意味着 DPO 必须吃下全部 metagenomic 数据；相反，DPO 是高风险对齐步骤，更应该做：
  - cluster-balanced 子集选择
  - pair 质量筛选
  - 逐步放量
- 另外，这个数据集本身只覆盖 **38-71 aa 的小蛋白结构域**。即使你用全量训练，泛化边界也主要限于这一类小结构域，而不是所有蛋白。

---

## 设计取舍：允许“适度 diversity 下降”，但禁止“低复杂度/同聚”塌缩

> 备注（2026-03-10）：聚集性与氨基酸亲疏水/带电组成强相关，优化 ProAgg 预期会带来一定的 diversity 下降，这是合理的。
> 但 `results/dpo_v2/` 的失败不是“适度多样性下降”，而是出现大量低复杂度/同聚（单一残基占比极高、长同聚 run）序列形态，应视为需要硬性抑制的病态。

可选策略（按“最小改动→结构性改动”排序）：

1) **产出侧止血（decode-time / post-filter）**
   - 过滤长同聚 run（例如同一 AA 连续 ≥6）
   - 限制 `top_frac`、`charged_frac`、`hydrophobic_frac` 的极端阈值（把“可接受的组成区间”显式化）
   - 在满足约束的候选中再做 best-of-N（`ProAgg max@N`）

2) **训练目标加约束（anti-collapse regularization）**
   - 混入少量 reference/原模型的 CE/SFT 项（把分布拉回可设计性区域）
   - 或加入显式 KL/entropy 类正则（避免分布过尖）
   - `beta` 做 sweep：把偏好强度降到“不触发同聚”的区间

3) **偏好数据侧去偏（pairs / pairing / weighting）**
   - 提高 `num_samples` 与 `score_gap_delta`，并按 `score_gap` 加权（更可靠的偏好信号）
   - 修改 pairing：避免 winner/loser 主要由组成差异决定（例如加入“复杂度约束”再选 winner）
   - online DPO：定期用当前 policy 采样刷新 pairs，降低离线偏置被持续放大

目标是找到 Pareto：`ProAgg max@N` ↑，同时病态比例（同聚 run/极端 top_frac）≈0，`MPNN logprob` 不显著恶化。

---

## 实验记录

### 现象汇总：mini 好，全量塌缩（待验证的规模效应）

当前观测（截至 2026-03-10）：

- mini（80/10/10）多组实验（`beta=0.1,e=5`；`beta=0.5,e=10`；以及 `beta=0.5,e=10` 且评估 `N=64`）均未出现同聚塌缩，且 `ProAgg mean/max` 在 test=10 上一致提升。
- 全量 `results/dpo_v2/` 出现明显的 composition collapse（`has_run(>=6)≈98%`，`top_frac≈0.94`，`charged_frac≈0.96`），并伴随 `ProAgg max@N` 系统性退化。

可能解释（需要后续实验逐一排除）：

1) **训练总步数/有效更新量差异**：全量 pairs 数量大、epoch 多，优化更容易进入“偏好放大→分布变尖→塌缩”的动力学区域；mini 可能只是没训练到那个阶段。
2) **选 ckpt 指标偏置**：全量用离线 `val_loss/acc` 选 best，可能偏向 margin 更大但更尖的模型；mini 方差大且训练短，不易触发该偏置。
3) **数据覆盖与偏好去偏不足**：全量数据覆盖更广，pairs 的轻微组成偏置在更长训练中被持续放大；mini 覆盖小，偏置放大不明显。

建议的验证路径：

- 做一组“中等规模”实验（例如 train=400/1000, val=50, test=50），保持其余超参不变，观察塌缩是否随规模/步数出现。
- 在训练过程中定期做在线采样评估，记录 `ProAgg max@N`、`diversity`、病态比例（同聚 run/top_frac/charged_frac），并用它们参与 early stopping/选 ckpt。

**放量到全量（full）建议的准入条件（建议先满足再跑）**

1) 在 tr400（或 tr1000）规模上，找到一个 beta/配置使得：
   - `has_run(>=6)` 接近 0（或低于可接受阈值，例如 <1%）
   - `ProAgg max@N` 至少不系统性退化（`improve_max` 稳定 >50% 或更高）
   - `MPNN logprob` 不出现灾难性退化（例如避免像 beta=0.1 那种 -1.95 的大幅下降）
2) 把 “病态比例 + max@N” 纳入 early stopping/选 ckpt（否则全量训练更容易跑偏且浪费算力）
3) 先做一次 full 的短跑（例如 1-2 epoch）验证指标走向，再决定是否继续长训

### 2026-03-10：DPO mini（beta=0.1）用于快速跑通与 sanity check

**命令**

```bash
MAX_TRAIN_PDBS=80 MAX_VALID_PDBS=10 MAX_TEST_PDBS=10 \
DPO_BETA=0.1 DPO_EPOCHS=5 \
bash scripts/run_dpo_mini_pipeline.sh \
  results/lightning_logs/version_14/checkpoints/best_epoch=08_val_spearman=0.7548.ckpt \
  cuda:3
```

**产物**

- 输出目录：`results/dpo_mini/`
- 关键文件：
  - `results/dpo_mini/dpo_eval_report.txt`
  - `results/dpo_mini/dpo_eval_summary.csv`
  - `results/dpo_mini/training_history.json`
  - `results/dpo_mini/eval_results.json`

**结果摘要（test=10 backbones）**（来自 `results/dpo_mini/dpo_eval_report.txt`）

- ProAgg mean：orig=-0.4191 → dpo=0.0502（delta=+0.4694）
- ProAgg max：orig=0.0119 → dpo=0.1930（delta=+0.1811）
- MPNN logprob：orig=-1.2829 → dpo=-1.1808（delta=+0.1022）
- Diversity：orig=0.5096 → dpo=0.4806（delta=-0.0291）
- improved(mean)=10/10，improved(max)=10/10

**病态检查**（`python scripts/analyze_dpo_v2_pathology.py ...`）

- `has_run(>=6)`: 0%
- `top_frac_dpo` 均值约 0.193（正常）
- `charged_frac_dpo` 均值约 0.408（正常）
- 与 v2 的同聚/极端带电塌缩形态不同

**结论**

- 该 mini 配置（`beta=0.1`、`epochs=5`）属于“温和微调”，没有复现 v2 的塌缩；在小样本 test 上显示全面提升，适合作为 pipeline/sanity 的快速验证，但不能据此推断大规模泛化。

**下一步**

- 用 mini 做最小对照复现：把 `beta` 提到 v2 水平（例如 0.5），并把 `epochs` 提到 10；必要时把评估 `NUM_SAMPLES` 提到 64，以便更敏感地暴露 `max@N` 的 tail 是否消失。

---

### 2026-03-10：DPO mini 对照（beta=0.5, epochs=10）

**命令**

```bash
MAX_TRAIN_PDBS=80 MAX_VALID_PDBS=10 MAX_TEST_PDBS=10 \
DPO_BETA=0.5 DPO_EPOCHS=10 NUM_SAMPLES=12 TEMPERATURE=0.5 \
bash scripts/run_dpo_mini_pipeline.sh \
  results/lightning_logs/version_14/checkpoints/best_epoch=08_val_spearman=0.7548.ckpt \
  cuda:3
```

**产物**

- 输出目录：`results/dpo_mini_beta0.5_e10_n12_t0.5_tr80_va10_te10/`
- 关键文件：
  - `results/dpo_mini_beta0.5_e10_n12_t0.5_tr80_va10_te10/dpo_eval_report.txt`
  - `results/dpo_mini_beta0.5_e10_n12_t0.5_tr80_va10_te10/dpo_v2_pathology_enriched.csv`
  - `results/dpo_mini_beta0.5_e10_n12_t0.5_tr80_va10_te10/training_history.json`

**结果摘要（test=10 backbones）**（来自 report）

- ProAgg mean：orig=-0.4192 → dpo=0.1664（delta=+0.5855）
- ProAgg max：orig=0.0426 → dpo=0.2183（delta=+0.1757）
- MPNN logprob：orig=-1.2857 → dpo=-1.0739（delta=+0.2118）
- Diversity：orig=0.5100 → dpo=0.4404（delta=-0.0696）
- improved(mean)=10/10，improved(max)=10/10

**病态检查（composition collapse）**

`python scripts/analyze_dpo_v2_pathology.py ...` 输出：

- `has_run(>=6)`: 0%
- `top_frac_dpo` mean≈0.234（p50≈0.231）
- `charged_frac_dpo` mean≈0.486（p50≈0.484）
- `hydrophobic_frac_dpo` mean≈0.269（p50≈0.268）

该 mini 对照仍未复现 `results/dpo_v2/` 中的“极端同聚/带电塌缩”（v2 全量统计 `has_run(>=6)≈98%`）。

**结论**

- 在 mini 子集（80/10/10）上，把 `beta` 提到 0.5 并训练 10 epoch，仍然表现为“温和、可控”的微调：`ProAgg mean/max` 提升且无同聚病态，但 diversity 有更明显下降（-0.07）。
- 这提示：`results/dpo_v2/` 的 collapse 可能还需要其它条件共同触发（例如更大规模/更长训练、不同 pairs 质量或筛选策略、以及 best-of-N 评估的 N 更大导致 tail 现象更明显）。

**下一步**

- 在 mini 上把 `NUM_SAMPLES` 提到 64 评估（保持训练不变），验证 `max@N` 在更大 N 下是否仍稳定提升或开始退化。
- 若要更贴近 v2：确保训练时的 `score_gap_delta`、pairs 生成/过滤策略与 v2 完全一致，并在更大规模上验证（或做 online DPO/加入反塌缩约束）。

---

### 2026-03-10：DPO mini（beta=0.5, epochs=10）在更大 best-of-N 评估下（N=64）

> 说明：此实验的训练超参与上一条一致，主要变化是评估阶段 `NUM_SAMPLES=64`，用于更敏感地检查 tail 与 `max@N` 行为。

**产物**

- 输出目录：`results/dpo_mini_beta0.5_e10_n64_t0.5_tr80_va10_te10/`

**结果摘要（test=10 backbones）**（来自 report）

- ProAgg mean：orig=-0.3848 → dpo=0.1415（delta=+0.5262）
- ProAgg max：orig=0.1048 → dpo=0.2437（delta=+0.1389）
- MPNN logprob：orig=-1.2838 → dpo=-1.3914（delta=-0.1075）
- Diversity：orig=0.5078 → dpo=0.4694（delta=-0.0384）
- improved(mean)=10/10，improved(max)=10/10

**解读**

- 在 N=64 的 best-of-N 下，`ProAgg max` 仍然一致提升，说明在 mini 规模下没有出现 v2 那种“tail 消失导致 max@N 退化”的现象。
- 但 `MPNN logprob` 出现下降（更负），提示在更大 N、更广的采样覆盖下，DPO 模型的可设计性代理指标可能存在 trade-off（需要和 `ProAgg max@N` 一起做 Pareto 观察）。
- 病态检查仍为正常范围（`has_run(>=6)=0%`），未复现同聚塌缩。

**补充：mini pairs 的偏置**

对本次 mini pairs（train 2560 对 / val 320 对）做 AA 偏置统计，winner 仍表现为 “charged↑/hydrophobic↓”，且 top diff 仍以 K/E 上升、R/A/L 下降为主，模式与全量 pairs 一致。这说明：

- pairs 偏置本身不足以在 mini 设置下触发 collapse；
- v2 的 collapse 更可能与规模/训练动力学/约束缺失共同作用有关。

**下一步**

- 扩大训练规模（例如 train=400/1000）或训练更久（更大 epoch 或更激进超参）以尝试复现 collapse；
- 或者在全量设置下验证：降低 `beta` / 加反塌缩约束是否能抑制同聚病态并恢复 `max@N`。

---

### 2026-03-10：中等规模桥接实验（tr400/va50/te50, beta=0.5, N=64）

**产物**

- 输出目录：`results/dpo_mini_beta0.5_e10_n64_t0.5_tr400_va50_te50/`

**结果摘要（test=50 backbones）**（来自 `dpo_eval_report.txt`）

- ProAgg mean：orig=-0.3288 → dpo=-0.0489（delta=+0.2800），improve=40/50（80%）
- ProAgg max：orig=0.1704 → dpo=0.1908（delta=+0.0203），improve=32/50（64%）
- MPNN logprob：orig=-1.2818 → dpo=-0.8655（delta=+0.4163，显著更“自信/可设计性 proxy 更好”）
- Diversity：orig=0.5147 → dpo=0.2795（delta=-0.2352，显著下降）

**病态检查（关键）**（来自 `analyze_dpo_v2_pathology.py`）

- `has_run(>=6)`: **26%**（开始出现同聚/低复杂度序列）
- `top_frac_dpo`: mean=0.3653，p90=0.7981，max=0.8730（尾部出现极端值）
- `charged_frac_dpo`: mean=0.5573，p90=0.8832，max=0.9286
- 与 `delta_max` 强相关：
  - corr(delta_max, top_frac)=-0.6175
  - corr(delta_max, charged_frac)=-0.6620
  - corr(delta_max, hydrophobic_frac)=+0.6272
  - corr(delta_max, diversity)=+0.5080

**结论**

- “规模效应”开始显现：相比 tr80 的实验（同配置下 `has_run(>=6)=0%`），tr400 已出现 26% 病态同聚序列，且这些病态与 `ProAgg max` 变差强相关。
- 同时出现了一个典型 trade-off：`MPNN logprob` 大幅改善，但 diversity 大幅下降；`ProAgg max` 只小幅提升且已有 36% backbones 变差（`delta_max<0`）。
- 这更接近 `results/dpo_v2/` 的失败机制（只是程度较轻），支持“更长训练/更大数据会进入 collapse 区域”的假设。

**下一步**

1) 在该规模上做 `beta` 下调 sweep（优先）：`beta ∈ {0.05, 0.1, 0.2, 0.5}`，观察 `has_run(>=6)` 与 `delta_max` 的折中点。
2) 把病态比例纳入选 ckpt/early stop：当 `has_run` 或 `top_frac_p90` 超过阈值时停止或回滚 checkpoint。
3) 若仍难以抑制：加入反塌缩约束（decode-time 过滤 / 混入 CE / 显式 KL/entropy / 多目标偏好）。

---

### 2026-03-11：中等规模桥接实验（tr400/va50/te50, beta=0.1, N=64）

**产物**

- 输出目录：`results/dpo_mini_beta0.1_e10_n64_t0.5_tr400_va50_te50/`

**结果摘要（test=50 backbones）**（来自 `dpo_eval_report.txt`）

- ProAgg mean：orig=-0.3195 → dpo=0.0385（delta=+0.3580），improve=50/50（100%）
- ProAgg max：orig=0.1809 → dpo=0.2327（delta=+0.0518），improve=38/50（76%）
- MPNN logprob：orig=-1.2799 → dpo=-3.2313（delta=-1.9515，显著变差）
- Diversity：orig=0.5121 → dpo=0.5840（delta=+0.0719，上升）

**病态检查（composition collapse）**

`analyze_dpo_v2_pathology.py` 输出：

- `has_run(>=6)`: 0%
- `top_frac_dpo`：mean=0.2406，p90=0.2947，max=0.3824（正常）
- `charged_frac_dpo`：mean=0.4458，p90=0.5535，max=0.6029（正常）
- `n_unique`：original/dpo 均为 64（采样多样性未塌缩）

**与 beta=0.5（同规模同 N）的对比结论**

- beta 从 0.5 → 0.1 后，**病态同聚比例从 26% → 0%**，diversity 从 0.2795 → 0.5840（显著恢复/上升），说明更小的偏好强度能有效抑制 collapse。
- 代价是 `MPNN logprob` 大幅恶化（更负），提示模型在“ProAgg 优化”与“MPNN 一致性/可设计性 proxy”之间出现明显 trade-off。
- `ProAgg max` 的平均提升幅度反而更大（+0.0518 vs +0.0203），且提升比例更高（76% vs 64%），但由于 logprob 变差，需要后续确认这种提升是否对应更合理的序列（而不是对 MPNN 分布外的序列进行探索）。

**下一步**

1) 在 tr400 上补一个中间值 `beta=0.2`（以及可选 `beta=0.05`），绘制三点曲线：`has_run`、`delta_max`、`logprob_delta`，找 Pareto 点。
2) 若目标要求同时保持 logprob：考虑加入显式锚定（混入少量 CE/SFT 或显式 KL/entropy 正则），而不是单纯调 beta。

---

### 2026-03-12：beta sweep 总结（tr400/va50/te50, N=64, seed=42）

**产物**

- 汇总表：`results/dpo_sweep_tr400_va50_te50_n64_t0.5_e10_seed42_beta_sweep_summary.csv`

**sweep 结果总览**

| beta | delta_mean | delta_max | improve_max | delta_logprob | delta_diversity | has_run_rate |
|------|------------|-----------|-------------|---------------|-----------------|--------------|
| 0.05 | +0.3468 | +0.0424 | 74% | -2.6656 | +0.0604 | 0% |
| 0.10 | +0.3719 | +0.0476 | 78% | -2.0528 | +0.0891 | 0% |
| 0.20 | **+0.3787** | **+0.0529** | 82% | -1.1644 | +0.1237 | 0% |
| 0.50 | +0.2976 | +0.0420 | **84%** | **+0.3485** | -0.1975 | **14%** |

**关键结论**

1) **`beta=0.5` 是塌缩开始出现的拐点**
   - `has_run_rate=14%`
   - `top_frac_p90=0.70`，`charged_frac_p90=0.86`，`hydrophobic_frac_p10=0.10`
   - 虽然 `delta_logprob` 转正（更接近 MPNN 分布），但代价是病态序列开始出现，且 `delta_mean/delta_max` 反而不如 0.2。

2) **`beta=0.2` 是当前 tr400 规模下最好的 Pareto 点**
   - `has_run_rate=0%`
   - `delta_mean` 和 `delta_max` 都是 sweep 中最好
   - `improve_max=82%`
   - `delta_logprob=-1.16`，虽然仍有退化，但明显好于 0.05 / 0.1

3) **`beta=0.05/0.1` 太松，病态没有，但 logprob 退化过重**
   - 没有塌缩，diversity 也更高
   - 但 `delta_logprob` 分别到 -2.67 / -2.05，说明序列越来越偏离 MPNN 认可的设计分布

4) **离线 `val_reward_margin` 越大，不代表越好**
   - 0.05/0.1 的 `last_val_reward_margin` 反而最高（7.63 / 7.24），但 logprob 更差；
   - 0.5 的 `last_val_reward_margin` 最低（4.77），却最接近塌缩。
   - 这再次说明不能继续用 margin 作为主要判断依据。

**当前决策**

- 若继续往更大规模推进，优先使用 **`beta=0.2`** 作为基线。
- `beta=0.5` 不建议直接放大到 full；它已经在 tr400 上出现病态征兆，full 上大概率继续放大。

**何时上 full**

当前建议：

1) 先用 `beta=0.2` 做一次更大桥接规模（例如 tr1000/va100/te100）验证：
   - `has_run_rate` 仍接近 0
   - `delta_max` 不明显回落
   - `delta_logprob` 不进一步恶化
2) 若上述成立，再做 full 的短跑（1-2 epoch）观察趋势；
3) 若在 tr1000 就开始出现病态，则不要直接上 full，应先加入反塌缩约束或改变选 ckpt 逻辑。

---

### 2026-03-13：更大桥接规模验证（tr1000/va100/te100, beta=0.2, N=64）

**产物**

- 输出目录：`results/dpo_mini_beta0.2_e10_n64_t0.5_tr1000_va100_te100/`

**结果摘要（test=100 backbones）**

- ProAgg mean：orig=-0.3091 → dpo=-0.0903（delta=+0.2187），improve=73/100
- ProAgg max：orig=0.1683 → dpo=0.0732（delta=-0.0950），improve=26/100
- MPNN logprob：orig=-1.2790 → dpo=-0.7220（delta=+0.5569）
- Diversity：orig=0.5110 → dpo=0.1282（delta=-0.3828）

**病态检查（关键）**

- `has_run(>=6)`: **61%**
- `top_frac_dpo`: mean=0.6086，p50=0.7683，p90=0.9560，max=0.9857
- `charged_frac_dpo`: mean=0.7039，p50=0.8624，p90=0.9706，max=1.0000
- `hydrophobic_frac_dpo`: mean=0.1659，p50=0.0870

与 `delta_max` 的相关性非常强：

- corr(delta_max, top_frac) = **-0.8034**
- corr(delta_max, charged_frac) = **-0.7987**
- corr(delta_max, hydrophobic_frac) = **+0.8017**
- corr(delta_max, diversity) = **+0.6500**

**训练信号**

- `val_reward_margin` 从 3.20 持续增长到 **9.48**
- `val_accuracy` 最终到 **0.9075**

这再次说明：离线 DPO 训练指标可以持续变好，但采样分布已经明显进入 collapse 区域。

**结论**

1) `tr400` 上的 Pareto 点（`beta=0.2`）**不能直接外推到 `tr1000`**。
2) 一旦规模扩大到 `tr1000/va100/te100`，即使 `beta=0.2`，也已经出现严重病态：
   - `has_run=61%`
   - `ProAgg max@N` 平均值转负（`delta_max=-0.0950`）
   - diversity 大幅崩溃
3) 当前条件下 **不应该直接上 full 训练**；否则大概率继续放大到接近 `results/dpo_v2/` 的全量失败模式。

**当前最合理的下一步**

不是继续放量，而是先加护栏后再放量：

1) **选 ckpt 逻辑改造**
   - 不再仅用离线 `val_loss/val_accuracy`
   - 至少引入在线评估的 `ProAgg max@N` + 病态比例（`has_run/top_frac_p90`）

2) **反塌缩约束**
   - decode/post-filter：过滤长同聚 run、极端 `top_frac` / `charged_frac`
   - training regularization：混入少量 CE/SFT 或显式 KL/entropy，把模型锚回 MPNN 分布

3) **再做 beta sweep 时按规模重新找 Pareto**
   - 不能假设小规模最优 beta 在更大规模仍最优

---

### 2026-03-13：分层 DPO 子集构造工具

**新增脚本**

- `scripts/select_dpo_subset.py`

**目的**

在已有的 cluster-balanced representative backbone 基础上，再做一层更适合 DPO 的子集选择：

- 保持 split 不变（train/valid/test）
- 继承 “每个 cluster 一个 representative”
- 按 `sequence length × log2_fold_change_75_clip` 分层抽样

这样做不是为了减少数据量本身，而是为了：

- 保证结构/长度/标签动态范围覆盖
- 降低 DPO 被某些高频模式主导的风险
- 为后续逐步放量提供可解释的中间规模数据集

**默认分层规则**

- 长度 bins：`[0, 50), [50, 60), [60, 72)`
- 标签 bins：按 `log2_fold_change_75_clip` 做 split 内 quantile bin（默认 5 桶）
- 抽样策略：
  - 每个非空 stratum 尽量至少保留 1 个样本
  - 剩余名额按 stratum 大小比例分配

**示例命令**

```bash
python scripts/select_dpo_subset.py \
  --representatives_csv data/dpo/representative_pdbs/representatives.csv \
  --pdb_root data/dpo/representative_pdbs \
  --output_dir data/dpo/subsets/stratified_tr500_va100_te100 \
  --train_size 500 \
  --valid_size 100 \
  --test_size 100 \
  --seed 42
```

**如何接到现有 pipeline**

三个 pipeline 脚本现在都支持用环境变量覆盖 PDB 目录：

- `PDB_TRAIN`
- `PDB_VALID`
- `PDB_TEST`

例如：

```bash
PDB_TRAIN=data/dpo/subsets/stratified_tr500_va100_te100/train \
PDB_VALID=data/dpo/subsets/stratified_tr500_va100_te100/valid \
PDB_TEST=data/dpo/subsets/stratified_tr500_va100_te100/test \
DPO_BETA=0.2 NUM_SAMPLES=64 SEED=42 \
bash scripts/run_dpo_mini_pipeline.sh \
  results/lightning_logs/version_14/checkpoints/best_epoch=08_val_spearman=0.7548.ckpt \
  cuda:3
```

**当前建议**

下一步不要直接继续 random subset 放量，而是优先在这个“分层子集”上重新做一轮 `beta` 对照，观察是否比随机抽样更稳定。

---

### 2026-03-13：分层子集验证（beta=0.2, N=64）

> 输出目录已规范命名为 `results/dpo_stratified_tr500_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50/`，对应使用 `data/dpo/subsets/stratified_tr500_va100_te100/` 这一分层子集。

**结果摘要（test=50 backbones）**

- ProAgg mean：orig=-0.2588 → dpo=0.1309（delta=+0.3897），improve=50/50
- ProAgg max：orig=0.1862 → dpo=0.2732（delta=+0.0869），improve=47/50（94%）
- MPNN logprob：orig=-1.2821 → dpo=-2.3987（delta=-1.1166）
- Diversity：orig=0.5126 → dpo=0.5979（delta=+0.0853）

**病态检查**

- `has_run(>=6)`: 0%
- `top_frac_p90`: 0.3103
- `charged_frac_p90`: 0.5420
- `hydrophobic_frac_p90`: 0.4204

结论：没有出现同聚/极端带电塌缩。

**与“随机 tr400, beta=0.2”对比**

随机 tr400（之前）：

- `delta_max=+0.0529`
- `improve_max=82%`
- `has_run=0%`
- `delta_logprob≈-1.16`
- `delta_diversity≈+0.12`

分层子集（本次）：

- `delta_max=+0.0869`
- `improve_max=94%`
- `has_run=0%`
- `delta_logprob≈-1.12`
- `delta_diversity≈+0.09`

**结论**

1) 分层子集优于随机同规模子集：
   - `ProAgg max@N` 提升更明显
   - 提升覆盖率更高（94% vs 82%）
   - 没有额外引入病态
2) 这说明：对于当前 DPO 任务，**“结构覆盖优先 + 标签分布辅助平衡”的子集选择策略是有效的**。
3) 与 `tr1000 beta=0.2` 的强烈塌缩相比，这进一步支持：
   - 问题不只是 beta，而是“数据规模 + 数据组成 + 优化动力学”的共同作用
   - 一个设计得当的中等规模子集，当前比盲目放量更适合作为 DPO 训练集

**当前推荐**

- 继续以“分层子集 + beta=0.2”作为当前最稳基线。
- 下一步若要继续放量，不建议直接回到 random/full，而是：
  1. 先把分层子集规模扩大到 `tr800/va100/te100`
  2. 继续观察 `has_run`、`delta_max`、`delta_logprob`
  3. 只有在这些指标仍稳定时，再考虑 full

---

### 2026-03-10：Mini pipeline 汇总阶段参数 bug（已修复）

**现象**

Mini pipeline 的 Step 5/5 调用 `scripts/analyze_dpo_eval.py` 参数错误：

```
analyze_dpo_eval.py: error: unrecognized arguments: --eval_json --output_dir ...
```

**原因**

`scripts/analyze_dpo_eval.py` 的接口是：

- 位置参数：`eval_json`
- 可选参数：`--output <dir>`

**修复**

- `scripts/run_dpo_mini_pipeline.sh` 改为：
  - `python scripts/analyze_dpo_eval.py "$DPO_OUTPUT/eval_results.json" --output "$DPO_OUTPUT"`

---

### 2026-03-10：Mini 输出目录覆盖风险（已修复）

**问题**

在 sweep `beta/epochs/N/T` 时，若固定写 `results/dpo_mini/`，会互相覆盖，难以对照。

**修复**

`scripts/run_dpo_mini_pipeline.sh` 增加 `RUN_TAG` 并默认按超参自动生成：

- 输出目录：`results/dpo_${RUN_TAG}`
- pairs：`data/dpo/${RUN_TAG}_train_pairs.pt` / `data/dpo/${RUN_TAG}_val_pairs.pt`

可通过显式设置 `RUN_TAG` 与 `DPO_OUTPUT` 自定义更短名字。

---

### 2026-03-17：分层子集继续放量到 `tr800/va100/te100`，`beta=0.2` 仍稳定

**运行配置**

- 结果目录：`results/dpo_stratified_tr800_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50`
- 训练子集：`data/dpo/subsets/stratified_tr800_va100_te100/{train,valid,test}`
- 超参数：`beta=0.2`, `epochs=10`, `num_samples=64`, `temperature=0.5`, `seed=42`

**评估结果**（`dpo_eval_report.txt`）

- `ProAgg mean: -0.2588 -> 0.0975`，`delta=+0.3563`
- `ProAgg max:  0.1862 -> 0.2693`，`delta=+0.0831`
- `MPNN logprob: -1.2821 -> -2.1454`，`delta=-0.8634`
- `Diversity: 0.5126 -> 0.6172`，`delta=+0.1046`
- `improve_mean = 49/50 (98%)`
- `improve_max  = 45/50 (90%)`

**病态检查**（`dpo_v2_pathology_enriched.csv`）

- `has_run(>=6) = 0%`
- `top_frac_p90 = 0.3126`
- `charged_frac_p90 = 0.5665`
- `hydrophobic_frac_p90 = 0.4104`
- 与 `delta_max` 的相关性都很弱：
  - `corr(delta_max, top_frac) = -0.0662`
  - `corr(delta_max, charged_frac) = +0.0073`
  - `corr(delta_max, hydrophobic_frac) = +0.1826`
  - `corr(delta_max, diversity) = -0.0422`

**训练动力学**（`training_history.json`）

- `val_accuracy` 最终到 `0.8881`
- `val_reward_margin` 最终到 `6.47`
- 这个 margin 已经不低，但没有伴随病态塌缩，说明在分层子集上，较大的 margin 目前仍可接受。

**与分层 `tr500` 基线对比**

分层 `tr500, beta=0.2`（之前）：

- `delta_max = +0.0869`
- `improve_max = 94%`
- `delta_logprob ≈ -1.12`
- `has_run = 0%`

分层 `tr800, beta=0.2`（本次）：

- `delta_max = +0.0831`
- `improve_max = 90%`
- `delta_logprob ≈ -0.86`
- `has_run = 0%`

**结论**

1. 分层子集从 `tr500` 扩到 `tr800` 后，仍然没有出现 random `tr1000` 那种 collapse。
2. `ProAgg max@N` 依旧稳定提升，虽然相比 `tr500` 略有回落，但差距很小。
3. `delta_logprob` 反而变得更温和（`-1.12 -> -0.86`），这说明在当前分层策略下，继续放量并没有把模型进一步推离 MPNN 分布。
4. 当前最稳的结论是：**分层子集显著优于随机放量；`beta=0.2` 在分层子集上仍然是可用配置。**

**下一步建议**

- 优先做 `stratified tr1000/va100/te100, beta=0.2, N=64`，确认分层策略能否继续跨过 random `tr1000` 的塌缩阈值。
- 若 `tr1000` 仍稳定，再考虑 full 的短跑（先 `1-2 epoch`，不直接长训）。
- 若 `tr1000` 开始出现病态，则不要继续放量，应转向在线选 ckpt / decode 过滤 / 训练侧锚定正则。

---

### 2026-03-17：当前 `stratified` 规则下的“最大有效子集”统计

**统计对象**

- 数据源：`data/dpo/representative_pdbs/representatives.csv`
- 当前规则：`length_bin × log2_fold_change_75_clip q-bin`
- length bins：`[0,50), [50,60), [60,72)`
- label bins：每个 split 内 `qcut(..., q=5)`

**train split 的实际 strata 分布**

- train 总数：`4744`
- 当前规则下共有 `10` 个 stratum，但其中：
  - 5 个主 stratum：`len_60_71__q1..q5`，样本数分别为 `949, 948, 945, 948, 944`
  - 5 个稀有 stratum 合计只有 `10` 个样本（`1, 3, 1, 3, 2`）
- 也就是说，**99.79% 的 train backbone 都落在 `len_60_71 × q1..q5` 这 5 个主 stratum 里。**

**这意味着什么**

1. 当前 `stratified` 的主导因素其实不是长度，而是 `log2_fold_change_75_clip` 的 5 个分位桶。
2. 长度分层在当前数据上只对极少数短序列起作用；对绝大多数 backbone，当前策略近似等价于：
   - `train split` 内按 `log2_fold_change_75_clip` 五等分，再均衡抽样。
3. 因此，“最大有效 stratified 子集”主要由这 5 个主 stratum 决定，而不是由那几个 singleton/小 bin 决定。

**train split 下，不同目标子集大小对应的主 stratum 覆盖率**

- `target=800`：每个主 stratum 约取 `16.8%`
- `target=1000`：每个主 stratum 约取 `21.0%`
- `target=1500`：每个主 stratum 约取 `31.5%`
- `target=2000`：每个主 stratum 约取 `42.1%`
- `target=2400`：每个主 stratum 约取 `50.5%`
- `target=3000`：每个主 stratum 约取 `63.2%`
- `target=3600`：每个主 stratum 约取 `75.9%`
- `target=4200`：每个主 stratum 约取 `88.5%`

**结论**

- 如果把“stratified 仍然有实质选择空间”定义为：每个主 stratum 至少还保留约 `35%-50%` 的未抽样空间，那么：
  - `tr2400 ~ tr3000` 是当前规则下比较合理的上限区间。
- 到 `tr3600` 时，5 个主 stratum 都已经被取走约 `76%`，此时虽然技术上还叫 stratified，但“子集选择”的意义已经明显变弱。
- 到 `tr4200+` 时，基本接近 full train，当前这套 `stratified subset` 的价值就很有限了；如果还想保留 stratified 的思想，应改成 **训练时的分层 sampler / weighting**，而不是继续做静态子集抽样。

**实用建议**

- 当前阶段可以把 `tr1000`、`tr1500`、`tr2000` 看作仍然“强意义 stratified”的区间。
- 如果 `tr1000` 和 `tr1500` 都稳定，再尝试 `tr2000`。
- 不建议把当前这套静态子集策略一路推到 `tr3600+` 再称其为“真正的 stratified 训练”；那时更适合切换到 full + stratified sampler 的方案。

---

### 2026-03-17：新增 `Metagenomic_dG.csv` 后的判断：优先做“约束/多任务”，不建议直接做简单加权多目标

**数据概况**

- 文件：`data/rocklin/Metagenomic_dG.csv`
- 规模：`19455` 条、`27` 列
- 关键字段：`aa_seq`, `log10_K50_t`, `log10_K50_c`, `deltaG_t`, `deltaG_c`, `deltaG`
- 这是论文中提到的 cDNA display proteolysis 稳定性数据，用于刻画全局折叠稳定性。

**与当前 aggregation 数据的重叠**

- 与 `data/rocklin/rawdata/data.csv` 按 `name` 可对上 `13810` 条
- 覆盖率接近当前 aggregation 训练集主体，因此非常适合做辅助监督或约束

**与 aggregation 标签的关系**

在重叠的 `13810` 条上：

- `corr(log2_fold_change_75_clip, deltaG) = -0.2245`
- `corr(log2_fold_change_75_clip, deltaG_t) = -0.2006`
- `corr(log2_fold_change_75_clip, deltaG_c) = -0.2240`
- `corr(log2_fold_change_50_clip, deltaG) = -0.0934`
- `corr(log2_fold_change_4_clip, deltaG) = +0.0195`

**结论**

1. `ΔG` 与 aggregation 不是同一个目标，且在 `75 °C` 条件下甚至表现出轻度冲突（负相关）。
2. 因此，不建议直接把当前问题改成“简单线性加权的多目标最大化”。
3. 更合理的用法是：把 `ΔG` 当成 **稳定性约束/辅助任务**，而不是直接与 ProAgg 做等权求和。

**建议的使用顺序**

- 第一优先级：做 **多任务监督**
  - 用同一个 backbone encoder，同时预测 aggregation（现有任务）和 `ΔG`
  - 目的：让表征学习到“低聚集 ≠ 低复杂度作弊序列”的边界
- 第二优先级：做 **候选过滤/约束**
  - 在 DPO 生成 preference pairs 或最终 best-of-N 选择时，先过滤低 `ΔG` 候选，再按 ProAgg 排序
  - 比简单加权更稳，也更容易解释
- 第三优先级：若要进入 DPO，再尝试 **约束型偏好分数**
  - 例如只在 `ΔG >= 阈值` 的候选中构建 winner/loser
  - 或用 `score = ProAgg - λ * max(0, ΔG_min - ΔG_pred)` 这种 penalty 形式

**当前不建议**

- 不建议直接把 DPO reward 改成 `ProAgg + λ * ΔG`
- 因为 `ΔG` 与 `75 °C` aggregation 并不一致，简单加权很可能破坏当前已经有效的 aggregation 优化方向

**推荐下一步**

1. 先做一个基础统计/建模验证：`ΔG` 预测是否容易学、与当前 ProAgg 共享表征是否有益
2. 若有效，优先把 ProAgg 改成 `aggregation + ΔG` 的多任务模型
3. 等多任务模型稳定后，再考虑把 `ΔG` 引入 DPO 的候选过滤或偏好构造

---

### 2026-03-17：基于 `version_92` 的双目标 DPO 判断

**当前状态**

- `results/lightning_logs/version_92/hparams.yaml` 显示当前回归模型已经是双头：
  - 主头：aggregation `score`
  - 辅头：`deltaG`
- 使用的模型：`proagg_mlp_v36`
- 训练损失包含：`MSE + ranking + 0.1 * deltaG_loss`
- 最优 checkpoint：`results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- test 指标：`test_spearman=0.7561`, `test_pearson=0.7462`

**当前 DPO 流程仍然是单目标**

1. `sample_and_score.py`
   - 用 ProteinMPNN 采样序列
   - 只读取 ProAgg 的 `score` 头打分
   - 按 `score` 构建 winner/loser pairs
2. `dpo_train.py`
   - 只优化单一 preference margin
3. `evaluate.py`
   - 只汇报 `ProAgg mean/max`、`MPNN logprob`、`diversity`

也就是说：即使回归模型已经能预测 `deltaG`，当前 DPO 代码还没有利用它。

**已有 DPO 结果的整体结论**

- full/random 放量：`results/dpo`, `results/dpo_v2`, `results/dpo_mini_beta0.2_e10_n64_t0.5_tr1000_va100_te100`
  - 会出现同聚/极端带电塌缩
  - `ProAgg mean` 往往上升，但 `ProAgg max@N` 会退化
- random 中小规模：可用，但稳定区有限
- stratified 子集：目前最稳
  - `results/dpo_stratified_tr500_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50`
  - `results/dpo_stratified_tr800_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50`
  - 都能保持 `has_run=0%` 且 `delta_max > 0`

**双目标 DPO 的建议路线**

不建议直接做：

- `reward = score + λ * deltaG`

更建议先做 **约束型 DPO**：

1. `sample_and_score.py` 同时读出 `score` 和 `deltaG`
2. 对候选先做 `deltaG` 约束过滤，再按 `score` 排序
3. 只在满足 `deltaG >= threshold` 的候选中构建 preference pairs
4. `evaluate.py` 额外汇报：
   - `deltaG_mean/max`
   - 满足 `deltaG >= threshold` 的比例

如果约束过滤有效，再考虑第二阶段：

- `score_eff = score - λ * max(0, deltaG_min - deltaG_pred)`

也就是只惩罚稳定性不达标的序列，而不是把 `deltaG` 与 aggregation 等权求和。

**原因**

- `deltaG` 与 `75 °C` aggregation 在数据上是轻度冲突的
- 简单线性加权很可能破坏当前已经有效的 aggregation 优化方向
- 约束型 DPO 更符合你的实际目标：
  - 先不要生成明显不稳定的序列
  - 在这个前提下优化 aggregation

---

### 2026-03-17：参考 ProtAlign 论文后，对“双目标 ProteinMPNN 微调”的实施建议

**参考资料**

- `docs/PROPERTY-DRIVEN PROTEIN INVERSE FOLDING WITH MULTI-OBJECTIVE PREFERENCE ALIGNMENT.pdf`
- 其核心思路不是把多个性质直接加权成一个标量 reward，而是：
  1. 为每个性质分别构建 preference dataset
  2. 训练时在不同性质之间均衡采样
  3. 用自适应 margin 缓解目标冲突
  4. 用 semi-online 迭代刷新 rollout

**你的当前条件**

- aggregation predictor：`results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
  - `test_spearman = 0.7561`
- `deltaG` predictor：`results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt`
  - `test_spearman = 0.7001`
- 这两个 predictor 彼此独立，当前 DPO 流程尚未支持多目标训练。

**推荐实现：多数据集 + 约束/自适应 margin 的多目标 DPO**

不建议：

- 直接把两个 predictor 的分数做简单加权：`score = w1 * agg + w2 * deltaG`
- 原因：aggregation 与 `deltaG` 在你的数据上并非一致目标，简单 scalarization 容易重新引入 collapse 或损失 aggregation 提升

建议：

1. **为每个目标单独构建 pair 数据集**
   - `D_agg`：按 `version_92` 的 aggregation 分数排序构建 pairs
   - `D_dg`：按 `version_96` 的 `deltaG` 分数排序构建 pairs
   - 仍使用 rank-aligned pairing + property-specific `score_gap_delta`

2. **训练时在两个目标之间均衡采样**
   - 每个 step 随机选择目标 `k in {agg, dg}`
   - 再从对应 `D_k` 里采一个 pair 做 DPO 更新
   - 这一步对齐了 ProtAlign 的“sample entries evenly across properties”思想

3. **加入自适应冲突 margin，而不是直接分数加权**
   - 若当前训练 pair 来自 `agg`，但 winner 在 `deltaG` 上明显更差，则降低该 pair 的有效 margin
   - 实用形式可写成：
     - `effective_term = beta * (log_ratio_w - log_ratio_l) - lambda_conflict * max(0, deltaG_l - deltaG_w)`
   - 对 `deltaG` pair 同理，用 aggregation 差异作为 conflict penalty
   - 这比论文里的完整公式更易落地，也更符合你当前代码风格

4. **第一阶段先做 semi-offline，不必立刻 full semi-online**
   - 先用当前 policy 采样一次，同时打两个 predictor 的分
   - 构造 `D_agg` / `D_dg`
   - 训练多目标 DPO
   - 先验证是否比单目标更稳
   - 只有这一步有效，再做 semi-online 迭代刷新 pairs

**建议的实现顺序**

- Phase 1：最小可行版本
  - 改 `sample_and_score.py`：同时支持两个 predictor，输出两个 pair 集合
  - 改 `dpo_train.py`：支持 `pairs_path_agg` / `pairs_path_dg`，训练时 1:1 采样
  - 暂不加 adaptive margin，只看“分开 pairs + 均衡采样”是否已经优于单目标

- Phase 2：加入 conflict-aware margin
  - 在每个 pair 上记录 auxiliary predictor 的分数差
  - 在 DPO loss 中减去 conflict penalty

- Phase 3：semi-online
  - 每若干 epoch 用当前 policy 重新 rollout
  - 刷新 `D_agg` / `D_dg`
  - 保留一部分旧数据作为 replay buffer，避免训练震荡

**评估必须同步升级**

如果做双目标 DPO，`evaluate.py` 不能再只看 aggregation：

- aggregation: `mean/max`
- `deltaG`: `mean/max`
- designability: `mpnn_logprob`
- diversity
- pathology: `has_run`, `top_frac_p90`, `charged_frac_p90`
- 最好再加一个 Pareto/constraint 视角：
  - `agg_max` under `deltaG >= threshold`
  - `deltaG_max` under `agg >= threshold`

**当前最稳的工程结论**

- 若现在就要开做，最合理的起点不是“完全照论文实现 full semi-online MoDPO”，而是：
  - `version_92` + `version_96`
  - 双 predictor 打分
  - 双 pair dataset
  - 训练时均衡采样
  - 先不做 scalarized reward
- 这样改动最小，也最容易判断到底是“多目标本身有效”，还是“又引入了新的训练不稳定性”。

---

### 2026-03-18：用 `version_92` 重新打分并重建 pairs 后的 `stratified tr800, beta=0.2` DPO 结果

**说明**

- 结果目录：`results/dpo_stratified_tr800_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50`
- 本次 run 已确认使用：
  - `proagg_ckpt = results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
  - `proagg_config = configs/proagg_final_candidate.yaml`
- 这意味着该目录当前内容已经被新的 `version_92` run 覆盖，不能再与之前旧 annotator 的同名目录混淆。

**评估结果**（`dpo_eval_report.txt`）

- `ProAgg mean: -0.1878 -> 0.1441`，`delta=+0.3319`
- `ProAgg max:  0.4117 -> 0.4742`，`delta=+0.0625`
- `MPNN logprob: -1.2821 -> -2.5268`，`delta=-1.2447`
- `Diversity: 0.5126 -> 0.6298`，`delta=+0.1173`
- `improve_mean = 43/50 (86%)`
- `improve_max  = 34/50 (68%)`

**病态检查**（`dpo_v2_pathology_enriched.csv`）

- `has_run(>=6) = 0%`
- `top_frac_p90 = 0.2805`
- `charged_frac_p90 = 0.4943`
- `hydrophobic_frac_p90 = 0.4572`
- 与 `delta_max` 的相关性都很弱：
  - `corr(delta_max, top_frac) = -0.1236`
  - `corr(delta_max, charged_frac) = -0.0840`
  - `corr(delta_max, hydrophobic_frac) = +0.0575`
  - `corr(delta_max, diversity) = +0.0299`

**训练动力学**（`training_history.json`）

- `val_accuracy` 最终到 `0.8813`
- `val_reward_margin` 最终到 `6.68`
- 训练过程平稳，没有出现 collapse 的迹象

**与旧 annotator 的 stratified tr800 结果对比**

旧 annotator（之前）：

- `delta_mean = +0.3563`
- `delta_max = +0.0831`
- `improve_max = 90%`
- `delta_logprob = -0.8634`
- `has_run = 0%`

`version_92` annotator（本次）：

- `delta_mean = +0.3319`
- `delta_max = +0.0625`
- `improve_max = 68%`
- `delta_logprob = -1.2447`
- `has_run = 0%`

**结论**

1. 用更强的 aggregation predictor（`version_92`）重新打分并重建 pairs 后，DPO 仍然稳定，没有重新引入病态塌缩。
2. 但从 `delta_max`、`improve_max`、`delta_logprob` 看，**这次并没有优于旧 annotator**；相反，它在当前 `stratified tr800 + beta=0.2` 设置下更保守，且对 MPNN 分布偏离更大。
3. 因此，`version_92` 更强的回归性能，并不自动意味着“作为 DPO annotator 更好”。
4. 下一步如果继续沿 `version_92` 方向推进，不应直接默认它优于旧 annotator；更合理的是在小规模上额外扫一圈 `beta / score_gap_delta`，看它是否只是需要不同的 preference 强度。

---

### 2026-03-18：对 `stratified tr800, beta=0.2, N=64` 的单例 AF3 结构验证

**操作**

- 从 `results/dpo_stratified_tr800_va100_te100_beta0.2_e10_n64_t0.5_tr400_va50_te50/eval_results.json` 中挑选了一个 `best_seq`
- 使用 AlphaFold 3 预测该序列结构
- 与原始 backbone 对齐后，得到 `RMSD = 0.604 Å`

**解释**

- 这是一个很强的正面信号：至少对该单个候选而言，DPO 生成的高分序列仍然与目标 backbone 高度一致。
- 这说明当前 pipeline 并非只是“通过低复杂度或统计偏置骗过 predictor”，至少有样本同时满足：
  - aggregation predictor 高分
  - 结构预测后与目标 backbone 高一致性

**但这条证据的边界也要明确**

- 这只是单例验证，不能直接外推到整体分布。
- 它可以证明“存在真实可设计候选”，但不能证明“整体 `best_seq` 都有类似结构质量”。
- 若要把这条结果用于更强结论，应进一步在多个 backbone 上重复，至少统计：
  - AF3/ESMFold/RoseTTAFold 预测结构的 RMSD 或 TM-score
  - 与 `ProAgg max`、`MPNN logprob`、`deltaG` 的关系

**当前判断**

- 这条验证支持当前 `stratified tr800 + beta=0.2` 路线具有实际可设计性潜力。
- 如果后续要从“DPO 指标分析”走向“真实设计可行性”，结构预测复核应成为下一阶段关键验证项。

---

### 2026-03-18：若拿到最终更优 DPO 模型，建议的下游分析框架

参考 `docs/PROPERTY-DRIVEN PROTEIN INVERSE FOLDING WITH MULTI-OBJECTIVE PREFERENCE ALIGNMENT.pdf`，对当前项目最有价值的下游分析应分为 4 层：

1. **生成层指标（最基础）**
   - `ProAgg mean/max`
   - `MPNN logprob`
   - `diversity`, `n_unique`
   - `has_run`, `top_frac`, `charged_frac`, `hydrophobic_frac`
   - 目的：判断模型是否真的改善 aggregation，同时没有 collapse

2. **结构一致性层（最关键的外部验证）**
   - 对每个 backbone 的 `best_seq`（或 top-k）做结构预测
   - 统计 backbone-aligned `RMSD`、`TM-score`、结构置信度（如 pLDDT/pTM）
   - 至少在一个固定子集上批量做，而不是只看单例
   - 目的：确认高 aggregation 分的序列仍然兼容目标 backbone

3. **候选质量层（面向真实设计）**
   - 若有稳定性 predictor，则同时统计 `deltaG`
   - 计算约束下的指标：
     - `agg_max` under `deltaG >= threshold`
     - `agg_max` under `RMSD <= threshold`
   - 输出 Pareto / frontier，而不是只看单一 score

4. **机制解释层（面向论文）**
   - 氨基酸组成变化（全局、表面/核心若可得）
   - 与病态指标、结构指标、designability 指标的相关性
   - 与 baseline / old annotator / no-DPO 的直接对比
   - 目的：解释“模型为什么变好/为什么失败”

**推荐的分析顺序**

- Step A：先跑固定 benchmark（如 stratified test set）上的标准 eval JSON
- Step B：从每个 backbone 取 `best_seq` 和 `top-k` 候选，批量做结构预测复核
- Step C：做约束分析（满足结构/稳定性阈值后，aggregation 还能剩多少提升）
- Step D：做机制图和表（组成、病态、相关性、ablation）

**建议形成的最终结果包**

- 主表 1：DPO vs baseline 的 `mean/max/logprob/diversity`
- 主表 2：结构复核结果（RMSD/TM-score/pLDDT）
- 主表 3：约束后候选质量（例如 `agg_max` under structure-pass）
- 图 1：`delta_max` 分布
- 图 2：病态指标 vs `delta_max` 相关性
- 图 3：结构指标 vs `ProAgg` / `MPNN logprob` 的关系
- 图 4：氨基酸组成变化或表面/核心组成分析

---

### 2026-03-18：两篇外部 amyloid 数据论文可如何服务当前项目

**文献**

- `docs/sciadv_ben_lehner.pdf`
  - Aβ42 大规模突变-成核自由能景观（`>140,000` 变体）
  - 产出的是 `ΔΔG‡` / nucleation energetic landscape，重点是疾病相关 amyloid nucleation
- `docs/science_advance_aggregation.pdf`
  - 大规模随机短序列 aggregation 数据
  - 提出了 `CANYA`，强调 aggregation motif/grammar 学习与跨数据集泛化

**总体判断**

- 这两篇数据 **不适合直接并入当前主训练集**：
  - 你的主任务是小型 globular domain 的 stress-induced aggregation / inverse folding
  - 这两篇主要是 amyloid / random-sequence aggregation，分布和机制都不同
- 但它们 **非常适合做外部 case study / 机制分析 / 额外泛化验证**。

**最合理的定位**

1. 不作为主训练数据
2. 作为外部验证集，回答：
   - 你的 predictor / DPO 是否学到一般性的 aggregation-averse 偏好？
   - 这些偏好是否与已知 amyloid motif/energetics 一致？
3. 作为 Discussion/Case Study 材料，而不是主 benchmark

**推荐的 3 个 case study 方向**

**Case Study 1：Aβ42 单点突变外部验证（最强机制型 case）**

目标：验证你的 aggregation predictor 是否能在完全外部分布上，至少恢复一部分 Aβ42 mutation effect 的方向性。

可做法：
- 从 `sciadv_ben_lehner.pdf` 对应数据中提取：
  - 单点突变 `Aβ42 WT -> mutant`
  - 对应 `ΔΔG‡` 或 nucleation effect sign
- 用你的 aggregation predictor 给 WT 和 mutant 打分
- 比较：
  - `Δ predictor score` 与 `ΔΔG‡` 的相关性
  - sign accuracy（是否预测出“更易/更难聚集”的方向）
- 若相关性存在，说明你的模型虽然训练在 globular domain 上，但学到了一部分更普适的 aggregation signal

边界：
- 不必期待很高相关，因为 Aβ42 是高度特化的 amyloid system
- 这更适合作为“external mechanistic sanity check”

**Case Study 2：amyloid motif grammar 对照（最适合论文图）**

目标：把你的 DPO 生成序列与 `CANYA` 论文中总结的 aggregation motif grammar 做定性/半定量对照。

可做法：
- 从 `science_advance_aggregation.pdf` 提炼 motif 规律：
  - hydrophobic blocks ↑ aggregation
  - charged / Pro-rich motifs ↓ aggregation
  - context-dependent aromatic / polar motifs
- 对比：
  - DPO 后序列 vs baseline 序列的 3-mer / 4-mer motif 频率变化
  - 是否减少已知 amyloid-promoting motifs
  - 是否增加 anti-aggregation motifs（charged/proline-disruptive motifs）
- 最终做成 motif enrichment 图或 heatmap

价值：
- 非常适合写 Discussion / mechanism figure
- 解释“DPO 究竟学到了什么局部序列规律”

**Case Study 3：外部筛选任务模拟（最贴近应用）**

目标：模拟“如果把你的 predictor 当成外部 aggregation filter，它能否在 random/amyloid-heavy sequence 空间里区分危险候选？”

可做法：
- 用 `science_advance_aggregation.pdf` 的外部随机序列集合/已标注 aggregator vs non-aggregator
- 直接测试你的 predictor：
  - AUROC / AUPR
  - 与 CANYA / CamSol / Aggrescan 等方法做对照（若数据可取）
- 这不是为了证明你超过这些模型，而是说明你的 predictor 在 out-of-domain aggregation detection 上不是完全失效

**建议优先级**

- 若想做 strongest case：优先 `Case 1 + Case 2`
- 若想做 practical appendix：再补 `Case 3`

**和 DPO 的关联方式**

最自然的写法不是“我用这些数据训练了 DPO”，而是：

- 主结果：DPO 在 Rocklin domain aggregation task 上有效
- 外部 case study：
  - predictor 与 Aβ42 nucleation effect 有一定一致性
  - DPO 生成序列减少了已知 amyloid-promoting motifs
  - 结构保持的同时，在外部 aggregation grammar 上也更合理

**一句话结论**

- 这两篇数据 **更适合做 external validation / mechanism case study**，
- **不建议直接并入主训练集**。

---

### 2026-03-18：`version_92` 在 `stratified tr800, beta=0.2` 下，`N=16` 明显优于 `N=64`

**运行配置**

- 结果目录：`results/dpo_v92_stratified_tr800_beta0.2_n16`
- `proagg_ckpt = results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- `num_samples = 16`, `temperature = 0.5`
- 训练/验证/测试子集：`stratified_tr800_va100_te100`

**评估结果**（`dpo_eval_report.txt`）

- `ProAgg mean: -0.1940 -> 0.2352`，`delta=+0.4292`
- `ProAgg max:  0.3202 -> 0.4428`，`delta=+0.1226`
- `MPNN logprob: -1.2764 -> -3.1681`，`delta=-1.8917`
- `Diversity: 0.5093 -> 0.5514`，`delta=+0.0420`
- `improve_mean = 45/50 (90%)`
- `improve_max  = 39/50 (78%)`

**病态检查**

- `has_run(>=6) = 0%`
- `top_frac_p90 = 0.2962`
- `charged_frac_p90 = 0.4933`
- 无明显病态塌缩

**与 `version_92, N=64` 直接对比**

`version_92, N=64`：

- `delta_mean = +0.3319`
- `delta_max = +0.0625`
- `improve_max = 68%`
- `delta_logprob = -1.2447`
- `delta_diversity = +0.1173`

`version_92, N=16`（本次）：

- `delta_mean = +0.4292`
- `delta_max = +0.1226`
- `improve_max = 78%`
- `delta_logprob = -1.8917`
- `delta_diversity = +0.0420`

**结论**

1. 对 `version_92` 而言，`N=64` 的 preference 构造过于激进；降到 `N=16` 后，aggregation 指标显著变好。
2. 这支持此前的怀疑：对更强、更敏感的 annotator，较大的 rollout `N` 会把偏好信号拉得过硬，反而不利于 DPO 对齐。
3. 但 `N=16` 的代价也很明确：`MPNN logprob` 下降更多（`-1.89`），说明模型更偏离原始 MPNN 分布。
4. 所以对 `version_92` 来说，当前最合理的判断是：
   - `N=16` 比 `N=64` 更适合做 preference 构造
   - 但还需要继续看 `N=16/32` 与 `beta` 的组合，寻找更好的 `delta_max` / `logprob` 折中点

**和旧 annotator 的位置关系**

- `version_92, N=16` 已经在 aggregation 指标上超过旧 annotator的 `stratified tr800` 结果（尤其 `delta_max`）
- 但它的 `delta_logprob` 代价更大
- 这意味着 `version_92` 不是“不能用”，而是它需要更温和的 rollout / preference 构造策略

---

### 2026-03-18：`version_92` 的 `N/beta` 四卡搜索当前仅有两组完整结果，且都是 mini 规模

**当前已完整落盘**

- `results/dpo_v92_stratified_tr800_n16_b01_gap005`
- `results/dpo_v92_stratified_tr800_n16_b02_gap005`

**尚未完整落盘**

- `results/dpo_v92_stratified_tr800_n32_b02_gap005`：只有训练 ckpt，尚无 eval 结果
- `n32_b01`：目录未见

**重要说明**

- 这两组已完成实验的 `dpo_eval_report.txt` 都显示 `N (test backbones): 10`
- 因此它们是 mini 规模 smoke test（`80/10/10`），而不是 full `tr800/va100/te100` 评估
- 只能用于快速比较超参趋势，不能直接与此前 50-test / 100-test 的结果做强结论对比

**当前两组对比（mini 规模）**

`N=16, beta=0.1, gap=0.05`：
- `delta_mean = +0.5722`
- `delta_max = +0.1149`
- `delta_logprob = +0.3063`
- `delta_diversity = -0.0776`

`N=16, beta=0.2, gap=0.05`：
- `delta_mean = +0.5732`
- `delta_max = +0.1143`
- `delta_logprob = +0.3048`
- `delta_diversity = -0.0770`

**临时结论**

1. 在这个 mini 规模下，`beta=0.1` 与 `beta=0.2` 几乎没有差别。
2. 当前还看不出明确的 beta 优势，说明在小规模 smoke test 上，这两个值都处在可工作的区间。
3. 下一步最重要的不是继续解读这两组，而是补齐：
   - `N=32, beta=0.1`
   - `N=32, beta=0.2`
   并且最好统一在更大的 test 集上评估一次。

---

### 2026-03-18：规模扩展 + 保守超参对照（三组已完成）的结果

**已完成并可直接比较的三组**

1. `results/dpo_v92_stratified_tr800_full_n16_b02_gap005`
   - 实际规模：`train/valid/test = 800/100/100`
   - `N=16`, `beta=0.2`, `gap=0.05`
2. `results/dpo_v92_stratified_tr800_tr400_n16_b01_gap005`
   - 实际规模：`400/50/50`
   - `N=16`, `beta=0.1`, `gap=0.05`
3. `results/dpo_v92_stratified_tr800_tr400_n16_b02_gap010`
   - 实际规模：`400/50/50`
   - `N=16`, `beta=0.2`, `gap=0.10`

另有：
- `results/dpo_v92_stratified_tr800_n32_b02_gap005`
  - 实际规模：`400/50/50`
  - `N=32`, `beta=0.2`, `gap=0.05`
  - 也已完成，可作为对照

**关键结果汇总**

- `tr800 full, N=16, beta=0.2, gap=0.05`
  - `delta_mean = +0.3324`
  - `delta_max = +0.0780`
  - `improve_max = 72/100 (72%)`
  - `delta_logprob = -2.7210`
  - `delta_diversity = +0.1046`
  - `has_run = 0%`

- `tr400, N=16, beta=0.1, gap=0.05`
  - `delta_mean = +0.3310`
  - `delta_max = +0.0710`
  - `improve_max = 33/50 (66%)`
  - `delta_logprob = -2.3961`
  - `delta_diversity = +0.0514`
  - `has_run = 0%`

- `tr400, N=16, beta=0.2, gap=0.10`
  - `delta_mean = +0.4483`
  - `delta_max = +0.1371`
  - `improve_max = 44/50 (88%)`
  - `delta_logprob = -1.6205`
  - `delta_diversity = +0.0285`
  - `has_run = 2%`

- `tr400, N=32, beta=0.2, gap=0.05`
  - `delta_mean = +0.3393`
  - `delta_max = +0.0570`
  - `improve_max = 31/50 (62%)`
  - `delta_logprob = -2.8329`
  - `delta_diversity = +0.1153`
  - `has_run = 2%`

**核心结论**

1. 对 `version_92` 而言，`N=16` 继续优于 `N=32`。
   - `N=32, beta=0.2, gap=0.05` 明显更差：`delta_max` 更低、`logprob` 更差、还出现轻微病态。
   - 这进一步确认了：`version_92` 需要更小、更温和的 rollout 候选池。

2. 提高 `score_gap_delta` 到 `0.10` 是当前最有效的改动。
   - 在 `tr400` 上，`beta=0.2, gap=0.10` 明显优于 `beta=0.1, gap=0.05`：
     - `delta_max: +0.1371 > +0.0710`
     - `improve_max: 88% > 66%`
     - `delta_logprob: -1.6205 > -2.3961`
   - 说明此前 `gap=0.05` 对 `version_92` 确实偏低，留下了太多弱偏好 pair。

3. 规模扩展到 `tr800 full` 仍然保持了优势，但性能有所回落。
   - `tr800 full, N=16, beta=0.2, gap=0.05` 仍然：
     - `delta_max > 0`
     - `has_run = 0%`
     - `improve_max = 72%`
   - 说明这条路线能放量，但当前配置在更大规模上仍有进一步优化空间。

**当前最合理的判断**

- 对 `version_92`：
  - `N=16` 是正确方向
  - `gap=0.10` 比 `gap=0.05` 更适合
- 下一步应优先验证：
  - `tr800 full, N=16, beta=0.2, gap=0.10`
- 这是当前最有希望同时满足“放量 + 保持 tr400 优势”的配置。

## 2026-03-18 — version_92 放量 + 保守超参对照（已完成 3/4 组）

已完成目录：
- `results/dpo_v92_stratified_tr800_full_n16_b02_gap005`
- `results/dpo_v92_stratified_tr800_tr400_n16_b01_gap005`
- `results/dpo_v92_stratified_tr800_tr400_n16_b02_gap010`

未完成：
- `results/dpo_v92_stratified_tr1200_full_n16_b02_gap005`（目前只有 checkpoint，无评估产物）

### 结果摘要

1. **放量到完整 `stratified tr800/va100/te100` 仍然稳定**
   - `delta_mean=+0.3324`
   - `delta_max=+0.0780`
   - `improve_max=72/100`
   - `delta_logprob=-2.7210`
   - `has_run=0%`
   - 结论：放量后没有 collapse，但相较 `tr400`，`max@N` 收益有所回落，且偏离 MPNN 分布更明显。

2. **`tr400, N=16, beta=0.1, gap=0.05` 不如当前 best**
   - `delta_mean=+0.3310`
   - `delta_max=+0.0710`
   - `improve_max=33/50`
   - `delta_logprob=-2.3961`
   - `has_run=0%`
   - 结论：单独把 `beta` 下调到 `0.1` 并没有带来更好的折中点。

3. **`tr400, N=16, beta=0.2, gap=0.10` 是当前最优配置**
   - `delta_mean=+0.4483`
   - `delta_max=+0.1371`
   - `improve_max=44/50`
   - `delta_logprob=-1.6205`
   - `has_run=2%`
   - 结论：把训练阶段 `score_gap_delta` 从 `0.05` 提到 `0.10` 显著改善了 version_92 的 DPO 行为；`max@N`、覆盖率、logprob 都更优，只引入了极轻微病态。

### 当前判断
- 对 `version_92`，关键旋钮已经比较清楚：
  - `NUM_SAMPLES=16` 优于更大的 rollout 候选池。
  - `score_gap_delta=0.10` 明显优于 `0.05`。
  - `beta=0.2` 仍优于更保守的 `0.1`。
- 当前最值得继续推进的下一组不是再扫小规模，而是：
  - **`stratified tr800 full, N=16, beta=0.2, gap=0.10`**
  - 这组将直接回答：当前 best 超参在完整 `tr800` 上能否保持 `tr400` 的优势。

## 2026-03-18 — version_92 追加结果（tr800 full gap010 / tr1200 full gap005）

新增完成目录：
- `results/dpo_v92_stratified_tr800_full_n16_b02_gap010`
- `results/dpo_v92_stratified_tr1200_full_n16_b02_gap005`

### 关键对比

**1. `tr800 full, N=16, beta=0.2`：`gap=0.10` 并没有比 `gap=0.05` 更好**
- `gap=0.05`:
  - `delta_mean=+0.3324`
  - `delta_max=+0.0780`
  - `improve_max=72/100`
  - `delta_logprob=-2.7210`
  - `has_run=0%`
- `gap=0.10`:
  - `delta_mean=+0.3145`
  - `delta_max=+0.0612`
  - `improve_max=64/100`
  - `delta_logprob=-2.7500`
  - `has_run=0%`
- 结论：`gap=0.10` 在 `tr400` 上是有利的，但放大到完整 `tr800` 后并未延续优势；`tr800 full` 上当前仍是 `gap=0.05` 更好。

**2. 继续放量到 `tr1200 full, N=16, beta=0.2, gap=0.05` 仍可运行，但收益继续回落**
- `delta_mean=+0.2946`
- `delta_max=+0.0704`
- `improve_max=68/100`
- `delta_logprob=-1.3047`
- `delta_diversity=+0.1240`
- `has_run=3%`
- 结论：从 `tr800 full` 再放大到 `tr1200 full`，`max@N` 收益继续下降，且开始出现轻微病态；不过 `logprob` 反而明显好于 `tr800 full`。这说明进一步放量没有带来稳定收益，更多是在改变 trade-off，而不是单调改进。

### 当前最重要判断
- `version_92` 下，已经基本可以确认：
  - `NUM_SAMPLES=16` 是更合适的 rollout 规模；
  - `beta=0.2` 仍是当前合理点；
  - `score_gap_delta` 的最优值会随训练规模变化：
    - `tr400` 上 `gap=0.10` 更好；
    - `tr800 full` 上 `gap=0.05` 更好。
- 因此，**不能把 `tr400` 上找到的 best 超参直接外推到更大规模**。
- 现阶段最稳的 production-style 选择仍然是：
  - 如果优先要更强的 `max@N` 提升，用 `tr400, N=16, beta=0.2, gap=0.10`；
  - 如果优先要更大覆盖面且保持稳定，用 `tr800 full, N=16, beta=0.2, gap=0.05`。

## 2026-03-18 — `stratified_tr800_va100_te100` 来源池分布特征

基于 `data/dpo/subsets/stratified_tr800_va100_te100/subset_manifest.csv`：

### Train split（800 backbones）
- `800/800` 都来自不同 cluster（无重复 cluster）。
- 长度分布极度集中：
  - `len_0_49`: 4
  - `len_50_59`: 1
  - `len_60_71`: 795
- `seq_len` 统计：
  - mean `67.12`
  - median `68`
  - IQR `65-70`
  - min/max `38/71`
- `log2_fold_change_75_clip` 被近乎完美地按五等分平衡：
  - `q1`: 159
  - `q2`: 160
  - `q3`: 160
  - `q4`: 160
  - `q5`: 161
- 交叉分布表明：当前 stratified 方案在 train 上本质上是 **“按 75°C aggregation label 五等分 + 几乎全部来自 60-71 aa 长度段”**。

### 对 full train pool 的含义
对照 `data/dpo/representative_pdbs/representatives.csv` 的 train split（4744 backbones）：
- 原始 train 本来就几乎全是 `60-71 aa`：
  - `len_38_49`: 8
  - `len_50_59`: 2
  - `len_60_71`: 4734
- 因此，`stratified_tr800` 并没有真正提供“长度多样性”；它主要提供的是：
  - **cluster 去重**
  - **75°C aggregation label 的均衡覆盖**

### Valid/Test
- valid `100`：全部来自 `len_60_71`，label bin 完全均衡（20×5）
- test `100`：`len_60_71` 为主（96/100），label bin 也完全均衡（20×5）

### 结论
- `tr800 full` 的 representation backbone 分布特点，不是“结构长度很多样”，而是：
  - **几乎全部都是 60-71 aa 的小结构域**；
  - **cluster 覆盖干净**；
  - **75°C aggregation label 覆盖非常均衡**。
- 所以你后面如果说“扩大到 tr800/full 后囊括了更多蛋白”，更准确的表述应该是：
  - **囊括了更多 cluster / 更多 aggregation label 区间的 representative backbone**，
  - 而不是囊括了更多长度类型。

## 2026-03-18 — full train (`4744` representatives) 的 `log2_fold_change_75_clip` 分布

基于 `data/dpo/representative_pdbs/representatives.csv` 中 `split == train` 的 `4744` 个 cluster-representative backbone：

### 总体统计
- count: `4744`
- mean: `-0.7244`
- std: `1.0449`
- min/max: `-5.1886 / 1.0400`
- 25% / 50% / 75%: `-1.2497 / -0.3585 / 0.0243`

### 分位点
- 10%: `-2.3245`
- 20%: `-1.5588`
- 40%: `-0.6234`
- 60%: `-0.1620`
- 80%: `0.0787`
- 90%: `0.2220`

### 粗直方图
- `< -4`: `42`
- `[-4, -3)`: `164`
- `[-3, -2)`: `449`
- `[-2, -1)`: `778`
- `[-1, 0)`: `2020`
- `[0, 0.5)`: `1147`
- `[0.5, 1.04]`: `144`

### 当前 `q1~q5` 的边界（按 train split 五等分）
- `q1`: `(-5.19, -1.559]`
- `q2`: `(-1.559, -0.623]`
- `q3`: `(-0.623, -0.162]`
- `q4`: `(-0.162, 0.0787]`
- `q5`: `(0.0787, 1.04]`

### 直观解释
- 分布明显偏向负值，说明大部分 representative backbone 的 `75°C` 标签落在较低区间。
- 中位数 `-0.3585`，而 75% 分位才刚过 `0`，说明正值样本不是主流。
- 你现在的 `q1~q5` 分桶，本质上是在这个偏负的分布上做等频切分，而不是按绝对数值阈值切分。

## 2026-03-18 — 新增 capped-bin 数据子集构造

已在 `scripts/select_dpo_subset.py` 中加入 `--mode capped_bin`：
- 在绝对 label 值空间按固定 bin 宽度分箱；
- 每个非空 bin 最多保留 `max_per_bin` 个样本；
- 稀疏 bin 全保留；
- 适合削弱 `log2_fold_change_75_clip` 的长尾峰值主导。

已构造子集：
- `data/dpo/subsets/capped_bin_bw0.5_cap100`

参数：
- `bin_width = 0.5`
- `max_per_bin = 100`
- `seed = 42`

结果规模：
- train: `1016`
- valid: `499`
- test: `721`

说明：
- capped-bin 模式当前按 split 独立削峰，不再强制使用 `train_size/valid_size/test_size`。
- 如果后续要做正式 DPO，可直接把该子集作为来源池，再通过 `MAX_*_PDBS` 控制实际 run 规模。

## 2026-03-19 — capped-bin 训练集首次正式结果（te100 对齐）

目录：
- `results/dpo_v92_cappedbin_bw05_cap100_tr1000_te100_n16_b02_gap005`

设置：
- train pool: `data/dpo/subsets/capped_bin_bw0.5_cap100/train`（实际约 `1016` 个）
- test pool: `data/dpo/subsets/stratified_tr800_va100_te100/test`
- `MAX_TEST_PDBS=100`
- annotator: `version_92`
- `N=16`, `beta=0.2`, `gap=0.05`

结果：
- `delta_mean=+0.3506`
- `delta_max=+0.0924`
- `improve_mean=85/100`
- `improve_max=79/100`
- `delta_logprob=-1.8005`
- `delta_diversity=+0.1206`
- `has_run=2%`
- `top_frac_p90=0.3030`
- `charged_frac_p90=0.4921`

对比：
- 相比 `tr800 full stratified, gap=0.05`：
  - `delta_max` 更好（`+0.0924` vs `+0.0780`）
  - `improve_max` 更高（`79%` vs `72%`）
  - `delta_logprob` 更好（`-1.8005` vs `-2.7210`）
- 相比 `tr1200 full stratified, gap=0.05`：
  - `delta_max` 更好（`+0.0924` vs `+0.0704`）
  - `improve_max` 更高（`79%` vs `68%`）
  - `logprob` 稍差于 `tr1200`（`-1.8005` vs `-1.3047`），但整体 trade-off 更优。

结论：
- capped-bin 训练集第一次正式尝试就是正向结果。
- 与同样 `te100` 对齐的 stratified 放量方案相比，capped-bin 更接近我们想要的目标：
  - 更好的 `max@N`
  - 更好的覆盖率
  - 没有明显 collapse
- 这说明“按绝对 label 空间削峰”比单纯 quantile stratification 更适合 version_92 当前的 DPO 对齐。

## 2026-03-19 — threshold-balance 子集（保留所有 `< -1`，从 `>= -1` 随机抽等量）

已新增 `scripts/select_dpo_subset.py --mode threshold_balance`，支持按阈值做二分平衡抽样。

已生成子集：
- `data/dpo/subsets/threshold_balance_ltneg1_eq`

规则：
- 保留全部 `log2_fold_change_75_clip < -1`
- 从 `log2_fold_change_75_clip >= -1` 中随机抽取等数目样本
- `seed = 42`

结果规模：
- train: `2866`（`1433 < -1` + `1433 >= -1`）
- valid: `366`（`183 + 183`）
- test: `778`（`389 + 389`）

分布图：
- `results/plots/train_log2_fold_change_75_threshold_balanced_minus1.png`

说明：
- 这个方案比 capped-bin 更温和：保留了更多样本量，同时明显削弱了 `>= -1` 区间的主导性。
- 如果后续正式做 DPO，对齐 `te100` 评估时，仍建议继续使用旧的 `stratified_tr800_va100_te100/test` 作为测试来源池，以保持和已有实验可比。

## 2026-03-19 — threshold-balance 训练集上的 4 组正式超参搜索

目录：
- `results/dpo_v92_thbal_m1_tr2866_te100_n16_b01_gap005`
- `results/dpo_v92_thbal_m1_tr2866_te100_n16_b01_gap010`
- `results/dpo_v92_thbal_m1_tr2866_te100_n16_b02_gap005`
- `results/dpo_v92_thbal_m1_tr2866_te100_n16_b02_gap010`

统一设置：
- train: `data/dpo/subsets/threshold_balance_ltneg1_eq/train` (`2866`)
- valid: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid` (`366`)
- test: `data/dpo/subsets/stratified_tr800_va100_te100/test`, `MAX_TEST_PDBS=100`
- annotator: `version_92`
- `N=16`, `T=0.5`

### 结果总览
- `beta=0.1, gap=0.05`
  - `delta_mean=+0.3556`
  - `delta_max=+0.1014`
  - `improve_max=80/100`
  - `delta_logprob=-1.2595`
  - `has_run=0%`
- `beta=0.1, gap=0.10`
  - `delta_mean=+0.3887`
  - `delta_max=+0.1047`
  - `improve_max=87/100`
  - `delta_logprob=-1.3356`
  - `has_run=0%`
- `beta=0.2, gap=0.05`
  - `delta_mean=+0.2044`
  - `delta_max=+0.0498`
  - `improve_max=68/100`
  - `delta_logprob=-0.2703`
  - `has_run=12%`
- `beta=0.2, gap=0.10`
  - `delta_mean=+0.2445`
  - `delta_max=+0.0684`
  - `improve_max=71/100`
  - `delta_logprob=-0.5339`
  - `has_run=2%`

### 结论
- threshold-balance 训练集上，**`beta=0.1` 明显优于 `beta=0.2`**。
- 在 `beta=0.1` 内部，`gap=0.10` 又优于 `gap=0.05`。
- 当前这 4 组里的最佳配置是：
  - **`beta=0.1, gap=0.10`**
  - 它给出：`delta_max=+0.1047`, `improve_max=87%`, `delta_logprob=-1.3356`, `has_run=0%`

### 重要观察
- `beta=0.2, gap=0.05` 的训练阶段指标其实最好看：
  - `val_loss` 最低
  - `val_accuracy` 最高
- 但它的下游结果反而最差，并且出现 `12%` 的病态序列。
- 这再次说明：
  - **`val_loss / val_accuracy / reward_margin` 不能作为 DPO 主要选模指标**；
  - 真正应该看的仍然是 `delta_max`、`improve_max`、`logprob`、`has_run`。

### 和此前最佳结果对比
- 相比 capped-bin 方案 `results/dpo_v92_cappedbin_bw05_cap100_tr1000_te100_n16_b02_gap005`：
  - `delta_max`: `+0.1047` vs `+0.0924`
  - `improve_max`: `87%` vs `79%`
  - `delta_logprob`: `-1.3356` vs `-1.8005`
  - `has_run`: `0%` vs `2%`
- 因此，当前 **全局最优** 已更新为：
  - `threshold_balance_ltneg1_eq + version_92 + N=16 + beta=0.1 + gap=0.10`

## 2026-03-19 — 当前全局最优模型的 full test (`1350`) 结果

目录：
- `results/dpo_v92_thbal_m1_fulltest1350`

评估模型：
- `results/dpo_v92_thbal_m1_tr2866_te100_n16_b01_gap010/mpnn_dpo_best.pt`
- 对 `data/dpo/representative_pdbs/test` 全部 `1350` 个 representative backbone 评估

结果：
- `delta_mean=+0.3906`
- `delta_max=+0.1098`
- `improve_mean=1242/1350 (92.0%)`
- `improve_max=1142/1350 (84.6%)`
- `delta_logprob=-1.3408`
- `delta_diversity=+0.0924`
- `has_run=0.59%`
- `top_frac_p90=0.2687`
- `charged_frac_p90=0.5143`

### 结论
- 这组 full-test 结果非常强，而且和 `te100` 小规模正式验证保持一致，没有出现放大后退化。
- 相比同一模型在 `te100` 上的结果（`delta_max=+0.1047`, `improve_max=87%`, `has_run=0%`）：
  - full `1350` 上 `delta_max` 进一步提升到 `+0.1098`
  - `improve_max` 仍保持在高位（`84.6%`）
  - 病态比例仍极低（`0.59%`）
- 这说明当前 best 配置并不是只在小 test 子集上偶然成立，而是在完整 test split 上也稳定成立。

### 当前最佳结论
- 当前全局最优配置：
  - `threshold_balance_ltneg1_eq + version_92 + N=16 + beta=0.1 + gap=0.10`
- 该配置已经在：
  - `te100` 正式对齐实验
  - `test=1350` 全集评估
  上都验证通过。
- 因此，后续优先级应从“继续大规模调参”切换到“下游分析 / case study / 结构复核”。

## 2026-03-19 — 新增 epoch sweep 脚本

新增脚本：
- `scripts/run_dpo_epoch_sweep.sh`

用途：
- 固定 train/val pairs，不重新采样；
- 对多个 `epochs` 依次执行：训练 -> 选 best ckpt -> full test 评估 -> 汇总；
- 适合当前 best 配置的 epoch 敏感性检查。

默认参数已对齐当前全局最优配置：
- train pairs: `data/dpo/v92_thbal_m1_tr2866_te100_n16_b01_gap010_train_pairs.pt`
- val pairs: `data/dpo/v92_thbal_m1_tr2866_te100_n16_b01_gap010_val_pairs.pt`
- test set: `data/dpo/representative_pdbs/test`
- `N=16`, `beta=0.1`, `gap=0.10`
- 默认 `EPOCHS_LIST="12 15 20 25"`

## 2026-03-20 — epoch sweep 训练完成，但自动 full-test 评估失败

运行目录：
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e15`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e20`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e25`

训练部分全部完成，且都保存了 `mpnn_dpo_best.pt`。

### 训练内验证趋势
- `e12`: `best_val_loss=0.3758`, `best_epoch=12`
- `e15`: `best_val_loss=0.3530`, `best_epoch=15`
- `e20`: `best_val_loss=0.3201`, `best_epoch=20`
- `e25`: `best_val_loss=0.2910`, `best_epoch=25`

说明：
- 在当前配置下，validation DPO loss 随 epoch 持续下降；
- 训练内最佳 checkpoint 一直出现在最后一轮；
- 单从训练内指标看，更多 epoch 会继续改善 `val_loss / val_accuracy / reward_margin`。

### 但自动评估没有成功
`*_fulltest` 目录为空。日志显示自动评估阶段失败，原因是：
- `proagg_ckpt` 参数被错误解析成了目录 `results/lightning_logs/version_92/checkpoints`，而不是具体 checkpoint 文件；
- 同时 `cuda:X` 被 shell 误解析为单独命令。

因此，这一轮 **只能得出训练内验证趋势，不能得出 full-test 最终优劣结论**。

## 2026-03-20 — 修复 epoch sweep / eval sweep 脚本参数解析

修复内容：
- `scripts/run_dpo_epoch_sweep.sh`
- `scripts/run_dpo_epoch_eval_sweep.sh`

修复点：
- 支持通过环境变量 `PROAGG_CKPT` / `DEVICE` 传参；
- 如果传入的是 checkpoint 目录（如 `results/lightning_logs/version_92/checkpoints`），脚本会自动解析其中的 `best_epoch=*.ckpt`；
- 对缺失 checkpoint 会直接报错，而不是进入错误评估流程。

原因：
- 之前后台命令里，`proagg_ckpt` 被 shell 误传成了 checkpoint 目录，导致 `evaluate.py` 在加载 ProAgg 时把目录当文件打开；
- 修复后即使传目录，也会自动解析到具体 `.ckpt` 文件。

## 2026-03-21 — epoch sweep full-test (`1350`) 结果

目录：
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12_fulltest`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e15_fulltest`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e20_fulltest`
- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e25_fulltest`

统一配置：
- train/val pairs 固定为当前全局最优配置对应的 threshold-balanced pairs
- 只改变训练 epoch 上限
- full test = `1350` representative backbones

### full-test 指标
- `e12`
  - `delta_mean=+0.4067`
  - `delta_max=+0.1211`
  - `improve_max=1158/1350 (85.8%)`
  - `delta_logprob=-1.5244`
  - `has_run=0.59%`
- `e15`
  - `delta_mean=+0.3623`
  - `delta_max=+0.1137`
  - `improve_max=1139/1350 (84.4%)`
  - `delta_logprob=-1.4104`
  - `has_run=1.11%`
- `e20`
  - `delta_mean=+0.2630`
  - `delta_max=+0.0805`
  - `improve_max=1036/1350 (76.7%)`
  - `delta_logprob=-1.0530`
  - `has_run=5.70%`
- `e25`
  - `delta_mean=+0.1869`
  - `delta_max=+0.0154`
  - `improve_max=859/1350 (63.6%)`
  - `delta_logprob=-0.8056`
  - `has_run=20.96%`

### 结论
- **最佳 epoch 不是 10，而是 12。**
- `e12` 在 full test 上超过此前 `e10`：
  - `delta_max: +0.1211` vs `+0.1098`
  - `improve_max: 85.8%` vs `84.6%`
- 但继续增加 epoch 会系统性恶化：
  - `e15` 开始回落；
  - `e20` 明显变差；
  - `e25` 已接近再次进入 collapse 区间（`has_run=20.96%`, `top_frac_p90=0.831`）。

### 关键观察
- 训练内 `val_loss` 随 epoch 持续下降：
  - `e12`: `0.3758`
  - `e15`: `0.3530`
  - `e20`: `0.3201`
  - `e25`: `0.2910`
- 但 full-test 下游指标同时持续恶化。
- 这进一步证明：
  - **validation DPO loss 与最终 design-time 指标是错位的**；
  - 更低的 `val_loss` 并不代表更好的 `max@N`，反而可能意味着更强的过度对齐和病态序列增加。

### 当前全局最佳更新
- 当前全局最佳配置更新为：
  - `threshold_balance_ltneg1_eq + version_92 + N=16 + beta=0.1 + gap=0.10 + epoch=12`
- 对应 full-test 目录：
  - `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12_fulltest`
