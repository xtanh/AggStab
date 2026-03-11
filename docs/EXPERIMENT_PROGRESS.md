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
