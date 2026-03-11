# DPO v2 结果排查与问题分析（系统复盘）

> 更新日期：2026-03-10  
> 目标：对 `results/dpo_v2/` 的评测结果做“可复现、可定位”的系统排查，明确当前 v2 的主要问题形态、可能根因与下一步解决路径。

---

## 1. 本次排查所依据的产物

- DPO v2 训练记录：`results/dpo_v2/training_history.json`
- DPO v2 评估 JSON：`results/dpo_v2/eval_results.json`
- DPO v2 评估汇总：`results/dpo_v2/dpo_eval_report.txt`、`results/dpo_v2/dpo_eval_summary.csv`

评估脚本为：`src/dpo/evaluate.py`（同一次评估中同时跑 ORIGINAL 与 DPO-finetuned）。

---

## 2. v2 的“宏观结果”：mean 上升，但 max@N 与 diversity 崩溃

从 `results/dpo_v2/dpo_eval_report.txt`：

- N(test backbones)=1350
- ProAgg mean：orig=-0.2633，dpo=-0.1232，**delta=+0.1401**
- ProAgg max@N：orig=0.1142，dpo=-0.0815，**delta=-0.1957**
- MPNN logprob：orig=-1.2974，dpo=-0.2214，**delta=+1.0761**
- Diversity：orig=0.5175，dpo=0.0434，**delta=-0.4741**

按 backbone 计数：
- mean 提升：811/1350（60.1%）
- max 提升：103/1350（7.6%）

**核心矛盾**：如果下游使用方式是 best-of-N（对同一 backbone 采样 N 条，取 `ProAgg max@N`），那么 v2 的结果在目标指标上是显著退化的；并且伴随强烈的多样性崩溃。

---

## 3. v2 的“分布结果”：`delta_max` 基本全为负

根据 `results/dpo_v2/dpo_eval_summary.csv` 的分位数统计（基于 `delta_max`）：

- `delta_max` 中位数为 **-0.2171**
- 90 分位为 **-0.0342**（仍为负）
- 10 分位为 **-0.3281**

这意味着：**不是少数失败样本拖累均值，而是“几乎全体 backbones 的 best-of-N 都变差”。**

这通常对应两类原因之一：

1) 采样分布变尖（tail 消失）→ best-of-N 不再有效；或  
2) 生成了大批“看似符合训练偏好、但对真实评估目标无效/有害”的序列形态（例如低复杂度、极端组成）。

---

## 4. v2 的“序列形态”异常：低复杂度/同聚序列占比极高

对 `results/dpo_v2/eval_results.json` 中 **DPO 模型的 best_seq** 做抽样统计（200 个随机 backbones，统计 best-of-12 中的 best_seq）：

- ORIGINAL 的 best_seq：
  - 最常见氨基酸占比（top_frac）均值约 **0.17**
  - hydrophobic（AILMFWV）占比均值约 **0.34**
  - charged（DEKRH）占比均值约 **0.35**
  - 长同聚 run（同一 AA 连续 ≥6）计数：**0/200**

- DPO v2 的 best_seq：
  - top_frac 均值约 **0.94**
  - hydrophobic 占比均值约 **0.02**
  - charged 占比均值约 **0.96**
  - 长同聚 run（同一 AA 连续 ≥6）计数：**197/200**

此外，在 `delta_max` 最差的一批样本中，best_seq 的 top_frac 甚至达到 0.97~0.99，且几乎全为带电残基。

**结论**：v2 的输出中出现了极其明显的低复杂度/近似同聚（poly-charged）序列。这是“多样性崩溃”的直接证据，也提示存在强烈的目标偏置：模型把大量概率质量集中到某种极端序列模式上。

> 注：这种序列形态会天然降低疏水聚集倾向（因此可能被 ProAgg 偏好），但它们往往不是合理的 fixed-backbone 设计解；同时会让 best-of-N 的增益消失（因为采样空间近似退化）。

---

## 5. 可能根因（按优先级）

### 根因 A（P0）：目标与使用方式不匹配（offline DPO vs best-of-N）

DPO 训练优化的是“给定 winner/loser 的相对偏好”（提高 winner 的相对 logprob），并不直接优化“从模型采样分布的尾部质量（tail）”。

当训练把分布推尖：
- mean 可能上升（集中到某类平均更安全的序列）
- tail 消失 → `max@N` 明显下降（best-of-N 失效）

当前 v2 同时出现 `mean↑`、`max@N↓`、`diversity↓↓`，与此模式高度一致。

### 根因 B（P0）：偏好信号过“单一维度”，导致组成极端化（composition collapse）

从 best_seq 的组成统计看，模型几乎把序列推成“高带电、低疏水、低复杂度”的极端模式。  
这往往意味着偏好/奖励信号只对某些简单统计量敏感（例如疏水比例），而缺少对“可设计性/结构一致性/序列复杂度”的约束。

### 补充证据：1350 个 backbones 的整体形态统计与相关性

对全部 1350 个 test backbones（评估参数 N=12, T=0.5），统计 DPO v2 best_seq 的形态指标：

- `has_run(≥6)`（存在同一 AA 连续 ≥6）：**98.44%**
- `top_frac`（单一 AA 占比）：mean=**0.9415**，p50=**0.9688**，p90=**0.9857**
- `charged_frac(DEKRH)`：mean=**0.9626**，p50=**0.9841**，p90=**1.0000**
- `hydrophobic_frac(AILMFWV)`：mean=**0.0197**，p50=**0.0143**，p90=**0.0423**

与 `delta_max` 的相关性（Pearson）：

- corr(`delta_max`, `top_frac`) = **-0.2407**
- corr(`delta_max`, `charged_frac`) = **-0.2328**
- corr(`delta_max`, `hydrophobic_frac`) = **+0.2362**
- corr(`delta_max`, `diversity`) = **+0.1991**

解释：越接近“单一带电残基同聚”，`max@N` 越差；即便 `n_unique` 不总是 1，这类序列形态仍能让 best-of-N 失去意义。

### 根因 B2（P0）：训练 pairs 存在轻微“带电↑/疏水↓”偏置，DPO 可能放大该偏置

对训练/验证 pairs（`data/dpo/train_pairs.pt`、`data/dpo/val_pairs.pt`）汇总 AA 频率（统计所有 winner 序列拼接后的 AA 分布 vs loser）：

- Train pairs（28464 对）：
  - winner：charged=**0.3383**，hydrophobic=**0.3582**
  - loser： charged=**0.3035**，hydrophobic=**0.3945**
- Val pairs（3942 对）：
  - winner：charged=**0.3390**，hydrophobic=**0.3581**
  - loser： charged=**0.3020**，hydrophobic=**0.3954**

Top 差异 AA（winner - loser）在 train/val 上一致：
- K、E 明显更高；R、A、L 更低（详见 `scripts/analyze_dpo_v2_pathology.py` 输出）

结论：pairs 的偏好信号确实倾向“更带电、更少疏水”。该偏置本身并不极端，但在较强的偏好优化强度（例如较大 `beta`）和缺少反塌缩约束时，可能被放大为“极端同聚/低复杂度”。

### 根因 C（P1）：`beta` 在当前实现中是偏好梯度强度，而非显式 KL 系数

在 `src/dpo/dpo_train.py` 中，loss 为：

`-logsigmoid(beta * (log_ratio_w - log_ratio_l))`

因此 `beta` 越大越“激进”。如果把它当成“更保守/更强 KL”并单调增大，可能适得其反：更容易造成分布变尖与组成崩溃。

### 根因 D（P1）：checkpoint 选择指标不对齐（val_loss/acc vs max@N/diversity）

如果始终用离线 `val_loss/val_accuracy` 选 best，很可能会系统性偏向“margin 拉得更开”的模型，而不是“采样后 max@N 更好且不塌缩”的模型。

---

## 6. 当前最需要先回答的 3 个问题（最小证据链）

1) **训练 pairs 的 winner 是否确实富集带电残基？**
   - 结论：**是（轻微但一致）**。winner 的 charged 比例更高、hydrophobic 更低（见上文“根因 B2”）。
   - 含义：DPO 会倾向放大这一偏置；若缺少反塌缩约束，最终可能走向极端组成（composition collapse）。

2) **v2 的 collapse 是“采样温度过低”还是“模型分布本身过尖”？**
   - 现在评估是 T=0.5 仍然塌缩，说明主要是模型分布问题，而非仅温度设置。
   - 仍建议 sweep T（例如 0.3/0.5/0.7/1.0）观察 `n_unique/diversity/max@N` 的趋势。

3) **这些低复杂度序列在 fixed-backbone 下是否明显不可行？**
   - 需要引入额外 sanity checks（例如基本的氨基酸组成约束、低复杂度过滤、或结构一致性代理）来确认这类序列是否应被禁止/惩罚。

---

## 7. 下一步（优先级建议，先止血再优化）

1) **先止血：在评估/产出侧加过滤与约束**
   - 过滤低复杂度/长同聚 run（例如同一 AA 连续 ≥6）
   - 加氨基酸组成约束（限制 charged/hydrophobic 比例的极端值）
   - 这不是最终方案，但能快速避免明显病态输出污染后续分析。

2) **beta sweep（优先）**
   - 尝试显著降低 beta（0.01/0.05/0.1/0.2/0.5）
   - 观察 `diversity` 与 `proagg_max@N` 是否回升，确认“偏好梯度过强导致 collapse”的假说。

3) **把 checkpoint 选择对齐到 `max@N` + collapse penalty**
   - 每个 epoch 在小规模 valid backbones 上做在线采样评估，记录 `max@N` 与 diversity；
   - 用 `max@N - λ * collapse_penalty` 选 best ckpt，而不是只用离线 val_loss。

4) **online DPO（中期）**
   - 定期用当前 policy 重新采样→ProAgg 重新打分→刷新 pairs，缓解 off-policy 与分布漂移。

---

## 8. 可复现排查命令

### 8.1 评估产物的病态统计（无需 torch）

```bash
python scripts/analyze_dpo_v2_pathology.py \
  --eval_json results/dpo_v2/eval_results.json \
  --summary_csv results/dpo_v2/dpo_eval_summary.csv
```

输出：
- `results/dpo_v2/dpo_v2_pathology_enriched.csv`
- 终端打印：同聚 run 比例、top/charged/hydro 分位数、与 `delta_max` 的相关性

### 8.2 训练/验证 preference pairs 的组成偏置（需要 torch）

```bash
bash -lc 'eval "$(conda shell.bash hook)"; conda activate SaProt; \
  python scripts/analyze_dpo_v2_pathology.py \
    --eval_json results/dpo_v2/eval_results.json \
    --summary_csv results/dpo_v2/dpo_eval_summary.csv \
    --train_pairs_pt data/dpo/train_pairs.pt \
    --val_pairs_pt data/dpo/val_pairs.pt'
```

---

## 9. 最小复现实验（建议先做这一组）

目标：验证 “`beta` 过激 + 缺少反塌缩约束 → 组成崩溃 → `max@N` 退化” 是否为主因。

### 9.1 实验矩阵（保持其它不变）

- 训练：固定同一份 pairs（不重新采样），只 sweep `beta`
  - `beta ∈ {0.01, 0.05, 0.1, 0.2, 0.5}`
- 评估：固定 `T=0.5`，同时增加一个更大的 N 用于验证 best-of-N
  - `N ∈ {12, 64}`（至少要评到 64）

### 9.2 判据（每次训练都要输出）

对 test（或小规模 valid 在线评估）同时看四类指标：

1) `proagg_max@N`（主要目标）
2) `proagg_mean`（辅助）
3) `diversity/n_unique`（是否塌缩）
4) 病态序列比例（`has_run(>=6)`、`top_frac`、`charged_frac`）

预期：若降低 `beta` 能明显降低病态比例并恢复 `proagg_max@N`，则说明 collapse 是主因；否则需更强约束（例如显式正则/在线刷新 pairs/改 pairing 策略）。
