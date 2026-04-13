# 当前结果总结（中文）

## 1. 文档目的

本文档总结目前项目已经得到的主要结果，包括：

- 单目标 aggregation DPO 的探索结果
- 双目标 / joint-preference DPO 的探索结果
- semi-online joint DPO 的最终工作流
- 候选序列导出
- Chai-1 结构预测
- 最终用于分析的目录、文件和脚本

当前推荐的最终模型是：

- 运行目录：`results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`
- 最终选中的 checkpoint：`results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dpo/mpnn_dpo_epoch1.pt`

---

## 2. 问题定义

本项目的目标是在固定 backbone 的 inverse folding 场景下，同时优化：

- 抗聚集能力（`ProAgg`）
- 稳定性预测（`deltaG`）

同时尽量保持：

- ProteinMPNN 的设计性先验（`mpnn_logprob`）
- 下游结构可信度（Chai-1 的 `pLDDT`、`pTM`）

整个项目中反复观察到的核心问题是：

- 如果过度优化 aggregation，稳定性常常会下降；
- 如果训练目标设计不够好，即使引入双目标，也可能重新偏离 ProteinMPNN 的合理序列分布。

---

## 3. 属性模型与建模假设

当前所有属性优化实验都基于 **SaProt predictor**。

### aggregation predictor

- checkpoint：`results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- config：`configs/proagg_final_candidate.yaml`

### stability predictor

- checkpoint：`results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt`
- config：`configs/proagg_deltaG_only.yaml`

### 重要假设

对于一条新设计序列，属性打分时使用的是原 backbone 对应的结构 token。

因此这里的属性预测本质上是：

- **fixed-backbone conditional property prediction**

也就是说：

- 这是在“默认 backbone 不变”的条件下评估序列属性；
- 最终仍然必须通过结构预测做验证。

---

## 4. 最终使用的数据划分

### 训练 / 验证 / 测试

- train：`data/dpo/subsets/threshold_balance_ltneg1_eq/train`
- pair 构造用 validation：`data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- checkpoint selection 用 validation：`data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- final test：`data/dpo/representative_pdbs/test`

### 为什么这样划分

- train/valid 都采用 threshold-balanced 子集，是因为这部分 backbone 本身是围绕 aggregation 目标进行筛选和平衡过的；
- final test 保持使用 representative full test，用于最终泛化报告。

---

## 5. DPO 实验路线复盘

### 5.1 单目标 aggregation DPO

最开始的 DPO 只优化 aggregation。

主要现象：

- aggregation 确实能提升；
- 但 stability 往往会下降；
- 训练中的 `val_loss` 不能可靠反映最终设计质量。

相关结果目录：

- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12_fulltest`

这一阶段的主要结论是：

- threshold-balanced 数据分布是有帮助的；
- offline DPO 可以推动 aggregation；
- 但 aggregation–stability trade-off 很明显。

### 5.2 dual mixing

随后实现了 dual-objective mixing：

- `src/dpo/sample_and_score_dual.py`
- `scripts/run_dpo_dual_pipeline.sh`

做法是：

- 单独构造 aggregation pairs
- 单独构造 stability pairs
- 训练时混合采样

结果表明：

- 这种方式不够强；
- 在大规模训练下，stability 目标容易被 aggregation 淹没。

### 5.3 static joint-preference DPO

接着转向 joint-preference：

- `src/dpo/sample_and_score_joint.py`
- `scripts/run_dpo_joint_pipeline.sh`

joint 的定义是：

- winner 必须同时在 `ProAgg` 和 `deltaG` 上优于 loser；
- 并经过 stability gate 过滤。

这比 dual mixing 更好，但 full-scale static training 仍然会失稳。

### 5.4 semi-online joint DPO

最终有效的方向是：

- `scripts/run_dpo_joint_semi_online_pipeline.sh`

核心思想：

- 每一轮都用当前 policy 重新采样候选
- 重新构造 joint pairs
- 每轮只训练很短时间
- 每轮训练后用 validation 上的 downstream 指标选 checkpoint

这样做的原因是：

- static offline pair 会随着 policy 漂移而逐渐失真；
- semi-online refresh 可以缓解这一问题。

---

## 6. 当前最佳训练方案

当前最优方案是：

- **semi-online joint DPO**
- 一共 `2` 个 round
- 每个 round 只训练 `1` 个 epoch
- train backbone 切成两个互不重叠的随机 half
- round 0 用第一半
- round 1 用第二半
- 每个 round 之后做 validation-based checkpoint selection

### 具体超参数

- `NUM_SAMPLES=16`
- `TEMPERATURE=0.5`
- `DPO_BETA=0.1`
- `ROUNDS=2`
- `ROUND_EPOCHS=1`
- `ROUND_TRAIN_SPLIT_MODE=halves`
- `ROUND_TRAIN_SPLIT_SEED=42`
- `AGG_SCORE_GAP_DELTA=0.20`
- `STAB_SCORE_GAP_DELTA=0.20`
- `STABILITY_GATE_MODE=wt_absolute`
- `STABILITY_GATE_MARGIN=0.5`
- `VALID_SELECTION_ENABLED=1`
- `VALID_SELECTION_METRIC=joint_sum`
- `VALID_SELECTION_MAX_PENALTY=0.1`

### 为什么要用 disjoint halves

如果每一轮都使用 full train pool：

- 更新太激进；
- 很容易重新把模型推偏；
- full-scale 下 stability 会再次崩掉。

而 disjoint halves 的优点是：

- 每轮压力更小；
- 两轮加起来仍覆盖完整的 threshold-balanced train 数据；
- 计算量也更可控。

---

## 7. checkpoint selection 的逻辑

当前明确采用：

- **不能用 test 选模型**
- **也不能只看 training 的 val_loss**

而是：

- 对每个保存下来的 epoch checkpoint
- 在 threshold-balanced valid 上跑一次 downstream dual evaluation
- 再据此选择 best checkpoint

### 使用脚本

- `scripts/select_best_joint_checkpoint.py`

### 选择规则

#### Tier A

validation 上必须同时满足：

- `delta_proagg_mean >= 0`
- `delta_proagg_max >= 0`
- `delta_deltaG_mean >= 0`
- `delta_deltaG_max >= 0`

在这些 checkpoint 中，按：

- `joint_sum = delta_proagg_mean + delta_deltaG_mean`

排序，取最佳。

#### Tier B

如果 Tier A 为空，则要求：

- `delta_proagg_mean >= 0`
- `delta_deltaG_mean >= 0`

仍按 `joint_sum` 排序。

#### Tier C

如果 Tier B 也为空，则在所有 checkpoint 中按：

- `joint_sum - 0.1 * (# negative max metrics)`

进行回退选择。

### 当前最佳 run 的 validation selection 记录

- round 0：
  - `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/valid_epoch_selection/best_checkpoint.json`
- round 1：
  - `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/valid_epoch_selection/best_checkpoint.json`

两个 round 最终选中的 checkpoint 都属于 **Tier A**。

---

## 8. baseline cache 复用

为了避免 valid/test 上反复重算 baseline ProteinMPNN 结果：

- `src/dpo/evaluate.py` 增加了 `--original_results_cache`

### validation baseline cache

- `results/cache/thbal_valid_original_n16_t05_seed42.json`

### test baseline cache

- `results/cache/test_original_n16_t05_seed42.json`

在以下条件不变时，这些 cache 可以复用：

- backbone 集不变
- `num_samples` 不变
- `temperature` 不变
- `seed` 不变

---

## 9. 最终最佳 run 的测试结果

### 运行目录

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`

### round 0 test

文件：

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/dual_eval_report.txt`

结果：

- `delta_proagg_mean = +0.4050`
- `delta_proagg_max = +0.1137`
- `delta_deltaG_mean = +0.1007`
- `delta_deltaG_max = +0.0735`
- `delta_logprob_mean = +0.1032`
- `improve_proagg_mean = 1346 / 1350`
- `improve_proagg_max = 1261 / 1350`
- `improve_deltaG_mean = 974 / 1350`
- `improve_deltaG_max = 893 / 1350`

### round 1 test

文件：

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dual_eval_report.txt`

结果：

- `delta_proagg_mean = +0.5038`
- `delta_proagg_max = +0.1413`
- `delta_deltaG_mean = +0.0691`
- `delta_deltaG_max = +0.0293`
- `delta_logprob_mean = +0.1472`
- `improve_proagg_mean = 1348 / 1350`
- `improve_proagg_max = 1298 / 1350`
- `improve_deltaG_mean = 876 / 1350`
- `improve_deltaG_max = 825 / 1350`

### 为什么最终选择 round 1

因为 round 1 在保持 stability 仍为正的前提下，进一步提升了：

- aggregation
- max aggregation
- logprob

因此 round 1 是当前推荐的最终 checkpoint。

---

## 10. 候选序列导出

### best run 候选

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv`

### baseline 候选

- `results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv`

### 导出脚本

- `scripts/export_dpo_test_candidates.py`

导出的字段包括：

- `pdb_name`
- `candidate_rank`
- `sequence`
- `proagg_score`
- `deltaG`
- `wt_deltaG`
- `deltaG_minus_wt`
- `mpnn_logprob`

注：

- 有 `5` 个 backbone 缺少 `wt_deltaG`，因此这些 backbone 的 `deltaG_minus_wt` 为 `NaN`

### 候选级别统计

#### best run

- 全部候选：
  - `proagg_score = 0.3097`
  - `deltaG = 3.6180`
  - `deltaG_minus_wt = 0.9224`
  - `mpnn_logprob = -1.1497`
- 每个 backbone 的 rank1：
  - `proagg_score = 0.4675`
  - `deltaG_minus_wt = 0.8761`
  - `mpnn_logprob = -1.1362`

#### baseline

- 全部候选：
  - `proagg_score = -0.1942`
  - `deltaG = 3.5489`
  - `deltaG_minus_wt = 0.8528`
  - `mpnn_logprob = -1.2969`
- 每个 backbone 的 rank1：
  - `proagg_score = 0.3262`
  - `deltaG_minus_wt = 0.7219`
  - `mpnn_logprob = -1.2731`

### 候选池层面的解释

这说明当前最佳模型并不是只生成了少数几个好样本，而是：

- 整个候选池整体更强；
- aggregation、stability、logprob 都优于 baseline。

---

## 11. 结构预测前的 joint top3 筛选

在送 Chai-1 结构预测之前，我们对每个 backbone 的候选进行了 joint-aware 的 top3 筛选。

### 使用脚本

- `scripts/select_joint_topk_candidates.py`

### 选择规则

每个 backbone 依次尝试：

1. `strict_joint`
   - 病态过滤通过
   - `proagg_score > 0`
   - `deltaG_minus_wt > 0`
2. `strict_proagg`
   - 病态过滤通过
   - `proagg_score > 0`
3. `pathology_only`
4. `no_filter`

### 输出文件

#### best run

- csv：`results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv`
- fasta：`results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.fasta`

#### baseline

- csv：`results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv`
- fasta：`results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.fasta`

### stage usage

#### best run

- `strict_joint: 1149`
- `strict_proagg: 198`
- `pathology_only: 3`

#### baseline

- `strict_joint: 874`
- `strict_proagg: 268`
- `pathology_only: 208`

### top3 候选统计

#### best run

- `proagg_score = 0.4320`
- `deltaG_minus_wt = 1.0131`
- `mpnn_logprob = -1.1373`
- 至少有一条 joint-positive top3 候选的 backbone：
  - `1171 / 1350`

#### baseline

- `proagg_score = 0.2470`
- `deltaG_minus_wt = 0.8352`
- `mpnn_logprob = -1.2775`
- 至少有一条 joint-positive top3 候选的 backbone：
  - `1049 / 1350`

这说明：

- 当前最佳模型在送去结构预测之前，已经提供了更干净、更符合 joint 目标的 top3 候选池。

---

## 12. Chai-1 结构验证

### 结构预测脚本

- `scripts/run_chai_batch_from_candidates.py`

### Chai 输出目录

#### best run

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint`

#### baseline

- `results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint`

### 主要输出文件

两边都使用了：

- `summary_shard0.csv`
- `summary_shard1.csv`
- `summary_shard2.csv`
- `summary_shard3.csv`
- `stats_shard0.json`
- `stats_shard1.json`
- `stats_shard2.json`
- `stats_shard3.json`

两边的 `4050 / 4050` 条候选全部成功完成结构预测。

---

## 13. 最终结构分析口径

### 分析约定

对于每条候选序列：

- Chai 会给出多个结构样本；
- 先从这些样本中选出该候选自己的 **best structure**。

然后，对于每个 backbone：

- 对该 backbone 的 `top3` 候选做平均：
  - `best_plddt`
  - `best_ptm`
  - `proagg_score`
  - `deltaG_minus_wt`
  - `mpnn_logprob`

这就是当前最终使用的分析口径。

### backbone-level mean-of-top3 结果

#### `proagg_score`

- best run: `0.4320`
- baseline: `0.2470`
- 平均差值：`+0.1851`
- `98.7%` backbone 上 best run 更高

#### `deltaG_minus_wt`

- best run: `1.0131`
- baseline: `0.8352`
- 平均差值：`+0.1779`
- `63.0%` backbone 上 best run 更高

#### `mpnn_logprob`

- best run: `-1.1373`
- baseline: `-1.2775`
- 平均差值：`+0.1402`
- `97.8%` backbone 上 best run 更好

#### `best_plddt`

- best run: `0.8711`
- baseline: `0.8642`
- 平均差值：`+0.0069`
- `57.2%` backbone 上 best run 更高

#### `best_ptm`

- best run: `0.7886`
- baseline: `0.7836`
- 平均差值：`+0.0049`
- `51.5%` backbone 上 best run 更高

### 结构阈值通过率

这里的通过率同样是：

- 先对每条候选判断是否通过阈值；
- 再在每个 backbone 的 top3 上取平均；
- 最后对 backbone 求平均。

#### 阈值：`pLDDT >= 0.80` 且 `pTM >= 0.70`

- 结构平均通过率：
  - best run: `0.8449`
  - baseline: `0.8222`
- 联合平均通过率：
  - 条件 = `结构过线` + `proagg_score > 0` + `deltaG_minus_wt > 0`
  - best run: `0.7417`
  - baseline: `0.6022`

#### 阈值：`pLDDT >= 0.85` 且 `pTM >= 0.80`

- 结构平均通过率：
  - best run: `0.6501`
  - baseline: `0.6195`
- 联合平均通过率：
  - best run: `0.5723`
  - baseline: `0.4593`

### 结构结果的解释

这说明当前最佳模型不是靠少数极端样本取得优势。

在 backbone-level 的 top3 平均意义下，它同时提升了：

- aggregation
- stability
- ProteinMPNN prior compatibility
- 结构可信度

这是当前项目中最强的一条证据链。

### 最终严格联合成功定义

最终正式报告采用更严格的联合成功定义：

- 候选 `proagg_score >` 野生型 aggregation 标签
- 候选 `deltaG > WT deltaG`
- `best_plddt > 0.85`
- `best_ptm > 0.8`

其中野生型 aggregation 标签直接来自：

- `data/rocklin/rawdata/data.csv`
- 列：`log2_fold_change_75_clip`

该分析使用脚本：

- `scripts/analyze_joint_structure_vs_wt.py`

输出目录：

- `results/joint_structure_vs_wt_analysis_label75`

结果：

- backbone-level mean-of-top3 严格联合成功率
  - best run: `0.5533`
  - baseline: `0.4523`
  - 差值：`+0.1010`

这是 pure DPO 主线下最重要的最终指标。

### 引入 SFT 与混合损失后的最终方法比较

在 pure DPO 之外，我们又完成了以下三条线的完整结构验证：

- pure SFT
- DPO+SFT，`w_sft = 0.1`
- DPO+SFT，`w_sft = 1.0`

所有方法都采用同一个最终严格标准：

- `proagg_score > WT log2_fold_change_75_clip`
- `deltaG > WT deltaG`
- `best_plddt > 0.85`
- `best_ptm > 0.80`

按 backbone-level mean-of-top3 统计，最终结果如下：

| 方法 | `proagg_score` | `deltaG_minus_wt` | `mpnn_logprob` | `best_plddt` | `best_ptm` | strict joint success |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ProteinMPNN | `0.2470` | `0.8352` | `-1.2775` | `0.8642` | `0.7836` | `0.4523` |
| Pure DPO | `0.4320` | `1.0131` | `-1.1373` | `0.8711` | `0.7886` | `0.5533` |
| Pure SFT | `0.3741` | `1.1687` | `-0.8371` | `0.8801` | `0.8040` | `0.5891` |
| DPO+SFT `0.1` | `0.4089` | `1.1760` | `-0.8837` | `0.8776` | `0.7998` | `0.5896` |
| DPO+SFT `1.0` | `0.3778` | `1.1761` | `-0.8418` | `0.8796` | `0.8025` | `0.5921` |

当前解释是：

- pure DPO 在 aggregation 上最激进，但 stability 和最终结构联合成功不如其他方法；
- pure SFT 是一个非常强的基线，明显优于 pure DPO；
- DPO+SFT 能在保留 SFT 优势的同时，引入 preference learning 的收益；
- 当前最终最优方法是 **`w_sft = 1.0` 的 DPO+SFT**，其 strict joint success 为 `0.5921`。

---

## 14. 当前最终使用的关键目录和文件

### 训练与模型选择

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/dual_eval_report.txt`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dual_eval_report.txt`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/valid_epoch_selection/best_checkpoint.json`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/valid_epoch_selection/best_checkpoint.json`

### baseline cache

- `results/cache/thbal_valid_original_n16_t05_seed42.json`
- `results/cache/test_original_n16_t05_seed42.json`

### 候选导出

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv`
- `results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv`

### joint top3

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.fasta`
- `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv`
- `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.fasta`

### Chai 结构结果

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint`
- `results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint`

### 最终严格联合分析

- `results/joint_structure_vs_wt_analysis_label75/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75/backbone_mean_of_top3_vs_wt.csv`
- `results/joint_structure_vs_wt_analysis_label75_sft/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75_dpo_sft01/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75_dpo_sft10/joint_structure_vs_wt_report.json`

### 核心脚本

- `src/dpo/sample_and_score_joint.py`
- `src/dpo/evaluate.py`
- `scripts/run_dpo_joint_semi_online_pipeline.sh`
- `scripts/select_best_joint_checkpoint.py`
- `scripts/analyze_dual_eval.py`
- `scripts/export_dpo_test_candidates.py`
- `scripts/select_joint_topk_candidates.py`
- `scripts/run_chai_batch_from_candidates.py`

---

## 15. 当前结论

当前最佳整体系统是：

- **semi-online hybrid DPO+SFT**
- **两轮 disjoint train halves**
- **每轮只训练 1 个 epoch**
- **validation-based checkpoint selection**
- **混合权重 `w_sft = 1.0`**

当前最佳 checkpoint 是：

- `results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/round1/dpo_sft/mpnn_dpo_epoch1.pt`

这个系统目前实现了：

- 相比 baseline 大幅更高的 strict joint success；
- 相比 pure DPO 更好的最终联合结果；
- 相比 pure SFT 也略高的最终 strict joint success；
- 当前项目里最好的 end-to-end 结构联合成功结果。

因此，从训练方法上看，当前主线已经比较成熟。后续工作重点应转向：

- 图表整理
- 论文主表
- 补充材料
- 最终写作

---

## 16. 补充消融实验与结果

在得到初始的 DPO 和 SFT baseline 后，我们进一步做了几组补充实验，用来判断：

- 纯 DPO 是否因为训练 epoch 太少而没有充分收敛；
- 在 DPO 中加入 winner-only SFT 辅助损失，是否能同时获得 DPO 的 aggregation 优势和 SFT 的稳定性/结构优势。

### 实验动机

当前纯 DPO 的训练损失只包含 DPO preference loss，没有 SFT 辅助项。

同时，SFT baseline 的结果很强，尤其体现在：

- stability；
- ProteinMPNN log-probability；
- Chai 结构指标。

因此，一个自然的后续实验是测试混合目标：

```text
loss = DPO loss + w_sft * winner-only SFT loss
```

为了不破坏已有代码，原始 DPO 训练代码保持不变，混合损失单独实现。

### 新增 DPO+SFT 代码

- 混合训练器：
  - `src/dpo/dpo_sft_train.py`
- 混合 semi-online pipeline：
  - `scripts/run_dpo_sft_joint_semi_online_pipeline.sh`

混合损失定义为：

```text
L = L_DPO + DPO_SFT_LOSS_WEIGHT * NLL(seq_winner | backbone)
```

### DPO+SFT 权重扫描

三组实验都沿用当前最佳 DPO 的设置：

- `ROUNDS=2`
- `ROUND_EPOCHS=1`
- `ROUND_TRAIN_SPLIT_MODE=halves`
- `STABILITY_GATE_MODE=wt_absolute`
- `STABILITY_GATE_MARGIN=0.5`
- `AGG_SCORE_GAP_DELTA=0.20`
- `STAB_SCORE_GAP_DELTA=0.20`
- validation selection 使用 `threshold_balance_ltneg1_eq/valid`
- final test 使用 `representative_pdbs/test`

唯一变化是 SFT 辅助损失权重：

- `DPO+0.1*SFT`
  - `results/semi_joint_dpo_sft01_thbalfull_halves_r2_e1_valselect_m05`
- `DPO+0.5*SFT`
  - `results/semi_joint_dpo_sft05_thbalfull_halves_r2_e1_valselect_m05`
- `DPO+1.0*SFT`
  - `results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05`

### 纯 DPO 延长训练实验

另外还启动了一组纯 DPO 的延长训练对照，用于测试当前每轮 1 个 epoch 是否训练不足：

- `ROUNDS=2`
- `ROUND_EPOCHS=2`
- 其他设置与当前最佳 DPO 完全一致

结果目录：

- `results/semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05`

启动脚本：

- `scripts/run_dpo_e2_experiment.sh`

### 实验结果

`round1/dual_eval_report.txt` 的 property-level 对比如下：

| 方法 | `delta_proagg_mean` | `delta_proagg_max` | `delta_deltaG_mean` | `delta_deltaG_max` | `delta_logprob_mean` |
|---|---:|---:|---:|---:|---:|
| Pure DPO `r2/e1` | `0.5038` | `0.1413` | `0.0691` | `0.0293` | `0.1472` |
| Pure SFT `r2/e1` | `0.4175` | `0.0864` | `0.2774` | `0.0706` | `0.4510` |
| DPO+SFT `0.1` | `0.4903` | `0.1153` | `0.2594` | `0.0818` | `0.4064` |
| DPO+SFT `0.5` | `0.4406` | `0.0946` | `0.2731` | `0.0668` | `0.4376` |
| DPO+SFT `1.0` | `0.4268` | `0.0897` | `0.2829` | `0.0750` | `0.4466` |
| Pure DPO `r2/e2` | `0.5800` | `0.1546` | `-0.1365` | `-0.1418` | `0.3273` |

主要结论：

- 纯 DPO 延长训练到 `r2/e2` 不值得继续，aggregation 虽然更高，但 stability 又重新变负；
- DPO+SFT 明显比 pure DPO 更稳；
- 随着 `w_sft` 增大，模型逐渐向更好的 stability / prior 保持方向移动；
- 最终结构联合成功率表明，`w_sft = 1.0` 是当前最佳混合权重。

### 比较流程

这些实验的比较流程是：

- `round1/dual_eval_report.txt`
- `round1/valid_epoch_selection/best_checkpoint.json`

优先比较字段：

- `delta_proagg_mean`
- `delta_proagg_max`
- `delta_deltaG_mean`
- `delta_deltaG_max`
- `delta_logprob_mean`

只有 property-level 足够有竞争力的 run，才会继续进入：

1. full-test candidates 导出；
2. joint top3 筛选；
3. Chai-1 结构验证；
4. strict joint success 分析。
