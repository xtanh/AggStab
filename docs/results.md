# Results Summary

## Scope

This document summarizes the current end-to-end results for:

- single-objective aggregation-aware DPO exploration,
- dual-objective / joint-preference exploration,
- semi-online joint DPO,
- candidate export,
- Chai-1 structure validation,
- the directories and files used in the final analysis.

The current recommended model is the **semi-online joint DPO** run:

- run directory: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`
- selected checkpoint: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dpo/mpnn_dpo_epoch1.pt`

---

## 1. Problem setup

The project goal is to improve, under a fixed-backbone design setting:

- aggregation-related property score (`ProAgg`)
- predicted stability (`deltaG`)

while preserving:

- ProteinMPNN designability prior (`mpnn_logprob`)
- downstream structure confidence from Chai-1 (`pLDDT`, `pTM`)

The main difficulty observed throughout the project is a real trade-off:

- pushing aggregation too hard can hurt stability,
- and naive dual-objective training can still drift away from the ProteinMPNN prior.

---

## 2. Property models and backbone-conditioned assumption

All current property optimization experiments use **SaProt-based** predictors.

### Aggregation predictor

- checkpoint: `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- config: `configs/proagg_final_candidate.yaml`

### Stability predictor

- checkpoint: `results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt`
- config: `configs/proagg_deltaG_only.yaml`

### Important modeling assumption

For both predictors, a designed sequence is scored using the original backbone structural tokens.

That means the property prediction is:

- **fixed-backbone conditional prediction**, not free-folding prediction.

This is acceptable for inverse folding, but final structure validation is still required.

---

## 3. Data splits used in final training

### Final train/validation/test splits

- train: `data/dpo/subsets/threshold_balance_ltneg1_eq/train`
- validation for pair construction: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- validation for checkpoint selection: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- final test: `data/dpo/representative_pdbs/test`

### Why this setup

- train/valid use the same threshold-balanced distribution, because this subset was explicitly designed to emphasize backbones relevant to the aggregation objective.
- final test stays on the representative full test set to report generalization.

---

## 4. Evolution of the DPO experiments

### 4.1 Single-objective aggregation DPO

Initial DPO experiments focused only on aggregation.

Main observations:

- aggregation improved,
- but stability often dropped,
- and validation loss did not reliably predict downstream design quality.

Relevant run:

- `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12_fulltest`

This line established that:

- threshold-balanced data helps,
- static offline DPO can work for aggregation,
- but the aggregation–stability trade-off remains.

### 4.2 Dual-objective mixing

We then implemented:

- `src/dpo/sample_and_score_dual.py`
- `scripts/run_dpo_dual_pipeline.sh`

This version built:

- aggregation pairs,
- stability pairs,
- and mixed them during training.

Main conclusion:

- dual mixing was not strong enough,
- stability was often overwhelmed by aggregation,
- especially at scale.

### 4.3 Static joint-preference DPO

We then switched from dual mixing to **joint pairs**:

- `src/dpo/sample_and_score_joint.py`
- `scripts/run_dpo_joint_pipeline.sh`

Joint-pair rule:

- winner must beat loser on both `ProAgg` and `deltaG`,
- after a stability gate.

This was better than dual mixing, but full-scale static training still drifted.

### 4.4 Semi-online joint DPO

The final successful direction was:

- `scripts/run_dpo_joint_semi_online_pipeline.sh`

Core idea:

- regenerate joint preference pairs from the **current policy**,
- train only briefly per round,
- use validation-based downstream checkpoint selection.

This reduced the mismatch between:

- the current policy distribution,
- and the fixed offline preference pairs.

---

## 5. Final best training recipe

The current best recipe is:

- **semi-online joint DPO**
- `2` rounds
- `1` epoch per round
- train set split into **two disjoint random halves**
- one half used in round 0
- the other half used in round 1
- validation-based checkpoint selection after each round

### Exact hyperparameters

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

### Why the halves split mattered

Using the full train pool in every round was too aggressive.

The disjoint-halves strategy reduced drift by:

- lowering per-round update pressure,
- still covering the full threshold-balanced train pool across rounds,
- and keeping computation manageable.

---

## 6. Validation-based checkpoint selection

We do **not** use test results for model selection.

We also do **not** trust training `val_loss` alone.

Instead, each saved checkpoint is evaluated on the threshold-balanced validation set using downstream property metrics, and the best checkpoint is selected there.

### Selector

- script: `scripts/select_best_joint_checkpoint.py`

### Tiered selection rule

#### Tier A

Require all of the following on validation:

- `delta_proagg_mean >= 0`
- `delta_proagg_max >= 0`
- `delta_deltaG_mean >= 0`
- `delta_deltaG_max >= 0`

Then rank by:

- `joint_sum = delta_proagg_mean + delta_deltaG_mean`

#### Tier B

If Tier A is empty, require:

- `delta_proagg_mean >= 0`
- `delta_deltaG_mean >= 0`

Then rank by `joint_sum`.

#### Tier C

If Tier B is empty, use all epochs and rank by:

- `joint_sum - 0.1 * (# negative max metrics)`

### Validation selection records for the best run

- round 0:
  - `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/valid_epoch_selection/best_checkpoint.json`
- round 1:
  - `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/valid_epoch_selection/best_checkpoint.json`

Both selected checkpoints are **Tier A**.

---

## 7. Baseline cache reuse

To avoid recomputing baseline ProteinMPNN valid/test results every time:

- `src/dpo/evaluate.py` supports `--original_results_cache`

### Validation baseline cache

- `results/cache/thbal_valid_original_n16_t05_seed42.json`

### Test baseline cache

- `results/cache/test_original_n16_t05_seed42.json`

These caches are reused whenever:

- the backbone set,
- `num_samples`,
- `temperature`,
- and `seed`

are unchanged.

---

## 8. Final test results for the best run

### Run directory

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`

### Round 0 test report

File:

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/dual_eval_report.txt`

Metrics:

- `delta_proagg_mean = +0.4050`
- `delta_proagg_max = +0.1137`
- `delta_deltaG_mean = +0.1007`
- `delta_deltaG_max = +0.0735`
- `delta_logprob_mean = +0.1032`
- `improve_proagg_mean = 1346 / 1350`
- `improve_proagg_max = 1261 / 1350`
- `improve_deltaG_mean = 974 / 1350`
- `improve_deltaG_max = 893 / 1350`

### Round 1 test report

File:

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dual_eval_report.txt`

Metrics:

- `delta_proagg_mean = +0.5038`
- `delta_proagg_max = +0.1413`
- `delta_deltaG_mean = +0.0691`
- `delta_deltaG_max = +0.0293`
- `delta_logprob_mean = +0.1472`
- `improve_proagg_mean = 1348 / 1350`
- `improve_proagg_max = 1298 / 1350`
- `improve_deltaG_mean = 876 / 1350`
- `improve_deltaG_max = 825 / 1350`

### Interpretation

Round 1 is the preferred checkpoint because it offers:

- stronger aggregation improvement,
- still positive stability improvement,
- better ProteinMPNN log-probability,
- and strong validation support.

---

## 9. Candidate export for final comparison

### Best run candidates

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv`

### Baseline candidates

- `results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv`

These tables were exported with:

- `scripts/export_dpo_test_candidates.py`

Each row contains:

- `pdb_name`
- `candidate_rank`
- `sequence`
- `proagg_score`
- `deltaG`
- `wt_deltaG`
- `deltaG_minus_wt`
- `mpnn_logprob`

Note:

- `5` backbones are missing `wt_deltaG`, so their `deltaG_minus_wt` is `NaN`.

### Candidate-level summary

#### Best run

- all candidates:
  - `proagg_score = 0.3097`
  - `deltaG = 3.6180`
  - `deltaG_minus_wt = 0.9224`
  - `mpnn_logprob = -1.1497`
- rank 1 candidate per backbone:
  - `proagg_score = 0.4675`
  - `deltaG_minus_wt = 0.8761`
  - `mpnn_logprob = -1.1362`

#### Baseline

- all candidates:
  - `proagg_score = -0.1942`
  - `deltaG = 3.5489`
  - `deltaG_minus_wt = 0.8528`
  - `mpnn_logprob = -1.2969`
- rank 1 candidate per backbone:
  - `proagg_score = 0.3262`
  - `deltaG_minus_wt = 0.7219`
  - `mpnn_logprob = -1.2731`

### Candidate-pool interpretation

The final model improves the entire candidate pool, not just a few extreme sequences:

- higher `ProAgg`
- higher `deltaG_minus_wt`
- better `mpnn_logprob`

---

## 10. Joint top-3 selection before structure prediction

We then selected `top3` candidates per backbone for structure validation.

### Selection script

- `scripts/select_joint_topk_candidates.py`

### Selection stages

For each backbone:

1. `strict_joint`
   - pathology-safe
   - `proagg_score > 0`
   - `deltaG_minus_wt > 0`
2. `strict_proagg`
   - pathology-safe
   - `proagg_score > 0`
3. `pathology_only`
4. `no_filter`

### Selected files

#### Best run

- csv: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv`
- fasta: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.fasta`

#### Baseline

- csv: `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv`
- fasta: `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.fasta`

### Top-3 stage usage

#### Best run

- `strict_joint: 1149` backbones
- `strict_proagg: 198`
- `pathology_only: 3`

#### Baseline

- `strict_joint: 874` backbones
- `strict_proagg: 268`
- `pathology_only: 208`

### Top-3 summary

#### Best run top3

- `proagg_score = 0.4320`
- `deltaG_minus_wt = 1.0131`
- `mpnn_logprob = -1.1373`
- backbones with at least one joint-positive top3 candidate:
  - `1171 / 1350`

#### Baseline top3

- `proagg_score = 0.2470`
- `deltaG_minus_wt = 0.8352`
- `mpnn_logprob = -1.2775`
- backbones with at least one joint-positive top3 candidate:
  - `1049 / 1350`

Interpretation:

- the final model provides a cleaner top3 candidate set,
- with more backbones already satisfying the joint property objective before any structure filtering.

---

## 11. Chai-1 structure validation

### Chai runner

- `scripts/run_chai_batch_from_candidates.py`

### Chai output directories

#### Best run

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint`

#### Baseline

- `results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint`

### Chai output files used

For both models:

- `summary_shard0.csv`
- `summary_shard1.csv`
- `summary_shard2.csv`
- `summary_shard3.csv`
- `stats_shard0.json`
- `stats_shard1.json`
- `stats_shard2.json`
- `stats_shard3.json`

All `4050 / 4050` candidate structures completed successfully for both models.

---

## 12. Final structure-aware analysis

### Important analysis convention

For each candidate sequence:

- Chai produces multiple structure samples.
- We first select the **best structure for that candidate**.

Then, for each backbone:

- we take the **mean over the top3 candidates** for:
  - `best_plddt`
  - `best_ptm`
  - `proagg_score`
  - `deltaG_minus_wt`
  - `mpnn_logprob`

This is the final interpretation used for the current results.

### Backbone-level mean-of-top3 comparison

#### `proagg_score`

- best run: `0.4320`
- baseline: `0.2470`
- mean delta: `+0.1851`
- best run better on `98.7%` of backbones

#### `deltaG_minus_wt`

- best run: `1.0131`
- baseline: `0.8352`
- mean delta: `+0.1779`
- best run better on `63.0%` of backbones

#### `mpnn_logprob`

- best run: `-1.1373`
- baseline: `-1.2775`
- mean delta: `+0.1402`
- best run better on `97.8%` of backbones

#### `best_plddt`

- best run: `0.8711`
- baseline: `0.8642`
- mean delta: `+0.0069`
- best run better on `57.2%` of backbones

#### `best_ptm`

- best run: `0.7886`
- baseline: `0.7836`
- mean delta: `+0.0049`
- best run better on `51.5%` of backbones

### Structure threshold pass rates

These rates are averaged across the `top3` candidates per backbone.

#### Threshold: `pLDDT >= 0.80` and `pTM >= 0.70`

- structure pass rate:
  - best run: `0.8449`
  - baseline: `0.8222`
- joint pass rate:
  - condition = structure pass + `proagg_score > 0` + `deltaG_minus_wt > 0`
  - best run: `0.7417`
  - baseline: `0.6022`

#### Threshold: `pLDDT >= 0.85` and `pTM >= 0.80`

- structure pass rate:
  - best run: `0.6501`
  - baseline: `0.6195`
- joint pass rate:
  - best run: `0.5723`
  - baseline: `0.4593`

### Main interpretation

The final model is not just producing a few stronger outliers.

At the backbone-level mean-of-top3 level, it improves:

- aggregation,
- stability,
- ProteinMPNN prior compatibility,
- and structure confidence.

This is the strongest evidence obtained so far.

### Final strict joint success definition

For the final report, we adopt a stricter joint success definition:

- candidate `proagg_score >` WT aggregation label
- candidate `deltaG > WT deltaG`
- `best_plddt > 0.85`
- `best_ptm > 0.8`

Here WT aggregation is taken directly from:

- `data/rocklin/rawdata/data.csv`
- column: `log2_fold_change_75_clip`

This analysis is computed by:

- `scripts/analyze_joint_structure_vs_wt.py`

Output directory:

- `results/joint_structure_vs_wt_analysis_label75`

Result:

- backbone-level mean-of-top3 strict joint success rate
  - best run: `0.5533`
  - baseline: `0.4523`
  - delta: `+0.1010`

This was the initial headline metric for the pure-DPO line.

### Final method comparison after SFT and hybrid runs

We subsequently completed full structure-aware evaluation for:

- pure DPO
- pure SFT
- hybrid DPO+SFT with `w_sft = 0.1`
- hybrid DPO+SFT with `w_sft = 1.0`

All methods are evaluated with the same final strict criterion:

- `proagg_score > WT log2_fold_change_75_clip`
- `deltaG > WT deltaG`
- `best_plddt > 0.85`
- `best_ptm > 0.80`

Backbone-level mean-of-top3 results:

| Method | `proagg_score` | `deltaG_minus_wt` | `mpnn_logprob` | `best_plddt` | `best_ptm` | strict joint success |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ProteinMPNN | `0.2470` | `0.8352` | `-1.2775` | `0.8642` | `0.7836` | `0.4523` |
| Pure DPO | `0.4320` | `1.0131` | `-1.1373` | `0.8711` | `0.7886` | `0.5533` |
| Pure SFT | `0.3741` | `1.1687` | `-0.8371` | `0.8801` | `0.8040` | `0.5891` |
| DPO+SFT `0.1` | `0.4089` | `1.1760` | `-0.8837` | `0.8776` | `0.7998` | `0.5896` |
| DPO+SFT `1.0` | `0.3778` | `1.1761` | `-0.8418` | `0.8796` | `0.8025` | `0.5921` |

Current interpretation:

- pure DPO gives the strongest aggregation gain, but is weaker on stability and final structure-aware joint success;
- pure SFT is a very strong baseline and substantially improves over pure DPO;
- hybrid DPO+SFT retains the advantages of SFT while recovering some preference-alignment benefit;
- the current best final result is **DPO+SFT with `w_sft = 1.0`**, with strict joint success `0.5921`.

---

## 13. Key directories and files used in the final analysis

### Training and model-selection

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/dual_eval_report.txt`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dual_eval_report.txt`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round0/valid_epoch_selection/best_checkpoint.json`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/valid_epoch_selection/best_checkpoint.json`

### Baseline caches

- `results/cache/thbal_valid_original_n16_t05_seed42.json`
- `results/cache/test_original_n16_t05_seed42.json`

### Exported candidates

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv`
- `results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv`

### Top-3 candidate sets

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv`
- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.fasta`
- `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv`
- `results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.fasta`

### Chai structure outputs

- `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint`
- `results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint`

### Final strict joint analysis

- `results/joint_structure_vs_wt_analysis_label75/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75/backbone_mean_of_top3_vs_wt.csv`
- `results/joint_structure_vs_wt_analysis_label75_sft/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75_dpo_sft01/joint_structure_vs_wt_report.json`
- `results/joint_structure_vs_wt_analysis_label75_dpo_sft10/joint_structure_vs_wt_report.json`

### Main scripts used

- `src/dpo/sample_and_score_joint.py`
- `src/dpo/evaluate.py`
- `scripts/run_dpo_joint_semi_online_pipeline.sh`
- `scripts/select_best_joint_checkpoint.py`
- `scripts/analyze_dual_eval.py`
- `scripts/export_dpo_test_candidates.py`
- `scripts/select_joint_topk_candidates.py`
- `scripts/run_chai_batch_from_candidates.py`

---

## 14. Current conclusion

The current best overall system is:

- **semi-online hybrid DPO+SFT**
- **disjoint train halves across two rounds**
- **one epoch per round**
- **validation-based checkpoint selection**
- **hybrid weight `w_sft = 1.0`**

The current best checkpoint is:

- `results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/round1/dpo_sft/mpnn_dpo_epoch1.pt`

This system delivers:

- substantially better strict joint success than baseline,
- better final strict joint success than pure DPO,
- slightly better final strict joint success than pure SFT,
- and the best end-to-end structure-aware score currently obtained in the project.

At this point, the training story is sufficiently mature. The remaining work should focus on:

- figure generation,
- paper tables,
- supplementary analysis,
- and final writing.

---

## 15. Additional ablation experiments and outcomes

After obtaining the initial DPO and SFT baselines, we launched additional ablations to test whether either
longer DPO training or adding an auxiliary winner-only SFT term could improve over the original pure-DPO result.

### Motivation

- Current pure DPO uses only the DPO preference loss.
- The SFT baseline is strong, especially for stability, ProteinMPNN log-probability, and Chai structural metrics.
- A natural follow-up is therefore to test a hybrid objective:

```text
loss = DPO loss + w_sft * winner-only SFT loss
```

The original DPO code is kept unchanged. The hybrid loss is implemented separately.

### New code for hybrid DPO+SFT

- Hybrid trainer:
  - `src/dpo/dpo_sft_train.py`
- Hybrid semi-online pipeline:
  - `scripts/run_dpo_sft_joint_semi_online_pipeline.sh`

The hybrid loss is:

```text
L = L_DPO + DPO_SFT_LOSS_WEIGHT * NLL(seq_winner | backbone)
```

### Hybrid sweeps

All three runs use the same setup as the current best DPO recipe:

- `ROUNDS=2`
- `ROUND_EPOCHS=1`
- `ROUND_TRAIN_SPLIT_MODE=halves`
- `STABILITY_GATE_MODE=wt_absolute`
- `STABILITY_GATE_MARGIN=0.5`
- `AGG_SCORE_GAP_DELTA=0.20`
- `STAB_SCORE_GAP_DELTA=0.20`
- validation selection on `threshold_balance_ltneg1_eq/valid`
- final test on `representative_pdbs/test`

Only the SFT auxiliary weight changes:

- `DPO+0.1*SFT`
  - `results/semi_joint_dpo_sft01_thbalfull_halves_r2_e1_valselect_m05`
- `DPO+0.5*SFT`
  - `results/semi_joint_dpo_sft05_thbalfull_halves_r2_e1_valselect_m05`
- `DPO+1.0*SFT`
  - `results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05`

### Longer pure-DPO ablation

We also launched a longer pure-DPO control to test whether the current one-epoch-per-round setting
is undertrained:

- `ROUNDS=2`
- `ROUND_EPOCHS=2`
- all other settings identical to the current best DPO recipe

Run directory:

- `results/semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05`

Launch script:

- `scripts/run_dpo_e2_experiment.sh`

### Observed outcomes

Property-level comparison on `round1/dual_eval_report.txt`:

| Method | `delta_proagg_mean` | `delta_proagg_max` | `delta_deltaG_mean` | `delta_deltaG_max` | `delta_logprob_mean` |
|---|---:|---:|---:|---:|---:|
| Pure DPO `r2/e1` | `0.5038` | `0.1413` | `0.0691` | `0.0293` | `0.1472` |
| Pure SFT `r2/e1` | `0.4175` | `0.0864` | `0.2774` | `0.0706` | `0.4510` |
| DPO+SFT `0.1` | `0.4903` | `0.1153` | `0.2594` | `0.0818` | `0.4064` |
| DPO+SFT `0.5` | `0.4406` | `0.0946` | `0.2731` | `0.0668` | `0.4376` |
| DPO+SFT `1.0` | `0.4268` | `0.0897` | `0.2829` | `0.0750` | `0.4466` |
| Pure DPO `r2/e2` | `0.5800` | `0.1546` | `-0.1365` | `-0.1418` | `0.3273` |

Main takeaways:

- longer pure DPO (`r2/e2`) is not useful: aggregation rises, but stability turns negative again;
- hybrid DPO+SFT is consistently more stable than pure DPO;
- increasing the SFT weight shifts the model toward stronger stability and prior preservation;
- final structure-aware evaluation shows that `w_sft = 1.0` gives the best strict joint success.

### Comparison workflow

The comparison logic used for these ablations was:

- `round1/dual_eval_report.txt`
- `round1/valid_epoch_selection/best_checkpoint.json`

Primary fields to compare:

- `delta_proagg_mean`
- `delta_proagg_max`
- `delta_deltaG_mean`
- `delta_deltaG_max`
- `delta_logprob_mean`

Only runs competitive at the property level were promoted to:

1. full-test candidate export,
2. joint top3 candidate selection,
3. Chai-1 structure validation,
4. strict joint success analysis.
