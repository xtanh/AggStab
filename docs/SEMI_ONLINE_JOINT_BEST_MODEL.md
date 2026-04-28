# Semi-Online Joint DPO: Current Best Configuration

## Best run

- Run directory: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05`
- Selected checkpoint: `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/dpo/mpnn_dpo_epoch1.pt`
- Validation-selected checkpoint record:
  - `results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/round1/valid_epoch_selection/best_checkpoint.json`

## Final test result

- Test set: `data/dpo/representative_pdbs/test` (`1350` backbones)
- `delta_proagg_mean = +0.5038`
- `delta_proagg_max = +0.1413`
- `delta_deltaG_mean = +0.0691`
- `delta_deltaG_max = +0.0293`
- `delta_logprob_mean = +0.1472`
- `improve_proagg_mean = 1348 / 1350`
- `improve_proagg_max = 1298 / 1350`
- `improve_deltaG_mean = 876 / 1350`
- `improve_deltaG_max = 825 / 1350`

## Core idea

The current best recipe is a pragmatic compromise that fixed three failure modes seen in earlier experiments:

1. **Static full-scale training drifted**: full-data static joint or dual-mixing often improved `ProAgg` but hurt `deltaG`.
2. **Validation selection by loss was unreliable**: `val_loss` did not track downstream `ProAgg/deltaG`.
3. **Full train pool per round was too aggressive**: using all train backbones in every round pushed the policy too far.

The working recipe is therefore:

- **Joint preference pairs**
  - winner must beat loser on both `ProAgg` and `deltaG`
- **Semi-online refresh**
  - regenerate pairs each round from the current policy
- **Disjoint train halves**
  - round 0 uses one random half of train backbones
  - round 1 uses the other random half
- **Short training**
  - `2` rounds
  - `1` epoch per round
- **Validation-based checkpoint selection**
  - checkpoint selection uses `threshold_balance_ltneg1_eq/valid`
  - final test is run only after validation selection

## Data splits

- Train: `data/dpo/subsets/threshold_balance_ltneg1_eq/train`
- Validation for pair construction: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- Validation for checkpoint selection: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- Final test: `data/dpo/representative_pdbs/test`

## Property models

- Aggregation predictor:
  - checkpoint: `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
  - config: `configs/proagg_final_candidate.yaml`
- Stability predictor:
  - checkpoint: `results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt`
  - config: `configs/proagg_deltaG_only.yaml`

## Hyperparameters

- `NUM_SAMPLES=16`
- `TEMPERATURE=0.5`
- `DPO_BETA=0.1`
- `ROUNDS=2`
- `ROUND_EPOCHS=1`
- `AGG_SCORE_GAP_DELTA=0.20`
- `STAB_SCORE_GAP_DELTA=0.20`
- `STABILITY_GATE_MODE=wt_absolute`
- `STABILITY_GATE_MARGIN=0.5`
- `VALID_SELECTION_METRIC=joint_sum`
- `VALID_SELECTION_MAX_PENALTY=0.1`

## Baseline caches

- Validation baseline cache:
  - `results/cache/thbal_valid_original_n16_t05_seed42.json`
- Test baseline cache:
  - `results/cache/test_original_n16_t05_seed42.json`

These caches should be reused for future validation/test evaluation under the same:

- backbone set
- `num_samples`
- `temperature`
- `seed`

## Main pipeline files

- Joint pair construction:
  - `src/dpo/sample_and_score_joint.py`
- Evaluation with baseline cache reuse:
  - `src/dpo/evaluate.py`
- Semi-online joint pipeline:
  - `scripts/run_dpo_joint_semi_online_pipeline.sh`
- Validation-based checkpoint selection:
  - `scripts/select_best_joint_checkpoint.py`
- Dual/joint evaluation summary:
  - `scripts/analyze_dual_eval.py`

## Reproducing the best run

```bash
CUDA_VISIBLE_DEVICES=0 \
PDB_TRAIN=data/dpo/subsets/threshold_balance_ltneg1_eq/train \
PDB_VALID=data/dpo/subsets/threshold_balance_ltneg1_eq/valid \
PDB_TEST=data/dpo/representative_pdbs/test \
SELECT_PDB_VALID=data/dpo/subsets/threshold_balance_ltneg1_eq/valid \
MAX_TRAIN_PDBS=-1 MAX_VALID_PDBS=366 MAX_TEST_PDBS=1350 \
SELECT_MAX_VALID_PDBS=366 \
NUM_SAMPLES=16 TEMPERATURE=0.5 SEED=42 \
ROUNDS=2 ROUND_EPOCHS=1 \
ROUND_TRAIN_SPLIT_MODE=halves \
ROUND_TRAIN_SPLIT_SEED=42 \
DPO_BETA=0.1 DPO_BATCH_SIZE=32 DPO_PATIENCE=1 \
AGG_CONFIG=configs/proagg_final_candidate.yaml \
STAB_CONFIG=configs/proagg_deltaG_only.yaml \
STABILITY_CSV=data/rocklin/Metagenomic_dG.csv \
AGG_SCORE_GAP_DELTA=0.20 STAB_SCORE_GAP_DELTA=0.20 \
STABILITY_GATE_MODE=wt_absolute STABILITY_GATE_MARGIN=0.5 \
VALID_SELECTION_ENABLED=1 \
VALID_SELECTION_METRIC=joint_sum \
VALID_SELECTION_MAX_PENALTY=0.1 \
VALID_ORIGINAL_RESULTS_CACHE=results/cache/thbal_valid_original_n16_t05_seed42.json \
ORIGINAL_RESULTS_CACHE=results/cache/test_original_n16_t05_seed42.json \
RUN_TAG=semi_joint_thbalfull_halves_r2_e1_valselect_m05 \
OUTPUT_ROOT=results/semi_joint_thbalfull_halves_r2_e1_valselect_m05 \
bash scripts/run_dpo_joint_semi_online_pipeline.sh \
  results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt \
  results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt \
  cuda:0
```

## Recommended next step

Use this checkpoint for downstream candidate export and structure validation, not more training sweeps.
