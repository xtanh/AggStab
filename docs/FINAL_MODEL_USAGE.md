# Final Model Usage

## Final selected model

- DPO checkpoint: `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12/mpnn_dpo_best.pt`
- Full-test evaluation directory: `results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12_fulltest`
- Aggregation annotator checkpoint: `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- Aggregation annotator config: `configs/proagg_final_candidate.yaml`

## Final training recipe

- Train subset: `data/dpo/subsets/threshold_balance_ltneg1_eq/train`
- Validation subset: `data/dpo/subsets/threshold_balance_ltneg1_eq/valid`
- Test set: `data/dpo/representative_pdbs/test`
- Hyperparameters:
  - rollout size `N=16`
  - sampling temperature `0.5`
  - `beta=0.1`
  - `score_gap_delta=0.10`
  - `epoch=12`

## What this model does

- Starts from ProteinMPNN as the sequence generator
- Uses DPO to shift generated sequences toward lower predicted aggregation
- Uses the `version_92` predictor as the aggregation annotator during preference construction

## Generate sequences for a single backbone

Use `scripts/sample_single_backbone_compare.py` to compare baseline ProteinMPNN and the final DPO model on one backbone.

```bash
python scripts/sample_single_backbone_compare.py \
  --pdb <your_backbone.pdb> \
  --dpo_ckpt results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12/mpnn_dpo_best.pt \
  --output_dir results/<run_name> \
  --num_samples 64 \
  --temperature 0.5 \
  --device cuda:0
```

Outputs:
- `sample_compare.json`
- `baseline_sequences.txt`
- `dpo_sequences.txt`

## Batch evaluation on a backbone set

Main evaluation script:

```bash
python src/dpo/evaluate.py \
  --pdb_dir data/dpo/representative_pdbs/test \
  --dpo_mpnn_ckpt results/dpo_v92_thbal_m1_tr2866_n16_b01_gap010_e12/mpnn_dpo_best.pt \
  --proagg_ckpt results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt \
  --proagg_config configs/proagg_final_candidate.yaml \
  --output results/<eval_dir>/eval_results.json \
  --num_samples 16 \
  --temperature 0.5 \
  --proagg_batch_size 32 \
  --max_pdbs -1 \
  --device cuda:0 \
  --seed 42
```

## Summarize evaluation results

```bash
python scripts/analyze_dpo_eval.py \
  results/<eval_dir>/eval_results.json \
  --output results/<eval_dir>

python scripts/analyze_dpo_v2_pathology.py \
  --eval_json results/<eval_dir>/eval_results.json \
  --summary_csv results/<eval_dir>/dpo_eval_summary.csv
```

Key outputs:
- `dpo_eval_report.txt`
- `dpo_eval_summary.csv`
- `dpo_v2_pathology_enriched.csv`

## Recommended downstream workflow

1. Generate candidate sequences with the final DPO checkpoint.
2. Score candidates with the `version_92` aggregation predictor.
3. Apply additional downstream filters:
   - stability predictor
   - structure prediction / structure consistency
   - low-complexity / pathology checks (`has_run`, composition extremes)
4. Prioritize candidates by:
   - aggregation score
   - MPNN log-probability
   - structural plausibility

## Current final result

This model has already been validated on the full representative test split (`1350` backbones) and is the current global best configuration in the project.
