# Final Candidate: `version_92`

This note records the current main-line model candidate for the project.

## Identity

- Model: `proagg_mlp_v36`
- Experiment version: `version_92`
- Config:
  - `configs/proagg_final_candidate.yaml`
- Checkpoint:
  - `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`

## Why This Is The Current Main Line

`version_92` is the current strongest candidate after controlled single-variable tuning on top of the `version_82` line.

Compared with the previous strongest references:

- vs `version_82`:
  - higher `val_spearman`
  - higher `test_spearman`
  - higher `test_pearson`
  - slightly worse `test_loss`
- vs `version_46`:
  - higher `val_spearman`
  - higher `test_spearman`
  - higher `test_pearson`

## Metrics

### Validation

- `val_spearman = 0.764900`

### Test

- `test_spearman = 0.756144`
- `test_pearson = 0.746187`
- `test_loss = 0.554575`

### Bootstrap Test Spearman

- `0.755460 ± 0.020193`
- 95% CI: `[0.734387, 0.774773]`

### Bootstrap Delta vs `version_82`

- `+0.000757 ± 0.000353`
- 95% CI: `[0.000413, 0.001118]`

Interpretation:

- the improvement over `version_82` is small
- but under the current bootstrap comparison it is positive and does not cross `0`

## Training Recipe

### Main Task

- target: `log2_fold_change_75_clip`
- losses:
  - `MSE`
  - `pairwise ranking`

### Auxiliary Task

- target: `deltaG`
- auxiliary weight: `0.1`
- uncertainty weighting:
  - enabled through `deltaG_95CI`

### Backbone / Optimisation

- backbone: `SaProt`
- LoRA:
  - `r = 4`
  - `alpha = 8`
  - `dropout = 0.1`
- `weight_decay = 0.2`
- `ranking_weight = 1.2`
- `ranking_margin = 0.25`
- `gradient_clip_val = 0.5`
- monitor:
  - `val_spearman`

## What The `deltaG` Head Is Doing

The `deltaG` head is useful as an auxiliary signal, not as a standalone final objective.

Observed `deltaG` prediction quality for the `v36` line at the current best setting:

- train:
  - `Pearson = 0.6941`
  - `Spearman = 0.6491`
- valid:
  - `Pearson = 0.6834`
  - `Spearman = 0.6244`
- test:
  - `Pearson = 0.6618`
  - `Spearman = 0.6261`

Interpretation:

- the head learns a stable folding-stability signal
- that signal is strong enough to help the aggregation task
- but it should still be treated as auxiliary supervision

## Next Hyperparameters To Tune

Future tuning should stay close to this recipe.

Priority order:

1. `deltaG_weight`
   - current best: `0.1`
   - already tested:
     - `0.05` worse
     - `0.2` worse
2. `LoRA` regularization only if capacity is fixed
   - larger `LoRA` capacity (`r=8`, `alpha=16`) was worse
3. checkpoint selection / training strategy
   - this is likely a higher-value remaining lever than reopening old model branches

Do not reopen these directions before the above are exhausted:

- latent/codebook branches
- dual encoder branches
- MoE branches
- `val_loss` checkpoint selection
