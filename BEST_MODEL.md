# ProAgg Model Records

## Current Records

### Strongest Current Candidate

- Model: `proagg_mlp_v36`
- Experiment version: `version_92`
- Checkpoint:
  `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- Validation metrics:
  - `val_spearman = 0.764900`
- Test metrics:
  - `test_spearman = 0.756144`
  - `test_pearson = 0.746187`
  - `test_loss = 0.554575`
- Bootstrap test Spearman:
  - `0.755460 ± 0.020193`
  - 95% CI: `[0.734387, 0.774773]`
- Bootstrap delta vs `version_82`:
  - `+0.000757 ± 0.000353`
  - 95% CI: `[0.000413, 0.001118]`

Why this is the current strongest candidate:

- It improves on `version_82` after a controlled hyperparameter search.
- It is currently the strongest single-model result in the repository.
- Under the current bootstrap comparison, its test Spearman improvement over `version_82` does not cross zero.

### Best Validation Score

- Model: `proagg_mlp_v9`
- Experiment version: `version_21`
- Checkpoint:
  `results/lightning_logs/version_21/checkpoints/best_epoch=08_val_spearman=0.7646.ckpt`
- Validation metrics:
  - `val_spearman = 0.764566`
  - `val_pearson = 0.764760`
  - `val_mae = 0.488606`
  - `val_mse = 0.496259`

Why this is recorded separately:

- It is the highest observed validation checkpoint by `val_spearman`.
- But it belongs to the contrastive-learning branch.
- Its final `test_spearman` is only `0.743134`, so it is not a reliable final-model choice.

### Best Ranking-Line Validation Baseline

- Model: `proagg_mlp_v13`
- Experiment version: `version_25`
- Checkpoint:
  `results/lightning_logs/version_25/checkpoints/best_epoch=08_val_spearman=0.7613.ckpt`
- Validation metrics:
  - `val_spearman = 0.761268`
  - `val_pearson = 0.761313`
  - `val_mae = 0.506163`
  - `val_mse = 0.505429`

Why this is recorded separately:

- It is the cleanest validation winner inside the ranking-centered branch.
- It remains the main validation reference for the later ranking experiments.

### Best Observed Test Result

- Model: `proagg_mlp_v36`
- Experiment version: `version_92`
- Checkpoint:
  `results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt`
- Test metrics:
  - `test_spearman = 0.756144`
  - `test_pearson = 0.746187`

Why this is recorded separately:

- It is the best observed test result so far.
- It is also a strong validation-side candidate, unlike earlier test-only winners.
- It should now be treated as the main strongest candidate until a cleaner counterexample appears.

## Important Method Note

The project currently has two different notions of “best”:

- `version_21`: best raw validation score
- `version_25`: best ranking-line validation baseline
- `version_92`: strongest current candidate and best observed test result

This mismatch still matters methodologically, but `version_92` is the current best candidate that reduces the gap between the validation-side and test-side stories.
Future experiments should still be judged first by validation-side criteria, then confirmed by bootstrap test comparison.

So the repository already contains a concrete example showing that single-metric validation ranking can be misleading: `version_21`.

See:

- `docs/VAL_ERROR_ANALYSIS.md`

## Current Best-Supported Training Recipe

The strongest experimentally supported direction so far is now:

- `proagg_mlp_v36`
- `aggregation_75` main head
- `deltaG` auxiliary regression head
- ranking loss enabled on the aggregation head
- `ranking_weight = 1.2`
- `ranking_margin = 0.25`
- `deltaG_weight = 0.1`
- `deltaG_ci_weighting = true`
- regularization via `weight_decay = 0.2`

In practice, the most important useful components have been:

1. attention pooling
2. ranking loss
3. stability-aware auxiliary supervision through `deltaG`
4. moderate regularization

## Current Interpretation

- Architecture changes beyond the `v13` line still have weak evidence.
- The first strong clean gain beyond the ranking line came from adding `deltaG` as an auxiliary target.
- The hardest unresolved cases are still the strongly aggregation-prone negative-tail samples.

## Recommended Next-Step Standard

Before declaring a new candidate as the main line, compare it on validation with:

- `val_spearman`
- `val_pearson`
- `val_mae`
- validation large-gap pair error rate
- validation tail MAE at `<= -3.0`
- validation tail MAE at `<= -3.5`
