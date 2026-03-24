# Validation Model Selection Summary

This note is a short decision-oriented summary for the most important validation candidates so far:

- `version_21`
- `version_25`
- `version_29`
- `version_37`

The purpose is to keep validation-based model selection separate from final test reporting.

## Why This Summary Exists

The project currently has a methodological mismatch:

- `version_21` is the best checkpoint by `val_spearman`
- `version_37` is the best observed test result

So a single “best model” label is not precise enough.
This summary keeps the validation-side view explicit.

## Compared Checkpoints

| version | checkpoint |
|---------|------------|
| `21` | `results/lightning_logs/version_21/checkpoints/best_epoch=08_val_spearman=0.7646.ckpt` |
| `25` | `results/lightning_logs/version_25/checkpoints/best_epoch=08_val_spearman=0.7613.ckpt` |
| `29` | `results/lightning_logs/version_29/checkpoints/best_epoch=08_val_spearman=0.7584.ckpt` |
| `37` | `results/lightning_logs/version_37/checkpoints/best_epoch=08_val_spearman=0.7582.ckpt` |

## Test Caveat for `version_21`

`version_21` belongs to the contrastive-learning branch.
Its validation metrics are strong, but its final test result is much worse than the later ranking-line models:

- `test_spearman = 0.743134`
- `test_pearson = 0.725937`

This is the strongest evidence in the repository that single-metric validation ranking is not enough for final model selection.

## Validation Metrics

| version | val_spearman | val_pearson | val_mae | val_mse |
|---------|--------------|-------------|---------|---------|
| `21` | `0.764566` | `0.764760` | `0.488606` | `0.496259` |
| `25` | `0.761268` | `0.761313` | `0.506163` | `0.505429` |
| `29` | `0.758393` | `0.765299` | `0.497785` | `0.498131` |
| `37` | `0.758187` | `0.764629` | `0.505994` | `0.504661` |

Direct reading:

- Best `val_spearman`: `version_21`
- Best `val_pearson`: `version_29`
- Best `val_mae`: `version_21`
- Best `val_mse`: `version_21`

## Validation Hard-Case Metrics

### Negative Tail `target <= -3.0`

| version | mae | bias |
|---------|-----|------|
| `21` | `1.3509` | `+1.3358` |
| `25` | `1.3987` | `+1.3935` |
| `29` | `1.4079` | `+1.4058` |
| `37` | `1.4004` | `+1.3977` |

### Negative Tail `target <= -3.5`

| version | mae | bias |
|---------|-----|------|
| `21` | `1.8339` | `+1.8339` |
| `25` | `1.8760` | `+1.8760` |
| `29` | `1.9049` | `+1.9049` |
| `37` | `1.8982` | `+1.8982` |

Direct reading:

- None of these versions solves the hardest negative tail.
- But `version_21` is best on both tail buckets among the four compared versions.

## Validation Large-Gap Ranking

Sampled validation pairs with `|y_i - y_j| > 1.5`:

| version | error_rate |
|---------|------------|
| `21` | `0.053834` |
| `25` | `0.053784` |
| `29` | `0.052637` |
| `37` | `0.053186` |

Direct reading:

- `version_29` is best on large-gap ordering.
- `version_21` is not better than the ranking-line models on this metric.
- This is the strongest validation-side evidence that gap-aware ranking improved meaningful ranking structure even when total `val_spearman` did not increase.

## Practical Interpretation

### `version_21`

- Best single-metric validation winner overall.
- Best validation MAE/MSE overall.
- Best current tail MAE on `<= -3.0` and `<= -3.5`.
- But its test generalization is much worse, so it cannot be trusted as the default final model.

### `version_25`

- Best pure validation winner inside the ranking-centered branch under the current single-metric rule.
- Strongest clean ranking baseline.
- Still weak on the hardest negative tail.

### `version_29`

- Best validation-structure candidate.
- Strongest on `val_pearson`, `val_mae`, `val_mse`, and large-gap pair ranking.
- Slightly weaker than `version_25` on `val_spearman` and tail MAE.

### `version_37`

- Best observed test result, but not a clean validation winner.
- Keeps the gap-aware direction strong, but does not clearly improve validation over `version_29`.
- Should be treated as a promising test-side result, not automatically as the main validation-approved baseline.

## What `version_21` Learns Differently

The new detailed validation export for `version_21` makes the contrastive-learning branch easier to interpret.

### Where `version_21` is better

- It is best on overall `val_spearman`, `val_mae`, and `val_mse`.
- It is best on the negative tail:
  - `target <= -3.0`: `MAE = 1.3509`
  - `target <= -3.5`: `MAE = 1.8339`
- It is also best in the near-zero boundary band:
  - `target in [-0.25, 0.25]`: `MAE = 0.3158`
  - `target in [-0.5, 0.5]`: `MAE = 0.3523`

This suggests that contrastive learning helped local neighborhood consistency and reduced pointwise error in the tail and boundary regions.

### Where the ranking line is better

The ranking-line models, especially `version_29`, are stronger on large-gap ordering:

| version | gap `1.5-2.5` | gap `2.5-3.5` | gap `> 3.5` |
|---------|---------------|---------------|-------------|
| `21` | `0.064860` | `0.024984` | `0.000000` |
| `29` | `0.062972` | `0.025625` | `0.001894` |
| `37` | `0.063242` | `0.027119` | `0.001894` |

Interpretation:

- `version_21` is not clearly better on the ordering regimes that matter most for ranking-style generalization.
- `version_29` improves the most populated large-gap regime (`1.5-2.5`) even though its raw `val_spearman` is lower.

### Why This Likely Happened

The current evidence suggests a split between two kinds of validation improvements:

- `version_21` improves local fit:
  - lower MAE
  - better tail MAE
  - better near-zero boundary MAE
- `version_29/37` improve ranking structure:
  - better large-gap ordering
  - smaller systematic shrinkage on some middle and positive regions

So the most plausible explanation is:

- contrastive learning helped local representation smoothness
- ranking loss helped the model prioritize cross-sample order relationships that transfer better to test

This is why `version_21` can look best on validation aggregates but still lose on final test ranking.

## Recommended Use

Use the versions this way:

- `version_21`: validation upper-bound warning case
- `version_25`: ranking-line validation reference baseline
- `version_29`: best validation-structure baseline
- `version_37`: best observed test reference

If the next experiment is chosen strictly by validation logic, compare it against both:

- `version_21` for overall validation score and tail behavior
- `version_25` for ranking-line `val_spearman`
- `version_29` for broader validation quality

## Recommended Validation Checklist for Future Runs

Do not judge future candidates by `val_spearman` alone.
Use this minimum checklist:

- `val_spearman`
- `val_pearson`
- `val_mae`
- validation large-gap pair error rate
- validation tail MAE at `<= -3.0`
- validation tail MAE at `<= -3.5`

## Bottom Line

Under a strict validation-only view:

- `version_21` is the best single-metric validation model.
- `version_21` is also best on validation MAE/MSE and tail MAE.
- `version_29` is the most convincing validation-side structural improvement.
- `version_37` is best on test, but not yet the most defensible validation winner.
- `version_21` is also the clearest proof that current validation selection does not align reliably with final test generalization
