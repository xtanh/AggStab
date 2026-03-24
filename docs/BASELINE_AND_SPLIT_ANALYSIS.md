# Baseline And Split Analysis

This note answers two practical questions:

1. which completed experiment should be used as the baseline if model selection must rely on validation only
2. whether the train / validation / test split looks problematic from the label-distribution side

## 1. Validation-Only Baseline Selection

The selection rule here is strict:

- do not use test results to choose the baseline
- use the best recorded validation checkpoint metric only

### Validation Ranking of Completed Runs

Top completed runs by recorded checkpoint `val_spearman`:

| rank | version | recorded best `val_spearman` |
|------|---------|-------------------------------|
| 1 | `version_17` | `0.7646` |
| 1 | `version_21` | `0.7646` |
| 3 | `version_16` | `0.7645` |
| 4 | `version_22` | `0.7637` |
| 5 | `version_45` | `0.7636` |
| 6 | `version_24` | `0.7629` |
| 7 | `version_15` | `0.7617` |
| 8 | `version_25` | `0.7613` |
| 9 | `version_40` | `0.7610` |
| 9 | `version_41` | `0.7610` |

### Baseline Choice

If we rank only by raw recorded `val_spearman`, the best value is a tie:

- `version_17`
- `version_21`

But this is not the right baseline criterion.
Both the broader experiment history and the detailed validation analysis show that the high-validation contrastive branch is more prone to overfitting-like behavior and is not the cleanest foundation for the next stage.

Following Occam's razor, the clean baseline should satisfy:

- strong validation performance
- minimal extra tricks
- stable interpretation
- no reliance on components that showed unclear or fragile gains

Under that rule, the recommended clean baseline is:

- `version_25`
- checkpoint:
  `results/lightning_logs/version_25/checkpoints/best_epoch=08_val_spearman=0.7613.ckpt`

Why `version_25` is the recommended clean baseline:

- it stays on the simplest effective ranking-centered line
- it avoids contrastive auxiliary training
- it avoids gap-aware weighting and later hybrid additions
- it remains one of the strongest validation models in the main ranking branch
- it is easier to explain and reproduce than the later mixed variants

How to interpret the other key versions:

- `version_21`
  - best raw validation score
  - not recommended as the clean baseline because the contrastive branch showed unstable generalization behavior
- `version_29`
  - best validation-structure candidate
  - useful as a second reference, but not the simplest baseline
- `version_40/41`
  - best observed test results
  - explicitly not used for baseline selection here

Important note:

- `version_25` is the clean validation baseline
- it is not the best test model
- that distinction must remain explicit

## 2. Split Distribution Analysis

The split was constructed by cluster disjointness, and the cluster separation itself is clean:

- train clusters: `4744`
- valid clusters: `659`
- test clusters: `1351`
- train / valid overlap: `0`
- train / test overlap: `0`
- valid / test overlap: `0`

So there is no direct cluster leakage.

### Label Distribution Summary

#### Train

- `n = 9774`
- mean: `-0.7931`
- std: `1.0774`
- min: `-6.1108`
- max: `1.0400`

#### Validation

- `n = 1279`
- mean: `-0.7068`
- std: `1.0889`
- min: `-5.4505`
- max: `1.0400`

#### Test

- `n = 2800`
- mean: `-0.8419`
- std: `1.1065`
- min: `-5.5905`
- max: `1.0400`

### Tail Mass By Threshold

Fraction of samples below each threshold:

| threshold | train | valid | test |
|-----------|-------|-------|------|
| `<= -4.0` | `0.839%` | `1.173%` | `0.929%` |
| `<= -3.5` | `2.271%` | `2.267%` | `2.464%` |
| `<= -3.0` | `4.942%` | `5.160%` | `5.893%` |
| `<= -2.5` | `9.157%` | `8.991%` | `10.714%` |
| `<= -2.0` | `15.510%` | `14.152%` | `16.964%` |
| `<= -1.0` | `33.221%` | `29.320%` | `34.750%` |
| `<= 0.0`  | `74.596%` | `70.758%` | `75.786%` |

## What The Split Statistics Suggest

### 1. The split is clean in sequence-cluster terms

There is no cluster overlap, so the generalization setting itself is valid.

### 2. Validation is slightly less negative than train

Compared with train:

- validation has a less negative mean
- validation has a smaller fraction below `-1`
- validation has a smaller fraction below `0`

This means validation is slightly easier in the middle of the distribution.

### 3. Test is the most negative split

Compared with both train and validation, test contains:

- more samples below `-3`
- more samples below `-2.5`
- more samples below `-2`
- more samples below `-1`

This is important because the hardest failure mode of the model is the negative tail.

So even though the split is clean, the test set is not perfectly distribution-matched to validation.
It is slightly more tail-heavy on the negative side.

### 4. This helps explain the repeated validation/test mismatch

Current experiments show that:

- models with stronger validation fit do not always win on test
- models that improve hard-tail ranking sometimes win on test even with weaker validation aggregates

The split statistics make that easier to understand:

- validation is not badly constructed
- but it is a bit less harsh than test on the negative side
- therefore a model that looks best on validation may still underperform on the more tail-heavy test split

## Practical Takeaways

1. Use `version_25` as the clean validation baseline.
2. Keep the distinction between validation baseline and best test model explicit.
3. Do not conclude that the split is “wrong”; the cluster split is clean.
4. Do conclude that test is somewhat more negative-tail-heavy than validation.
5. Continue to evaluate hard-tail behavior explicitly on validation, because raw `val_spearman` alone is not enough.
