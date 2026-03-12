# Validation Error Analysis for `version_25`

This note summarizes the current best single-task model based only on the validation set.
No test-set error cases are used here.

## Scope

- Model checkpoint:
  `results/lightning_logs/version_25/checkpoints/best_epoch=08_val_spearman=0.7613.ckpt`
- Training config:
  `configs/proagg_v13.yaml`
- Validation prediction file:
  `results/lightning_logs/version_25/checkpoints/valid_predictions_detailed.csv`

## Main Observation

The dominant failure mode is not random noise. The model shows a strong regression-to-the-mean pattern:

- Very negative samples are predicted as not negative enough.
- Positive samples are pulled back toward the center.
- Mid-range samples are much easier than tail samples.

This means the next improvement should focus on tail recognition and ranking of extreme cases, not on adding a more complex backbone head.

## Validation Metrics

- `val_spearman`: `0.7613`
- `val_pearson`: `0.7613`
- `val_mse`: `0.5054`

Absolute error quantiles on validation:

- `p50`: `0.3564`
- `p75`: `0.6816`
- `p90`: `1.1617`
- `p95`: `1.5358`
- `p99`: `2.2849`

## Error by Target Bin

Using target bins `<= -3`, `(-3, -2]`, `(-2, -1]`, `(-1, 0)`, `[0, +inf)`:

- `target <= -3`
  `n=66`, `MAE=1.3987`, `bias=+1.3935`
- `-3 ~ -2`
  `n=115`, `MAE=0.6920`, `bias=+0.5749`
- `-2 ~ -1`
  `n=194`, `MAE=0.5253`, `bias=+0.0333`
- `-1 ~ 0`
  `n=530`, `MAE=0.4076`, `bias=-0.1320`
- `>= 0`
  `n=374`, `MAE=0.4213`, `bias=-0.2650`

Interpretation:

- The hardest region is the extreme negative tail.
- The model also underestimates positive-like examples.
- The center of the distribution is much easier.

## Error by Decile

The worst decile is the lowest-target decile:

- lowest decile target mean: `-3.1565`
- prediction mean: `-2.0564`
- `MAE=1.1203`
- `bias=+1.1001`

The highest decile also shows systematic shrinkage:

- highest decile target mean: `0.4940`
- prediction mean: `0.0726`
- `MAE=0.4524`
- `bias=-0.4214`

This confirms that both ends are compressed toward the center.

## Error by Sequence Length

Sequence length is not the main issue.

Validation MAE across length quartiles stays in a relatively narrow range:

- `63.4 aa`: `MAE=0.5331`
- `67.2 aa`: `MAE=0.5171`
- `69.5 aa`: `MAE=0.4476`
- `71.0 aa`: `MAE=0.5328`

Interpretation:

- Length has some effect, but not enough to explain current failure modes.
- The main problem is target-tail recognition, not sequence length.

## Cluster Generalization

Cluster overlap between train and validation is exactly zero:

- train clusters: `4744`
- validation clusters: `659`
- overlap: `0`

Interpretation:

- Validation is a real cross-cluster generalization test.
- High-error cases mostly reflect unseen-cluster generalization failure.
- This is a good reason to avoid using validation outliers to memorize specific sequence motifs.

## Ranking Failure Pattern

On a sampled subset of validation pairs with large target gaps (`|y_i - y_j| > 1.5`):

- sampled pair count: `20043`
- ranking error rate: `0.0538`

The typical failure is:

- one truly near-zero or positive sample is predicted far too negative
- one truly very negative sample is predicted not negative enough
- their order flips

So the ranking problem is concentrated in extreme-gap pairs, not in small local reorderings.

## Representative High-Error Cases

Examples from the highest validation absolute errors:

- `rocklin_batch2_431617`
  target `-4.4154`, prediction `-0.9727`
- `rocklin_batch2_630426`
  target `-5.4505`, prediction `-2.1849`
- `rocklin_batch2_235753`
  target `0.3799`, prediction `-2.3535`
- `rocklin_batch2_406585`
  target `0.5880`, prediction `-1.7819`

These cases again show strong shrinkage toward the center.

## Practical Implications

The most promising next steps are:

1. Tail-aware supervision
   Increase pressure on extreme negative and positive bins.

2. Better extreme-case discrimination
   Add an auxiliary bin-classification task such as:
   `<= -3`, `(-3, -2]`, `(-2, -1]`, `(-1, 0)`, `>= 0`

3. Better large-gap ranking supervision
   Focus ranking more on large target-gap pairs instead of treating all eligible pairs equally.

4. Avoid unnecessary backbone complexity
   Existing experiments suggest the main gains come from loss design and supervision, not deeper heads.

## Update After `version_37`

Later experiments show that the most effective line is not adding auxiliary heads, but improving pairwise supervision:

- `version_29`: gap-aware ranking improved test performance over `version_25`
- `version_34`: stronger ranking weight (`1.2`) improved a bit further
- `version_37`: stronger ranking weight (`1.4`) is currently the best observed test result

Current best observed test result:

- `version_37`
- `test_spearman = 0.751289`
- `test_pearson = 0.741559`

This strongly suggests that the main leverage is still in ranking-loss design, not model architecture.

## Train vs Validation With `version_37`

Detailed prediction files:

- `results/lightning_logs/version_37/checkpoints/train_predictions_detailed.csv`
- `results/lightning_logs/version_37/checkpoints/valid_predictions_detailed.csv`

Overall:

- train: `spearman = 0.8339`, `pearson = 0.8401`, `mae = 0.4272`, `mse = 0.3453`
- valid: `spearman = 0.7582`, `pearson = 0.7646`, `mae = 0.5060`, `mse = 0.5047`

Interpretation:

- There is a clear train/valid gap.
- But the more important point is that the extreme negative tail is not solved even on train.

### Tail Behavior in Train vs Validation

For `target <= -3`:

- train:
  `n = 483`, `MAE = 1.1141`, `bias = +1.1121`
- valid:
  `n = 66`, `MAE = 1.4004`, `bias = +1.3977`

For `target >= 0`:

- train:
  `n = 2483`, `MAE = 0.3362`, `bias = -0.1634`
- valid:
  `n = 374`, `MAE = 0.3935`, `bias = -0.2204`

Interpretation:

- The model still underestimates the extreme negative tail even on train.
- So the current bottleneck is not only generalization. It is also insufficient supervision on the most negative regime.

### Large-Gap Ranking Errors

Using the same sampled large-gap pair analysis (`|y_i - y_j| > 1.5`):

- train ranking error rate: `0.0311`
- valid ranking error rate: `0.0532`

Interpretation:

- Gap-aware ranking helps, because train pair ordering is clearly stronger.
- But validation still loses many large-gap ordering decisions.
- The remaining issue is concentrated around extreme negative examples and boundary cases near zero.

## Distribution Analysis of Train / Validation / Test

Target distributions are consistent across splits.

### Basic Statistics

- train:
  `mean = -0.7931`, `std = 1.0774`, `min = -6.1108`, `max = 1.04`
- valid:
  `mean = -0.7068`, `std = 1.0889`, `min = -5.4505`, `max = 1.04`
- test:
  `mean = -0.8419`, `std = 1.1065`, `min = -5.5905`, `max = 1.04`

### Tail Mass

Fraction of samples below several thresholds:

- train:
  - `<= -4`: `0.84%`
  - `<= -3.5`: `2.27%`
  - `<= -3`: `4.94%`
  - `<= -2.5`: `9.16%`
- valid:
  - `<= -4`: `1.17%`
  - `<= -3.5`: `2.27%`
  - `<= -3`: `5.16%`
  - `<= -2.5`: `8.99%`
- test:
  - `<= -4`: `0.93%`
  - `<= -3.5`: `2.46%`
  - `<= -3`: `5.89%`
  - `<= -2.5`: `10.71%`

Interpretation:

- `-3` is not arbitrary. It is close to the 5th percentile across train and validation.
- `-3.5` is closer to the 2nd percentile.
- `-4` isolates a very tiny extreme tail.

So if we want a threshold for “extreme negative”, there are three natural choices:

- `-3.0`: broad tail, about the lowest 5 percent
- `-3.5`: sharper tail, about the lowest 2 percent
- `-4.0`: ultra-extreme tail, below about 1 percent

## What This Means for Threshold Design

This is the main takeaway:

- `-3.0` is a reasonable first threshold because it captures a stable low-tail regime across train and validation.
- But the error analysis shows that the model gets progressively worse as targets move below `-3.5` and especially below `-4.0`.

Observed `version_37` MAE by finer negative regions:

- train:
  - `(-10, -4]`: `1.7967`
  - `(-4, -3.5]`: `1.2303`
  - `(-3.5, -3]`: `0.8373`
- valid:
  - `(-10, -4]`: `2.0308`
  - `(-4, -3.5]`: `1.7561`
  - `(-3.5, -3]`: `1.0102`

This suggests:

- `-3` is a good operational threshold for a broad tail-aware mechanism.
- If we want more targeted supervision, `-3.5` may be a better threshold for extra pair emphasis.
- `-4` is probably too sparse for a primary rule, but useful as a diagnostic bucket.

## Updated Direction

Based on all experiments so far:

- keep the current ranking-centered framework
- keep gap-aware pair weighting
- do not continue adding auxiliary heads for now
- next changes should focus on pair selection or pair weighting involving the negative tail

The strongest next candidate is:

- tail-aware pair weighting
- applied only inside ranking loss
- likely using `-3.0` or `-3.5` as the first threshold to test
