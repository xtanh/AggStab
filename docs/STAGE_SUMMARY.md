# ProAgg Stage Summary

This note is a convergence summary of the current experimental stage.
Its purpose is to answer four questions:

1. which directions have shown real value
2. which directions are no longer worth exploring
3. what the train/validation error cases jointly imply
4. what the next stage should focus on

## Current Situation

The project is now in a clear plateau regime.

- Many recent changes improve `test_spearman` only by `0.000x ~ 0.001x`
- Several directions improve one aspect of validation while hurting another
- Repeated single-run tuning on one validation split has made model selection ambiguous

So the priority is no longer “try one more small hyperparameter tweak”.
The priority is to identify the real bottleneck and stop spending time on low-leverage branches.

## What Has Actually Worked

### 1. Ranking supervision is the main source of real gain

The strongest stable improvements came from pairwise ranking, not from increasing head complexity.

Useful components:

- attention pooling
- ranking loss
- gap-aware pair weighting
- moderate regularization

This is the strongest and most repeatable signal across the whole experiment set.

### 2. Gap-aware ranking improves the right kind of structure

`version_29` and the later ranking-line variants showed that:

- large-gap ordering improves
- test ranking improves
- the gains are more meaningful than simple MSE/Pearson improvements

This matters because the downstream objective is closer to ranking quality than pure numeric fit.

### 3. Light contrastive regularization can improve final test ranking

`version_40/41` established that a hybrid line is viable:

- `ranking + light contrastive` achieved the best observed `test_spearman`
- the gain is not huge, but it is consistent enough to be taken seriously

However, this came with a cost:

- worse `test_pearson`
- worse `test_loss`
- worse validation MAE/MSE

So this direction is not a clean global improvement. It is a targeted ranking improvement.

## What Has Not Worked

The following directions did not show strong enough evidence to justify more time right now:

- deeper or wider MLP heads
- multi-layer hidden-state fusion
- hard tail-weighted regression
- bin auxiliary classification
- ordinal auxiliary task
- tail-aware pair weighting
- direct `attention + mean + max` feature concatenation

The key lesson is:

- most architecture-side changes were weak or destructive
- most extra auxiliary losses improved one local behavior but did not improve the whole task

## What Train and Validation Error Cases Jointly Mean

This is the most important conclusion from the current stage.

### 1. The hardest problem is the strong negative tail

The most aggregation-prone sequences remain the hardest cases.

This pattern is visible on both:

- train
- validation

That means:

- the bottleneck is not only generalization
- the current representation is not fully separating these hard tail cases even during training

### 2. There is a persistent tradeoff between tail discrimination and middle-range calibration

Different branches improve different regions:

- `version_21`:
  - best validation fit
  - best tail MAE
  - best near-zero boundary MAE
- `version_29/37`:
  - better large-gap ordering
  - better ranking-style generalization
- `version_40/41`:
  - best observed test Spearman
  - strongest tail improvement among recent ranking/hybrid models
  - but noticeably worse middle-range and numeric calibration

This implies the model is still using limited representation capacity to trade off:

- extreme hard-case separation
- middle-range fine-grained calibration

### 3. The bottleneck now looks more like a feature bottleneck than a loss bottleneck

Loss engineering has already produced most of the available cheap gains.

What remains is more structural:

- the model does not yet represent the hardest sequence patterns cleanly enough
- when the loss is pushed to favor hard tails, the middle range is damaged
- when the loss is softened to preserve the middle range, hard tails remain under-modeled

This is why continued loss tweaking is now low leverage.

## Interpretation of the Main Versions

### `version_21`

- strongest validation model by raw metrics
- strongest tail and boundary fit
- but poor final test generalization

Interpretation:

- contrastive learning improved local fit
- but did not improve the kind of ordering that transfers best to test

### `version_29` and `version_37`

- clearest proof that ranking-centered supervision is the most reliable main line
- stronger on large-gap ordering
- better aligned with final test improvements

Interpretation:

- these versions improved structural ordering rather than only local fit

### `version_40` and `version_41`

- current best observed `test_spearman`
- very likely a real ranking-side gain
- but still not a clean validation-side winner

Interpretation:

- light contrastive regularization helped recover some hard-tail discrimination
- but the hybrid objective still distorts middle-range calibration

### `version_42`

- feature-side attempt with direct `attention + mean + max`
- clearly failed on both validation and test

Interpretation:

- not every richer pooled representation helps
- naive multi-statistic concatenation can destroy the learned geometry

## What This Means for Checkpoint Selection

Do not select checkpoints by `val_loss` alone.
Do not select checkpoints by `val_spearman` alone either.

The current evidence shows both are incomplete.

For the next stage, validation-side judgment should include:

- `val_spearman`
- `val_pearson`
- `val_mae`
- validation large-gap pair error
- validation tail MAE at `<= -3.0`
- validation tail MAE at `<= -3.5`

In practice:

- `val_spearman` should remain the primary metric
- but candidates should be filtered by the rest of the validation structure

## Next-Stage Priority

### Highest priority

- move from loss micro-tuning to representation-focused improvements

Why:

- loss engineering is near saturation
- the current bottleneck is how hard cases are represented

### Reasonable next directions

1. more careful feature design, but only with minimal and interpretable changes
2. better analysis of which sequence-level signals define the hardest tail
3. possibly two-stage training rather than one-stage mixed objectives

### Lower priority

- more tuning of `ranking_weight`
- more tuning of `cl_weight`
- more auxiliary-task experiments
- more direct tail weighting heuristics

These are now low-leverage compared with the time they cost.

## Working Summary

The current stage supports the following working belief:

- ranking supervision was the right backbone decision
- hybrid ranking + light contrastive can push test Spearman a bit further
- but the remaining limitation is not just optimization
- the remaining limitation is that the current feature representation does not simultaneously preserve:
  - hard-tail discrimination
  - large-gap ordering
  - middle-range calibration

So the next stage should stop chasing tiny loss-side gains and start focusing on feature representation with much tighter hypotheses.

## Next Representation Hypotheses

The next representation stage should stay hypothesis-driven.
The current best candidates are:

1. local hotspot pooling
   Add a small feature branch that summarizes only the top-attended residues.
   Rationale:
   strong aggregation may be driven by a few critical local regions rather than the full-sequence average.

2. conservative two-layer mixing
   Use a very light mix of the last two transformer layers instead of a large multi-layer fusion.
   Rationale:
   the final layer alone may be too task-agnostic, but previous wide multi-layer fusion was too destructive.

3. residual dual-scale fusion
   Keep the current global attention-pooled representation as the anchor, and add a small residual local feature rather than concatenating many statistics.
   Rationale:
   the failed `attention + mean + max` result suggests that preserving the original geometry matters.

The first implementation choice for the next stage should be:

- residual local hotspot pooling

Why this is the best first test:

- it directly targets the hard-tail hypothesis
- it is more conservative than full multi-statistic concatenation
- it preserves the existing `v13` global representation as the main path
