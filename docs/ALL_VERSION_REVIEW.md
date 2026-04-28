# All Version Review

This note summarizes the current signal across all completed `results/lightning_logs/version_*` runs and records the most defensible next step.

## What Clearly Worked

### 0. `deltaG` auxiliary supervision is the first clean new win

The newest `v36` line is the first branch that improves validation and test together without the previous tradeoff pattern:

- `version_82`
  - `val_spearman = 0.7643`
  - `test_spearman = 0.7554`
  - `test_pearson = 0.7456`
  - `test_loss = 0.5541`

Direct follow-up ablations showed:

- `version_83` (`deltaG_weight = 0.05`) is slightly worse
- `version_84` (no CI weighting) is clearly worse on test
- `version_85` (`deltaG_weight = 0.2`) raises validation but overfits test

Interpretation:

- `deltaG` supervision is genuinely useful
- the best current operating point is:
  - `deltaG_weight = 0.1`
  - `deltaG_ci_weighting = true`
- this is the strongest current single-model candidate

### 1. The ranking line is the most reliable source of gain

The strongest and most repeatable improvements came from adding ranking supervision on top of the `v13` backbone family:

- `version_25`: clean ranking baseline
  - `val_spearman = 0.7613`
  - `test_spearman = 0.7494`
- `version_29`: gap-aware ranking improves beyond plain ranking
  - `test_spearman = 0.7506`
- `version_37`: stronger ranking weight still helps
  - `test_spearman = 0.7513`
- `version_46`: simple ranking line also reached a strong single-run result
  - `test_spearman = 0.7517`

Interpretation:

- `ranking` is indispensable for good final ordering.
- `gap-aware ranking` is useful, but its gains are small and already mostly saturated.
- The clean `v13 + ranking` line remains the most trustworthy baseline family.

### 2. Latent + contrastive is a real signal, but not yet a clean win

The latent branch around `proagg_mlp_v24` consistently pushed validation high:

- `version_49`
  - `val_spearman = 0.7662` (best observed validation score)
  - `test_spearman = 0.7489`

Follow-up runs showed:

- removing ranking (`version_51`) hurts badly
- regime-aware and tail-weighted contrastive (`52/53`) recover some behavior
- codebook (`54`) does not improve over `49`
- stronger contrastive weights (`60/61/62`) improve this family somewhat
  - best among them: `version_62`
  - `test_spearman = 0.7499`

Interpretation:

- the latent-space idea is not nonsense
- contrastive shaping is learning something useful
- but this family still does not cleanly beat the ranking line on test
- the main issue looks like training dynamics, not lack of representational signal

## What Did Not Leave a Strong Signal

These directions should not be prioritized right now:

- complex pooling variants (`version_42`, `version_43`)
- SaProt + ESM2 dual encoder (`version_48`)
- current lightweight codebook/prototype regularization (`version_54`)
- `val_loss` as the main checkpoint metric (`version_70`, `72`, `73` are severe failures)
- removing ranking entirely (`version_51`, `63`, `68`)

Interpretation:

- backbone/pooling redesign is not the best current lever
- ranking should remain in the objective
- the selection metric should stay ranking-oriented, not loss-oriented

## Current Best Anchors

There are three different anchors depending on what is being optimized:

- clean baseline:
  - `version_25`
- strongest ranking-family test result:
  - `version_46`
- strongest raw validation result:
  - `version_49`
- strongest current overall candidate:
  - `version_82`

This split is important because these versions do not represent the same thing:

- `version_25` is the cleanest baseline
- `version_46` is the best simple ranking-family run
- `version_49` is the strongest proof that the latent-space branch captures useful validation structure
- `version_82` is the strongest current model under both validation and test-side evidence

## Most Defensible Next Step

The next experiment should avoid adding new large model structure.

The strongest current resolved hypothesis is now:

- folding stability carries useful signal for `75°C` aggregation prediction
- this signal works best as an auxiliary target, not as a replacement target
- uncertainty-aware weighting via `deltaG_95CI` matters

That makes `version_82` the current main line.

The most defensible next ablations should stay close to this line:

1. check bootstrap significance against older anchors
2. analyze the `deltaG` head itself
3. only then consider a small extension such as combining `deltaG` auxiliary supervision with a stronger ranking-family backbone if needed

## Practical Recommendation

If only one new direction should be pursued next:

- keep `version_25` as the formal clean baseline
- keep `version_46` as the best current ranking-family reference
- keep `version_82` as the strongest current candidate
- avoid reopening latent/codebook/MoE branches unless `version_82` stalls under stronger validation/bootstrap scrutiny
