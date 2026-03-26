# Semi-Online DPO Plan

## Goal

Build a **semi-online dual-objective DPO** workflow for ProteinMPNN that periodically refreshes preference data using the current policy, instead of training only on a fixed offline pair dataset.

The immediate goal is not to jump to full online RL. The goal is to reduce the main weakness of offline DPO in this project:

- the model distribution drifts,
- but the preference pairs stay fixed,
- so shortcut behaviors can be reinforced even when they no longer match the current policy distribution.

## Why Semi-Online Instead of Full Online RL

Full online RL is not the right next step yet.

Reasons:

- reward design is still evolving,
- structure constraints are not yet inside the training loop,
- aggregation and stability rewards can conflict,
- full online RL would amplify reward-model bias faster than offline DPO.

Semi-online DPO is the lower-risk next step because it:

- reuses the current DPO codebase,
- keeps the optimization objective simple,
- updates candidate distributions iteratively,
- is easier to debug and compare against the current offline baseline.

## Working Definition

Semi-online DPO in this project means:

1. Start from an initial ProteinMPNN or DPO checkpoint.
2. Sample candidates from the **current policy** on a train subset.
3. Score candidates with:
   - aggregation predictor (`version_92`)
   - stability predictor (`version_96`)
4. Build dual-objective preference pairs.
5. Train DPO for a short stage.
6. Replace the policy checkpoint with the updated checkpoint.
7. Repeat the sample → score → pair → train cycle for multiple rounds.

This is effectively **iterative offline DPO with refreshed preference data**.

## Current Baseline to Compare Against

The current reference system is:

- dual-objective DPO
- SaProt-based aggregation + SaProt-based stability predictors
- `N=16`
- `beta=0.1`
- stability-conditioned aggregation pairs
- stability-heavy sampling ratio

Semi-online experiments should always be compared against this static dual-objective baseline.

## Main Research Questions

1. Does refreshing pairs improve dual-objective alignment relative to static dual DPO?
2. Does semi-online refresh reduce the aggregation–stability trade-off?
3. Does it improve joint success under:
   - higher `ProAgg`
   - higher `deltaG`
   - acceptable structure confidence
4. Does it reduce shortcut behavior without requiring full online RL?

## MVP Scope

The MVP should be intentionally narrow.

### Fixed for MVP

- keep SaProt as the property predictor backbone
- keep dual-objective DPO formulation
- keep `N=16`
- keep current dual pair construction logic
- keep current evaluation pipeline

### New for MVP

- iterative rounds of:
  - candidate generation
  - dual pair rebuilding
  - short DPO fine-tuning

### Explicitly Out of Scope for MVP

- PPO / GRPO / policy gradient RL
- structure-in-the-loop training
- ESM2 predictor control branch
- multi-round structure prediction during training

## Proposed Training Schedule

### Round 0

Initialize from:

- original ProteinMPNN, or
- current best offline dual-objective DPO checkpoint

For first implementation, start from **original ProteinMPNN** for clarity.

### Per Round

For each round `r`:

1. Use current policy checkpoint `policy_r.pt`
2. Sample train candidates
3. Build:
   - `agg_pairs_r.pt`
   - `stab_pairs_r.pt`
4. Build validation pair sets using the same policy for consistency
5. Fine-tune DPO for a small number of epochs
6. Save updated checkpoint `policy_{r+1}.pt`

### Suggested Initial Schedule

- rounds: `3`
- per-round DPO epochs: `2` to `4`
- evaluate after every round

The first target is not maximal quality. The first target is to verify whether iterative refresh changes the direction of optimization in a useful way.

## Data Construction Strategy

Use the current dual-objective pair builder as the starting point.

### Aggregation pairs

Keep the current constrained construction:

- only candidates with acceptable predicted stability can participate,
- then rank by aggregation score,
- then build `agg_pairs`.

### Stability pairs

Keep direct ranking by predicted `deltaG`.

### Important Constraint

The train pairs for round `r` must be sampled from the **current policy** of round `r`, not reused from round `r-1`.

That is the entire point of semi-online refresh.

## Evaluation Protocol

Each round should produce the same evaluation outputs as the current dual pipeline.

### Property-Level Metrics

- `delta_proagg_mean`
- `delta_proagg_max`
- `delta_deltaG_mean`
- `delta_deltaG_max`
- `delta_logprob_mean`
- `delta_diversity`

### Improvement Counts

- `improve_proagg_mean`
- `improve_proagg_max`
- `improve_deltaG_mean`
- `improve_deltaG_max`

### Joint Success Metrics

For later structure-evaluated runs, define success as:

- `ProAgg > WT`
- `deltaG > WT`
- `pLDDT >= threshold`
- `pTM >= threshold`

Backbone success:

- at least one top-k candidate satisfies all conditions

## Recommended Experiment Ladder

### Stage 1: Smoke Test

Use:

- train/valid/test = `80 / 10 / 10`
- rounds = `2`
- per-round epochs = `2`

Goal:

- verify all round outputs are produced,
- verify pair refresh actually changes the data,
- verify metrics do not collapse immediately.

### Stage 2: Bridge Test

Use:

- `400 / 50 / 50`
- rounds = `2` or `3`

Goal:

- compare static dual DPO vs semi-online dual DPO.

### Stage 3: Full Train Pool

Use:

- full `threshold_balance_ltneg1_eq` train/valid
- held-out test fixed as current `te100`

Goal:

- verify whether improvements persist at scale.

## Engineering Plan

### New Script

Add a new orchestration script:

- `scripts/run_dpo_dual_semi_online_pipeline.sh`

Responsibilities:

1. choose starting checkpoint
2. for each round:
   - generate refreshed train pairs
   - generate refreshed val pairs
   - run short dual-objective DPO training
   - advance checkpoint pointer
3. run final evaluation
4. optionally save per-round evaluation summaries

### Minimal File Changes

- reuse `src/dpo/sample_and_score_dual.py`
- reuse `src/dpo/dpo_train.py`
- reuse `src/dpo/evaluate.py`
- add only one new outer loop script first

This keeps the first implementation small.

## Logging Requirements

Each round should save:

- train pair file
- val pair file
- training history
- checkpoint
- evaluation JSON
- dual summary JSON / CSV / TXT

Suggested directory structure:

- `results/<run_tag>/round0/`
- `results/<run_tag>/round1/`
- `results/<run_tag>/round2/`
- `results/<run_tag>/final/`

## Risks

### 1. Reward Drift

Refreshing from the current policy can strengthen shortcuts if the gating is too weak.

Mitigation:

- keep stability-conditioned aggregation pair construction,
- keep strong stability-heavy training ratio.

### 2. Policy Collapse Across Rounds

Repeated refresh can narrow sequence diversity.

Mitigation:

- monitor diversity every round,
- stop if diversity drops sharply.

### 3. Validation Leakage by Moving Targets

If validation pairs are refreshed each round, validation becomes a moving target.

Mitigation:

Two variants should be considered:

- refreshed validation pairs
- fixed validation pairs

For MVP, use **refreshed validation pairs**, but keep this as an explicit design choice in reports.

### 4. Compute Cost

Refreshing pairs every round increases runtime significantly.

Mitigation:

- short per-round training,
- smoke test first,
- only scale after the direction is validated.

## Success Criteria for MVP

The MVP is successful if, relative to static dual DPO:

- `deltaG` no longer degrades,
- `ProAgg` still improves,
- `delta_logprob` does not collapse,
- dual summary metrics are stable across rounds,
- no obvious pathology explosion appears.

## Decision Rule After MVP

### If semi-online helps

Proceed to:

- bridge-scale comparison,
- structure pilot comparison,
- paper story: static vs refreshed dual-objective DPO.

### If semi-online does not help

Do not escalate to full online RL immediately.

Instead:

- revise gating,
- revise objective ratio,
- revise pair construction,
- then rerun semi-online MVP.

## Immediate Next Step

Implement:

- `scripts/run_dpo_dual_semi_online_pipeline.sh`

and run:

- `80/10/10`
- `2` rounds
- `2` epochs per round

This should be the first concrete semi-online experiment.
