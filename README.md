# AggStab

Code for multi-objective protein inverse folding with aggregation- and stability-aware preference alignment.

## Scope

This repository contains the code used for:

- training the aggregation predictor,
- training the stability predictor,
- semi-online preference optimization on top of ProteinMPNN,
- winner-only SFT and hybrid DPO+SFT baselines,
- candidate export, top-k selection, and Chai-1 structure validation.

Large artifacts are intentionally excluded from version control:

- `data/`
- `results/`
- checkpoints
- generated figures and logs

The repository is organized for code review and paper reproduction, not for shipping pretrained weights.

## Repository layout

- `src/models/`
  - predictor architectures
- `src/ln/`
  - Lightning training and evaluation for property predictors
- `src/dpo/`
  - preference optimization trainers and scoring utilities
- `scripts/`
  - experiment pipelines, plotting, export, and analysis
- `configs/`
  - predictor configs
- `docs/`
  - experiment notes, final result summaries, and paper-facing records

## Main code paths

### Property predictors

- Aggregation / stability model definition:
  - `src/models/ProAgg.py`
- Predictor training:
  - `src/ln/lightning_train.py`
- Detailed predictor evaluation:
  - `src/ln/evaluate_detailed.py`

### Preference optimization

- Joint pair construction:
  - `src/dpo/sample_and_score_joint.py`
- Pure DPO trainer:
  - `src/dpo/dpo_train.py`
- Winner-only SFT trainer:
  - `src/dpo/sft_train.py`
- Hybrid DPO+SFT trainer:
  - `src/dpo/dpo_sft_train.py`
- Cached evaluation:
  - `src/dpo/evaluate.py`

### Experiment pipelines

- Semi-online pure DPO:
  - `scripts/run_dpo_joint_semi_online_pipeline.sh`
- Semi-online pure SFT:
  - `scripts/run_sft_joint_semi_online_pipeline.sh`
- Semi-online hybrid DPO+SFT:
  - `scripts/run_dpo_sft_joint_semi_online_pipeline.sh`
- Validation-based checkpoint selection:
  - `scripts/select_best_joint_checkpoint.py`

### Final downstream analysis

- Export full-test candidates:
  - `scripts/export_dpo_test_candidates.py`
- Select joint top-k candidates:
  - `scripts/select_joint_topk_candidates.py`
- Run Chai-1 in shards:
  - `scripts/run_chai_batch_from_candidates.py`
- Final strict joint analysis:
  - `scripts/analyze_joint_structure_vs_wt.py`

## Final method summary

The current best end-to-end method is:

- semi-online hybrid DPO+SFT
- two rounds
- one epoch per round
- disjoint train halves across rounds
- validation-based checkpoint selection
- best hybrid weight: `w_sft = 1.0`

Key result summary is maintained in:

- `docs/results.md`
- `docs/chinese.md`

## Reproduction prerequisites

This code expects local access to:

- ProteinMPNN weights
- SaProt checkpoints / tokenizer assets
- Rocklin dataset files
- Chai-1 runtime environment

Those assets are not committed here.

## Recommended review order

For a fast code review, read files in this order:

1. `docs/results.md`
2. `src/dpo/sample_and_score_joint.py`
3. `src/dpo/dpo_train.py`
4. `src/dpo/sft_train.py`
5. `src/dpo/dpo_sft_train.py`
6. `scripts/run_dpo_joint_semi_online_pipeline.sh`
7. `scripts/run_dpo_sft_joint_semi_online_pipeline.sh`
8. `scripts/analyze_joint_structure_vs_wt.py`

## Notes

- The repository currently tracks historical experiment notes in `docs/`.
- The codebase is still research code; the important part is that the training and analysis paths are explicit and auditable.
