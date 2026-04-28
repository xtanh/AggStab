# Two-Stage Training

This is the current minimal two-stage training setup for ProAgg.

## Goal

Separate:

1. representation shaping
2. ranking fine-tuning

The working hypothesis is:

- contrastive learning can improve local/tail representation
- ranking training can then optimize the final ordering structure more cleanly

## Stage 1: Contrastive Pretraining

Run:

```bash
python src/ln/lightning_train_cl.py --config configs/proagg_stage1_cl_v18.yaml
```

This trains `proagg_mlp_v18` with the contrastive-learning branch.

## Stage 2: Ranking Fine-Tuning

Edit `configs/proagg_stage2_rank_v18.yaml`:

- set `train.init_checkpoint` to the best checkpoint path from Stage 1

Then run:

```bash
python src/ln/lightning_train_ranking.py --config configs/proagg_stage2_rank_v18.yaml
```

The ranking script will:

- load the Stage 1 checkpoint
- copy the underlying `self.model.*` weights into the ranking model
- continue training with ranking loss

## Notes

- Stage 2 loads model weights only, not optimizer or trainer state.
- This is intentional. The goal is initialization, not resumed training.
- The architecture for both stages must match. The current setup uses `proagg_mlp_v18` in both stages.
