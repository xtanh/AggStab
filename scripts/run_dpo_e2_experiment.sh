#!/bin/bash

set -e

cd /home/xy_th/Project_protein_aggregation

CUDA_VISIBLE_DEVICES=3 \
PDB_TRAIN=data/dpo/subsets/threshold_balance_ltneg1_eq/train \
PDB_VALID=data/dpo/subsets/threshold_balance_ltneg1_eq/valid \
PDB_TEST=data/dpo/representative_pdbs/test \
SELECT_PDB_VALID=data/dpo/subsets/threshold_balance_ltneg1_eq/valid \
MAX_TRAIN_PDBS=-1 \
MAX_VALID_PDBS=366 \
MAX_TEST_PDBS=1350 \
SELECT_MAX_VALID_PDBS=366 \
NUM_SAMPLES=16 \
TEMPERATURE=0.5 \
SEED=42 \
ROUNDS=2 \
ROUND_EPOCHS=2 \
ROUND_TRAIN_SPLIT_MODE=halves \
ROUND_TRAIN_SPLIT_SEED=42 \
DPO_BETA=0.1 \
DPO_BATCH_SIZE=32 \
DPO_PATIENCE=2 \
AGG_CONFIG=configs/proagg_final_candidate.yaml \
STAB_CONFIG=configs/proagg_deltaG_only.yaml \
STABILITY_CSV=data/rocklin/Metagenomic_dG.csv \
AGG_SCORE_GAP_DELTA=0.20 \
STAB_SCORE_GAP_DELTA=0.20 \
STABILITY_GATE_MODE=wt_absolute \
STABILITY_GATE_MARGIN=0.5 \
VALID_SELECTION_ENABLED=1 \
VALID_SELECTION_METRIC=joint_sum \
VALID_SELECTION_MAX_PENALTY=0.1 \
VALID_ORIGINAL_RESULTS_CACHE=results/cache/thbal_valid_original_n16_t05_seed42.json \
ORIGINAL_RESULTS_CACHE=results/cache/test_original_n16_t05_seed42.json \
RUN_TAG=semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05 \
OUTPUT_ROOT=results/semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05 \
nohup bash scripts/run_dpo_joint_semi_online_pipeline.sh \
  results/lightning_logs/version_92/checkpoints/best_epoch=09_val_spearman=0.7649.ckpt \
  results/lightning_logs/version_96/checkpoints/best_epoch=10_val_spearman=0.6817.ckpt \
  cuda:0 \
  > results/semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05.log 2>&1 &

echo "Started DPO r2/e2 experiment."
echo "Log: results/semi_joint_dpo_thbalfull_halves_r2_e2_valselect_m05.log"
