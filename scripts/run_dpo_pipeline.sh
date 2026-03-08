#!/bin/bash
# ============================================================
# End-to-end DPO pipeline: sample -> score -> train -> evaluate
# ============================================================
#
# Usage:
#   bash scripts/run_dpo_pipeline.sh \
#       <pdb_dir> \
#       <proagg_checkpoint> \
#       [device]
#
# Example:
#   bash scripts/run_dpo_pipeline.sh \
#       inputs/pdbs \
#       results/lightning_logs/version_0/checkpoints/best_epoch05_val_spearman0.6620.ckpt \
#       cuda:3

set -e

eval "$(conda shell.bash hook)"
conda activate surface

PDB_DIR=${1:?"Usage: $0 <pdb_dir> <proagg_ckpt> [device]"}
PROAGG_CKPT=${2:?"Usage: $0 <pdb_dir> <proagg_ckpt> [device]"}
DEVICE=${3:-"cuda:3"}

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

PAIRS_PATH="data/dpo_pairs.pt"
DPO_OUTPUT="results/dpo"

echo "============================================================"
echo "Step 1: Sample sequences + Score with ProAgg + Build pairs"
echo "============================================================"
python src/dpo/sample_and_score.py \
    --pdb_dir "$PDB_DIR" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config configs/default.yaml \
    --output "$PAIRS_PATH" \
    --num_samples 64 \
    --temperature 0.1 \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "Step 2: DPO fine-tuning"
echo "============================================================"
python src/dpo/dpo_train.py \
    --pairs_path "$PAIRS_PATH" \
    --output_dir "$DPO_OUTPUT" \
    --device "$DEVICE" \
    --epochs 10 \
    --lr 1e-5 \
    --beta 0.1

# Find the latest checkpoint
LATEST_CKPT=$(ls -t "$DPO_OUTPUT"/mpnn_dpo_epoch*.pt | head -1)

echo ""
echo "============================================================"
echo "Step 3: Evaluation"
echo "============================================================"
python src/dpo/evaluate.py \
    --pdb_dir "$PDB_DIR" \
    --dpo_mpnn_ckpt "$LATEST_CKPT" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --output "$DPO_OUTPUT/eval_results.json" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Pairs: $PAIRS_PATH"
echo "  DPO checkpoint: $LATEST_CKPT"
echo "  Evaluation: $DPO_OUTPUT/eval_results.json"
echo "============================================================"
