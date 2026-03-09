#!/bin/bash
# ============================================================
# End-to-end DPO pipeline:
#   1. Sample & score on train clusters -> train_pairs
#   2. Sample & score on valid clusters -> val_pairs
#   3. DPO training (best checkpoint on val)
#   4. Evaluate best checkpoint on test clusters
# ============================================================
#
# Usage:
#   bash scripts/run_dpo_pipeline.sh \
#       <proagg_checkpoint> \
#       [device]
#
# Example:
#   bash scripts/run_dpo_pipeline.sh \
#       results/lightning_logs/version_12/checkpoints/best.ckpt \
#       cuda:3
#
# Prerequisites:
#   Run scripts/select_cluster_representatives.py first to generate
#   data/dpo/representative_pdbs/{train,valid,test}/

set -e

eval "$(conda shell.bash hook)"
conda activate SaProt

PROAGG_CKPT=${1:?"Usage: $0 <proagg_ckpt> [device]"}
DEVICE=${2:-"cuda:3"}

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

PDB_TRAIN="data/dpo/representative_pdbs/train"
PDB_VALID="data/dpo/representative_pdbs/valid"
PDB_TEST="data/dpo/representative_pdbs/test"

TRAIN_PAIRS="data/dpo/train_pairs.pt"
VAL_PAIRS="data/dpo/val_pairs.pt"
DPO_OUTPUT="results/dpo"

NUM_SAMPLES=12
TEMPERATURE=0.5          # Used for both training pair generation AND evaluation
PROAGG_CONFIG="configs/default.yaml"

DPO_EPOCHS=10
DPO_LR=1e-5
DPO_BETA=0.5             # KL penalty; higher = less reward hacking (was 0.1)
DPO_BATCH_SIZE=32
DPO_PATIENCE=3
DPO_SCORE_GAP_DELTA=0.05 # Filter noisy pairs at training time (no re-sampling needed)

for dir in "$PDB_TRAIN" "$PDB_VALID" "$PDB_TEST"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: $dir does not exist."
        echo "Run: python scripts/select_cluster_representatives.py first."
        exit 1
    fi
done

echo "============================================================"
echo "Step 1/4: Sample & score on TRAIN clusters"
echo "============================================================"
python src/dpo/sample_and_score.py \
    --pdb_dir "$PDB_TRAIN" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config "$PROAGG_CONFIG" \
    --output "$TRAIN_PAIRS" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "Step 2/4: Sample & score on VALID clusters"
echo "============================================================"
python src/dpo/sample_and_score.py \
    --pdb_dir "$PDB_VALID" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config "$PROAGG_CONFIG" \
    --output "$VAL_PAIRS" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "Step 3/4: DPO fine-tuning (best on val)"
echo "============================================================"
python src/dpo/dpo_train.py \
    --pairs_path "$TRAIN_PAIRS" \
    --val_pairs_path "$VAL_PAIRS" \
    --output_dir "$DPO_OUTPUT" \
    --device "$DEVICE" \
    --epochs "$DPO_EPOCHS" \
    --lr "$DPO_LR" \
    --beta "$DPO_BETA" \
    --batch_size "$DPO_BATCH_SIZE" \
    --patience "$DPO_PATIENCE" \
    --score_gap_delta "$DPO_SCORE_GAP_DELTA"

BEST_CKPT="$DPO_OUTPUT/mpnn_dpo_best.pt"
if [ ! -f "$BEST_CKPT" ]; then
    echo "WARNING: best checkpoint not found, falling back to latest epoch checkpoint."
    BEST_CKPT=$(ls -t "$DPO_OUTPUT"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
fi

echo ""
echo "============================================================"
echo "Step 4/4: Evaluate on TEST clusters"
echo "============================================================"
python src/dpo/evaluate.py \
    --pdb_dir "$PDB_TEST" \
    --dpo_mpnn_ckpt "$BEST_CKPT" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --output "$DPO_OUTPUT/eval_results.json" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Train pairs:  $TRAIN_PAIRS"
echo "  Val pairs:    $VAL_PAIRS"
echo "  Best ckpt:    $BEST_CKPT"
echo "  Test results: $DPO_OUTPUT/eval_results.json"
echo "============================================================"
