#!/bin/bash
# DPO Pipeline with Batched Training (GPU-efficient version)
# Usage: bash scripts/run_dpo_pipeline_batched.sh <subset_name> [options]
#
# Examples:
#   bash scripts/run_dpo_pipeline_batched.sh stratified_tr2000_va300_te600
#   bash scripts/run_dpo_pipeline_batched.sh stratified_tr4000_va600_te1200 --beta 0.2 --epochs 10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
SUBSET_NAME="${1:-stratified_tr2000_va300_te600}"
shift || true

# Default parameters
BETA="${BETA:-0.2}"
EPOCHS="${EPOCHS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
TEMPERATURE="${TEMPERATURE:-0.5}"
DEVICE="${DEVICE:-cuda:3}"
SCORE_GAP_DELTA="${SCORE_GAP_DELTA:-0.05}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --beta) BETA="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --accum_steps) ACCUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Paths
SUBSET_DIR="data/dpo/subsets/${SUBSET_NAME}"
TRAIN_PDB_DIR="${SUBSET_DIR}/train"
VALID_PDB_DIR="${SUBSET_DIR}/valid"
TEST_PDB_DIR="${SUBSET_DIR}/test"

PAIRS_NAME="train_n${NUM_SAMPLES}_t${TEMPERATURE}"
OUTPUT_DIR="results/${SUBSET_NAME}_beta${BETA}_e${EPOCHS}_n${NUM_SAMPLES}_t${TEMPERATURE}"

# ProAgg checkpoint (update this to your best model)
PROAGG_CKPT="${PROAGG_CKPT:-results/lightning_logs/version_X/checkpoints/best.ckpt}"
PROAGG_CONFIG="${PROAGG_CONFIG:-configs/default.yaml}"

echo "=========================================="
echo "DPO Pipeline (Batched Training)"
echo "=========================================="
echo "Subset:     ${SUBSET_NAME}"
echo "Train PDBs: ${TRAIN_PDB_DIR}"
echo "Valid PDBs: ${VALID_PDB_DIR}"
echo "Test PDBs:  ${TEST_PDB_DIR}"
echo "Beta:       ${BETA}"
echo "Epochs:     ${EPOCHS}"
echo "Samples:    ${NUM_SAMPLES}"
echo "Temp:       ${TEMPERATURE}"
echo "Output:     ${OUTPUT_DIR}"
echo "Accum Steps: ${ACCUM_STEPS}"
echo "=========================================="

# Check paths
if [ ! -d "$TRAIN_PDB_DIR" ]; then
    echo "ERROR: Train PDB directory not found: $TRAIN_PDB_DIR"
    echo "Please run: bash scripts/create_large_stratified_subset.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Stage 1: Sample and Score (Training Set)
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 1: Sample and Score (TRAIN)"
echo "=========================================="
TRAIN_PAIRS="${OUTPUT_DIR}/train_pairs.pt"

if [ -f "$TRAIN_PAIRS" ]; then
    echo "Train pairs already exist: $TRAIN_PAIRS"
else
    python src/dpo/sample_and_score.py \
        --pdb_dir "$TRAIN_PDB_DIR" \
        --proagg_ckpt "$PROAGG_CKPT" \
        --proagg_config "$PROAGG_CONFIG" \
        --output "$TRAIN_PAIRS" \
        --num_samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --score_gap_delta "$SCORE_GAP_DELTA" \
        --device "$DEVICE"
fi

# ============================================================================
# Stage 2: Sample and Score (Validation Set)
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 2: Sample and Score (VALID)"
echo "=========================================="
VALID_PAIRS="${OUTPUT_DIR}/val_pairs.pt"

if [ -f "$VALID_PAIRS" ]; then
    echo "Validation pairs already exist: $VALID_PAIRS"
else
    python src/dpo/sample_and_score.py \
        --pdb_dir "$VALID_PDB_DIR" \
        --proagg_ckpt "$PROAGG_CKPT" \
        --proagg_config "$PROAGG_CONFIG" \
        --output "$VALID_PAIRS" \
        --num_samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --score_gap_delta "$SCORE_GAP_DELTA" \
        --device "$DEVICE"
fi

# Count pairs
echo ""
echo "Pair counts:"
python3 -c "
import torch
train = torch.load('$TRAIN_PAIRS')
val = torch.load('$VALID_PAIRS')
print(f'  Train pairs: {len(train[\"pairs\"])}')
print(f'  Valid pairs: {len(val[\"pairs\"])}')
"

# ============================================================================
# Stage 3: DPO Training (Batched)
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 3: DPO Training (Batched)"
echo "=========================================="

python src/dpo/dpo_train_batched.py \
    --pairs_path "$TRAIN_PAIRS" \
    --val_pairs_path "$VALID_PAIRS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --lr 1e-5 \
    --beta "$BETA" \
    --score_gap_delta 0.0 \
    --accum_steps "$ACCUM_STEPS" \
    --patience 3 \
    --seed 42

# ============================================================================
# Stage 4: Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 4: Evaluation"
echo "=========================================="

EVAL_JSON="${OUTPUT_DIR}/eval_results.json"
TEST_PDB_COUNT=$(find "$TEST_PDB_DIR" -name "*.pdb" | wc -l)

echo "Evaluating on ${TEST_PDB_COUNT} test PDBs..."

python src/dpo/evaluate.py \
    --pdb_dir "$TEST_PDB_DIR" \
    --dpo_mpnn_ckpt "${OUTPUT_DIR}/mpnn_dpo_best.pt" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config "$PROAGG_CONFIG" \
    --output "$EVAL_JSON" \
    --num_samples 12 \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --seed 42

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation results: $EVAL_JSON"
echo ""
echo "Key files:"
echo "  - Best model: ${OUTPUT_DIR}/mpnn_dpo_best.pt"
echo "  - Training history: ${OUTPUT_DIR}/training_history.json"
echo "  - Evaluation: $EVAL_JSON"

# Print quick summary
if [ -f "$EVAL_JSON" ]; then
    echo ""
    echo "Quick Results:"
    python3 -c "
import json
with open('$EVAL_JSON') as f:
    data = json.load(f)
orig = data['original']
dpo = data['dpo']
mean_orig = sum(x['proagg_mean'] for x in orig) / len(orig)
mean_dpo = sum(x['proagg_mean'] for x in dpo) / len(dpo)
max_orig = sum(x['proagg_max'] for x in orig) / len(orig)
max_dpo = sum(x['proagg_max'] for x in dpo) / len(dpo)
print(f'  ProAgg mean: {mean_orig:.4f} -> {mean_dpo:.4f} ({mean_dpo-mean_orig:+.4f})')
print(f'  ProAgg max:  {max_orig:.4f} -> {max_dpo:.4f} ({max_dpo-max_orig:+.4f})')
"
fi
