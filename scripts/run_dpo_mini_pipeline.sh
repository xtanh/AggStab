#!/bin/bash
# ============================================================
# Mini DPO pipeline for fast debugging on a small subset of PDBs.
#
# Default:
#   train: 80 backbones
#   valid: 10 backbones
#   test : 10 backbones
#
# This is intended for quickly verifying code changes / diagnosing failure
# modes (e.g., collapse), not for reporting final numbers.
#
# Usage:
#   bash scripts/run_dpo_mini_pipeline.sh <proagg_ckpt> [device]
# ============================================================

set -e

eval "$(conda shell.bash hook)"
conda activate SaProt

PROAGG_CKPT=${1:?"Usage: $0 <proagg_ckpt> [device]"}
DEVICE=${2:-"cuda:3"}

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

if [[ "$DEVICE" == cuda:* ]]; then
  CUDA_OK=$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)
  if [[ "$CUDA_OK" != "1" ]]; then
    echo "WARNING: torch.cuda.is_available() is False; falling back to CPU for this run." >&2
    DEVICE="cpu"
  fi
fi

PDB_TRAIN=${PDB_TRAIN:-"data/dpo/representative_pdbs/train"}
PDB_VALID=${PDB_VALID:-"data/dpo/representative_pdbs/valid"}
PDB_TEST=${PDB_TEST:-"data/dpo/representative_pdbs/test"}

DEFAULT_PDB_ROOT="data/dpo/representative_pdbs"
SOURCE_TAG="reps"
if [[ "$PDB_TRAIN" != "${DEFAULT_PDB_ROOT}/train" || "$PDB_VALID" != "${DEFAULT_PDB_ROOT}/valid" || "$PDB_TEST" != "${DEFAULT_PDB_ROOT}/test" ]]; then
  SOURCE_TAG=$(basename "$(dirname "$PDB_TRAIN")")
fi

# Subset sizes
MAX_TRAIN_PDBS=${MAX_TRAIN_PDBS:-80}
MAX_VALID_PDBS=${MAX_VALID_PDBS:-10}
MAX_TEST_PDBS=${MAX_TEST_PDBS:-10}

# Sampling / eval
NUM_SAMPLES=${NUM_SAMPLES:-12}
TEMPERATURE=${TEMPERATURE:-0.5}
PROAGG_CONFIG=${PROAGG_CONFIG:-"configs/default.yaml"}
SEED=${SEED:-42}
PROAGG_BATCH_SIZE=${PROAGG_BATCH_SIZE:-16}

# DPO training
DPO_EPOCHS=${DPO_EPOCHS:-5}
DPO_LR=${DPO_LR:-1e-5}
DPO_BETA=${DPO_BETA:-0.1}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-32}
DPO_PATIENCE=${DPO_PATIENCE:-2}
DPO_SCORE_GAP_DELTA=${DPO_SCORE_GAP_DELTA:-0.05}

RUN_TAG=${RUN_TAG:-"${SOURCE_TAG}_beta${DPO_BETA}_e${DPO_EPOCHS}_n${NUM_SAMPLES}_t${TEMPERATURE}_tr${MAX_TRAIN_PDBS}_va${MAX_VALID_PDBS}_te${MAX_TEST_PDBS}"}

TRAIN_PAIRS=${TRAIN_PAIRS:-"data/dpo/${RUN_TAG}_train_pairs.pt"}
VAL_PAIRS=${VAL_PAIRS:-"data/dpo/${RUN_TAG}_val_pairs.pt"}
DPO_OUTPUT=${DPO_OUTPUT:-"results/dpo_${RUN_TAG}"}

for dir in "$PDB_TRAIN" "$PDB_VALID" "$PDB_TEST"; do
  if [ ! -d "$dir" ]; then
    echo "ERROR: $dir does not exist."
    echo "Run: python scripts/select_cluster_representatives.py first."
    exit 1
  fi
done

echo "============================================================"
echo "Mini DPO pipeline"
echo "  train/valid/test backbones: $MAX_TRAIN_PDBS / $MAX_VALID_PDBS / $MAX_TEST_PDBS"
echo "  N=$NUM_SAMPLES  T=$TEMPERATURE  beta=$DPO_BETA"
echo "  output: $DPO_OUTPUT"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1/5: Sample & score on TRAIN (subset)"
echo "============================================================"
python src/dpo/sample_and_score.py \
  --pdb_dir "$PDB_TRAIN" \
  --proagg_ckpt "$PROAGG_CKPT" \
  --proagg_config "$PROAGG_CONFIG" \
  --output "$TRAIN_PAIRS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --proagg_batch_size "$PROAGG_BATCH_SIZE" \
  --max_pdbs "$MAX_TRAIN_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 2/5: Sample & score on VALID (subset)"
echo "============================================================"
python src/dpo/sample_and_score.py \
  --pdb_dir "$PDB_VALID" \
  --proagg_ckpt "$PROAGG_CKPT" \
  --proagg_config "$PROAGG_CONFIG" \
  --output "$VAL_PAIRS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --proagg_batch_size "$PROAGG_BATCH_SIZE" \
  --max_pdbs "$MAX_VALID_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 3/5: DPO fine-tuning"
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
  --score_gap_delta "$DPO_SCORE_GAP_DELTA" \
  --seed "$SEED"

BEST_CKPT="$DPO_OUTPUT/mpnn_dpo_best.pt"
if [ ! -f "$BEST_CKPT" ]; then
  echo "WARNING: best checkpoint not found, falling back to latest epoch checkpoint."
  BEST_CKPT=$(ls -t "$DPO_OUTPUT"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
fi

echo ""
echo "============================================================"
echo "Step 4/5: Evaluate on TEST (subset)"
echo "============================================================"
python src/dpo/evaluate.py \
  --pdb_dir "$PDB_TEST" \
  --dpo_mpnn_ckpt "$BEST_CKPT" \
  --proagg_ckpt "$PROAGG_CKPT" \
  --proagg_config "$PROAGG_CONFIG" \
  --output "$DPO_OUTPUT/eval_results.json" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --proagg_batch_size "$PROAGG_BATCH_SIZE" \
  --max_pdbs "$MAX_TEST_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 5/5: Summaries"
echo "============================================================"
python scripts/analyze_dpo_eval.py "$DPO_OUTPUT/eval_results.json" --output "$DPO_OUTPUT"
python scripts/analyze_dpo_v2_pathology.py --eval_json "$DPO_OUTPUT/eval_results.json" --summary_csv "$DPO_OUTPUT/dpo_eval_summary.csv"

echo ""
echo "============================================================"
echo "Mini pipeline complete!"
echo "  Train pairs:  $TRAIN_PAIRS"
echo "  Val pairs:    $VAL_PAIRS"
echo "  Best ckpt:    $BEST_CKPT"
echo "  Eval results: $DPO_OUTPUT/eval_results.json"
echo "============================================================"
