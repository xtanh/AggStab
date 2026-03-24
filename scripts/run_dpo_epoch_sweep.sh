#!/bin/bash
# ============================================================
# Run DPO epoch sweep with fixed train/val pairs, then evaluate
# each resulting checkpoint on a common test set.
#
# Usage:
#   bash scripts/run_dpo_epoch_sweep.sh <proagg_ckpt> [device]
#
# Required env vars are optional if defaults below are suitable.
# ============================================================

set -e

eval "$(conda shell.bash hook)"
conda activate SaProt

PROAGG_CKPT=${PROAGG_CKPT:-${1:-""}}
DEVICE=${DEVICE:-${2:-"cuda:3"}}

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

if [[ -z "$PROAGG_CKPT" ]]; then
  echo "Usage: $0 <proagg_ckpt> [device]"
  exit 1
fi

if [[ -d "$PROAGG_CKPT" ]]; then
  RESOLVED_CKPT=$(ls -t "$PROAGG_CKPT"/best_epoch=*.ckpt "$PROAGG_CKPT"/*.ckpt 2>/dev/null | head -1)
  if [[ -z "$RESOLVED_CKPT" ]]; then
    echo "ERROR: checkpoint directory does not contain a .ckpt file: $PROAGG_CKPT"
    exit 1
  fi
  echo "Resolved checkpoint directory to: $RESOLVED_CKPT"
  PROAGG_CKPT="$RESOLVED_CKPT"
fi

if [[ ! -f "$PROAGG_CKPT" ]]; then
  echo "ERROR: ProAgg checkpoint not found: $PROAGG_CKPT"
  exit 1
fi

PROAGG_CONFIG=${PROAGG_CONFIG:-"configs/proagg_final_candidate.yaml"}
SEED=${SEED:-42}
PROAGG_BATCH_SIZE=${PROAGG_BATCH_SIZE:-32}

TRAIN_PAIRS=${TRAIN_PAIRS:-"data/dpo/v92_thbal_m1_tr2866_te100_n16_b01_gap010_train_pairs.pt"}
VAL_PAIRS=${VAL_PAIRS:-"data/dpo/v92_thbal_m1_tr2866_te100_n16_b01_gap010_val_pairs.pt"}

PDB_TEST=${PDB_TEST:-"data/dpo/representative_pdbs/test"}
MAX_TEST_PDBS=${MAX_TEST_PDBS:--1}

NUM_SAMPLES=${NUM_SAMPLES:-16}
TEMPERATURE=${TEMPERATURE:-0.5}
DPO_LR=${DPO_LR:-1e-5}
DPO_BETA=${DPO_BETA:-0.1}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-32}
DPO_PATIENCE=${DPO_PATIENCE:-3}
DPO_SCORE_GAP_DELTA=${DPO_SCORE_GAP_DELTA:-0.10}

EPOCHS_LIST=${EPOCHS_LIST:-"12 15 20 25"}
RUN_PREFIX=${RUN_PREFIX:-"dpo_v92_thbal_m1_tr2866_n16_b01_gap010"}

for req in "$TRAIN_PAIRS" "$VAL_PAIRS"; do
  if [ ! -f "$req" ]; then
    echo "ERROR: required file not found: $req"
    exit 1
  fi
done

if [ ! -d "$PDB_TEST" ]; then
  echo "ERROR: test directory not found: $PDB_TEST"
  exit 1
fi

echo "============================================================"
echo "DPO epoch sweep"
echo "  train pairs: $TRAIN_PAIRS"
echo "  val pairs:   $VAL_PAIRS"
echo "  test dir:    $PDB_TEST"
echo "  epochs:      $EPOCHS_LIST"
echo "============================================================"

for EPOCHS in $EPOCHS_LIST; do
  OUTPUT_DIR="results/${RUN_PREFIX}_e${EPOCHS}"
  EVAL_DIR="${OUTPUT_DIR}_fulltest"

  echo ""
  echo "============================================================"
  echo "Epoch sweep item: e${EPOCHS}"
  echo "  train output: $OUTPUT_DIR"
  echo "  eval output:  $EVAL_DIR"
  echo "============================================================"

  python src/dpo/dpo_train.py \
    --pairs_path "$TRAIN_PAIRS" \
    --val_pairs_path "$VAL_PAIRS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --lr "$DPO_LR" \
    --beta "$DPO_BETA" \
    --batch_size "$DPO_BATCH_SIZE" \
    --patience "$DPO_PATIENCE" \
    --score_gap_delta "$DPO_SCORE_GAP_DELTA" \
    --seed "$SEED"

  BEST_CKPT="$OUTPUT_DIR/mpnn_dpo_best.pt"
  if [ ! -f "$BEST_CKPT" ]; then
    echo "WARNING: best checkpoint not found, falling back to latest epoch checkpoint."
    BEST_CKPT=$(ls -t "$OUTPUT_DIR"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
  fi

  mkdir -p "$EVAL_DIR"

  python src/dpo/evaluate.py \
    --pdb_dir "$PDB_TEST" \
    --dpo_mpnn_ckpt "$BEST_CKPT" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config "$PROAGG_CONFIG" \
    --output "$EVAL_DIR/eval_results.json" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --proagg_batch_size "$PROAGG_BATCH_SIZE" \
    --max_pdbs "$MAX_TEST_PDBS" \
    --device "$DEVICE" \
    --seed "$SEED"

  python scripts/analyze_dpo_eval.py "$EVAL_DIR/eval_results.json" --output "$EVAL_DIR"
  python scripts/analyze_dpo_v2_pathology.py --eval_json "$EVAL_DIR/eval_results.json" --summary_csv "$EVAL_DIR/dpo_eval_summary.csv"
done

echo ""
echo "============================================================"
echo "Epoch sweep complete."
echo "============================================================"
