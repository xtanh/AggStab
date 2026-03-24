#!/bin/bash
# ============================================================
# Evaluate a set of already-trained DPO checkpoints from an
# epoch sweep on the common full test set.
#
# Usage:
#   bash scripts/run_dpo_epoch_eval_sweep.sh <proagg_ckpt> [device]
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
PDB_TEST=${PDB_TEST:-"data/dpo/representative_pdbs/test"}
MAX_TEST_PDBS=${MAX_TEST_PDBS:--1}
NUM_SAMPLES=${NUM_SAMPLES:-16}
TEMPERATURE=${TEMPERATURE:-0.5}
PROAGG_BATCH_SIZE=${PROAGG_BATCH_SIZE:-32}
SEED=${SEED:-42}

EPOCHS_LIST=${EPOCHS_LIST:-"12 15 20 25"}
RUN_PREFIX=${RUN_PREFIX:-"dpo_v92_thbal_m1_tr2866_n16_b01_gap010"}

if [ ! -d "$PDB_TEST" ]; then
  echo "ERROR: test directory not found: $PDB_TEST"
  exit 1
fi

echo "============================================================"
echo "DPO epoch eval sweep"
echo "  test dir: $PDB_TEST"
echo "  epochs:   $EPOCHS_LIST"
echo "============================================================"

for EPOCHS in $EPOCHS_LIST; do
  TRAIN_DIR="results/${RUN_PREFIX}_e${EPOCHS}"
  BEST_CKPT="${TRAIN_DIR}/mpnn_dpo_best.pt"
  EVAL_DIR="${TRAIN_DIR}_fulltest"

  if [ ! -f "$BEST_CKPT" ]; then
    echo "WARNING: missing checkpoint, skipping: $BEST_CKPT"
    continue
  fi

  mkdir -p "$EVAL_DIR"

  echo ""
  echo "============================================================"
  echo "Evaluating epoch ${EPOCHS}"
  echo "  ckpt: $BEST_CKPT"
  echo "  out:  $EVAL_DIR"
  echo "============================================================"

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
echo "Epoch evaluation sweep complete."
echo "============================================================"
