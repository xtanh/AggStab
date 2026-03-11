#!/bin/bash
# ============================================================
# Reproducible DPO beta sweep:
#   1) Generate train/val preference pairs ONCE (fixed seed)
#   2) Train DPO for multiple betas on the same pairs
#   3) Evaluate + summarize each run
#
# Intended for finding a Pareto point: max@N vs pathology vs logprob.
#
# Usage:
#   bash scripts/run_dpo_beta_sweep.sh <proagg_ckpt> [device]
#
# Notes:
#   - Run inside an environment with torch/transformers available (e.g. SaProt).
#   - Uses SEED for reproducible comparisons.
# ============================================================

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

# Scale knobs
MAX_TRAIN_PDBS=${MAX_TRAIN_PDBS:-400}
MAX_VALID_PDBS=${MAX_VALID_PDBS:-50}
MAX_TEST_PDBS=${MAX_TEST_PDBS:-50}

# Sampling / eval
NUM_SAMPLES=${NUM_SAMPLES:-64}
TEMPERATURE=${TEMPERATURE:-0.5}
PROAGG_CONFIG=${PROAGG_CONFIG:-"configs/default.yaml"}

# DPO training
DPO_EPOCHS=${DPO_EPOCHS:-10}
DPO_LR=${DPO_LR:-1e-5}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-32}
DPO_PATIENCE=${DPO_PATIENCE:-3}
DPO_SCORE_GAP_DELTA=${DPO_SCORE_GAP_DELTA:-0.05}

# Sweep
BETAS=${BETAS:-"0.05 0.1 0.2 0.5"}

# Repro
SEED=${SEED:-42}

RUN_TAG_BASE=${RUN_TAG_BASE:-"sweep_tr${MAX_TRAIN_PDBS}_va${MAX_VALID_PDBS}_te${MAX_TEST_PDBS}_n${NUM_SAMPLES}_t${TEMPERATURE}_e${DPO_EPOCHS}_seed${SEED}"}

TRAIN_PAIRS="data/dpo/${RUN_TAG_BASE}_train_pairs.pt"
VAL_PAIRS="data/dpo/${RUN_TAG_BASE}_val_pairs.pt"

SWEEP_DIR="results/dpo_${RUN_TAG_BASE}"
mkdir -p "$SWEEP_DIR"

for dir in "$PDB_TRAIN" "$PDB_VALID" "$PDB_TEST"; do
  if [ ! -d "$dir" ]; then
    echo "ERROR: $dir does not exist."
    echo "Run: python scripts/select_cluster_representatives.py first."
    exit 1
  fi
done

echo "============================================================"
echo "DPO beta sweep"
echo "  base:  $SWEEP_DIR"
echo "  betas: $BETAS"
echo "  train/valid/test backbones: $MAX_TRAIN_PDBS / $MAX_VALID_PDBS / $MAX_TEST_PDBS"
echo "  N=$NUM_SAMPLES  T=$TEMPERATURE  epochs=$DPO_EPOCHS  seed=$SEED"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1/3: Generate TRAIN/VAL pairs once"
echo "============================================================"
python src/dpo/sample_and_score.py \
  --pdb_dir "$PDB_TRAIN" \
  --proagg_ckpt "$PROAGG_CKPT" \
  --proagg_config "$PROAGG_CONFIG" \
  --output "$TRAIN_PAIRS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --max_pdbs "$MAX_TRAIN_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

python src/dpo/sample_and_score.py \
  --pdb_dir "$PDB_VALID" \
  --proagg_ckpt "$PROAGG_CKPT" \
  --proagg_config "$PROAGG_CONFIG" \
  --output "$VAL_PAIRS" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --max_pdbs "$MAX_VALID_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 2/3: Train+Eval each beta"
echo "============================================================"

for beta in $BETAS; do
  RUN_TAG="${RUN_TAG_BASE}_beta${beta}"
  OUT_DIR="results/dpo_${RUN_TAG}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "------------------------------"
  echo "beta=$beta -> $OUT_DIR"
  echo "------------------------------"

  python src/dpo/dpo_train.py \
    --pairs_path "$TRAIN_PAIRS" \
    --val_pairs_path "$VAL_PAIRS" \
    --output_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --epochs "$DPO_EPOCHS" \
    --lr "$DPO_LR" \
    --beta "$beta" \
    --batch_size "$DPO_BATCH_SIZE" \
    --patience "$DPO_PATIENCE" \
    --score_gap_delta "$DPO_SCORE_GAP_DELTA" \
    --seed "$SEED"

  BEST_CKPT="$OUT_DIR/mpnn_dpo_best.pt"
  if [ ! -f "$BEST_CKPT" ]; then
    echo "WARNING: best checkpoint not found, falling back to latest epoch checkpoint."
    BEST_CKPT=$(ls -t "$OUT_DIR"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
  fi

  python src/dpo/evaluate.py \
    --pdb_dir "$PDB_TEST" \
    --dpo_mpnn_ckpt "$BEST_CKPT" \
    --proagg_ckpt "$PROAGG_CKPT" \
    --proagg_config "$PROAGG_CONFIG" \
    --output "$OUT_DIR/eval_results.json" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max_pdbs "$MAX_TEST_PDBS" \
    --device "$DEVICE" \
    --seed "$SEED"

  python scripts/analyze_dpo_eval.py "$OUT_DIR/eval_results.json" --output "$OUT_DIR"
  python scripts/analyze_dpo_v2_pathology.py --eval_json "$OUT_DIR/eval_results.json" --summary_csv "$OUT_DIR/dpo_eval_summary.csv"
done

echo ""
echo "============================================================"
echo "Step 3/3: Build sweep summary"
echo "============================================================"
python scripts/summarize_dpo_beta_sweep.py --run_tag_base "$RUN_TAG_BASE"

echo ""
echo "============================================================"
echo "Sweep complete"
echo "  base: $SWEEP_DIR"
echo "============================================================"
