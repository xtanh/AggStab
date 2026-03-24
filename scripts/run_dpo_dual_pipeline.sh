#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate SaProt

AGG_CKPT=${1:?"Usage: $0 <agg_ckpt> <stab_ckpt> [device]"}
STAB_CKPT=${2:?"Usage: $0 <agg_ckpt> <stab_ckpt> [device]"}
DEVICE=${3:-"cuda:0"}

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

PDB_TRAIN=${PDB_TRAIN:-"data/dpo/representative_pdbs/train"}
PDB_VALID=${PDB_VALID:-"data/dpo/representative_pdbs/valid"}
PDB_TEST=${PDB_TEST:-"data/dpo/representative_pdbs/test"}

MAX_TRAIN_PDBS=${MAX_TRAIN_PDBS:-80}
MAX_VALID_PDBS=${MAX_VALID_PDBS:-10}
MAX_TEST_PDBS=${MAX_TEST_PDBS:-10}

NUM_SAMPLES=${NUM_SAMPLES:-16}
TEMPERATURE=${TEMPERATURE:-0.5}
SEED=${SEED:-42}
PREDICTOR_BATCH_SIZE=${PREDICTOR_BATCH_SIZE:-16}

AGG_CONFIG=${AGG_CONFIG:-"configs/proagg_final_candidate.yaml"}
STAB_CONFIG=${STAB_CONFIG:-"configs/proagg_deltaG_only.yaml"}
STABILITY_CSV=${STABILITY_CSV:-"data/rocklin/Metagenomic_dG.csv"}

DPO_EPOCHS=${DPO_EPOCHS:-5}
DPO_LR=${DPO_LR:-1e-5}
DPO_BETA=${DPO_BETA:-0.1}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-32}
DPO_PATIENCE=${DPO_PATIENCE:-2}
DUAL_AGG_RATIO=${DUAL_AGG_RATIO:-1}
DUAL_STAB_RATIO=${DUAL_STAB_RATIO:-1}
AGG_SCORE_GAP_DELTA=${AGG_SCORE_GAP_DELTA:-0.10}
STAB_SCORE_GAP_DELTA=${STAB_SCORE_GAP_DELTA:-0.10}
STABILITY_GATE_MODE=${STABILITY_GATE_MODE:-"absolute"}
STABILITY_GATE_MIN=${STABILITY_GATE_MIN:-0.0}
STABILITY_GATE_MARGIN=${STABILITY_GATE_MARGIN:-0.5}

RUN_TAG=${RUN_TAG:-"dual_beta${DPO_BETA}_e${DPO_EPOCHS}_n${NUM_SAMPLES}_agg${AGG_SCORE_GAP_DELTA}_stab${STAB_SCORE_GAP_DELTA}_gate${STABILITY_GATE_MODE}"}

TRAIN_DUAL=${TRAIN_DUAL:-"data/dpo/${RUN_TAG}_train_dual.pt"}
VAL_DUAL=${VAL_DUAL:-"data/dpo/${RUN_TAG}_val_dual.pt"}
DPO_OUTPUT=${DPO_OUTPUT:-"results/dpo_${RUN_TAG}"}
EVAL_JSON=${EVAL_JSON:-"${DPO_OUTPUT}/eval_results.json"}

echo "============================================================"
echo "Dual-objective DPO pipeline"
echo "  train/valid/test backbones: $MAX_TRAIN_PDBS / $MAX_VALID_PDBS / $MAX_TEST_PDBS"
echo "  N=$NUM_SAMPLES  T=$TEMPERATURE  beta=$DPO_BETA"
echo "  dual ratios agg:stab = $DUAL_AGG_RATIO:$DUAL_STAB_RATIO"
echo "  agg_gap=$AGG_SCORE_GAP_DELTA  stab_gap=$STAB_SCORE_GAP_DELTA"
echo "  stability_gate=$STABILITY_GATE_MODE"
echo "  output: $DPO_OUTPUT"
echo "============================================================"

echo ""
echo "============================================================"
echo "Step 1/4: Build dual-objective train pairs"
echo "============================================================"
python src/dpo/sample_and_score_dual.py \
  --pdb_dir "$PDB_TRAIN" \
  --agg_ckpt "$AGG_CKPT" \
  --agg_config "$AGG_CONFIG" \
  --stab_ckpt "$STAB_CKPT" \
  --stab_config "$STAB_CONFIG" \
  --stability_csv "$STABILITY_CSV" \
  --output "$TRAIN_DUAL" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --agg_score_gap_delta "$AGG_SCORE_GAP_DELTA" \
  --stab_score_gap_delta "$STAB_SCORE_GAP_DELTA" \
  --stability_gate_mode "$STABILITY_GATE_MODE" \
  --stability_gate_min "$STABILITY_GATE_MIN" \
  --stability_gate_margin "$STABILITY_GATE_MARGIN" \
  --predictor_batch_size "$PREDICTOR_BATCH_SIZE" \
  --max_pdbs "$MAX_TRAIN_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 2/4: Build dual-objective valid pairs"
echo "============================================================"
python src/dpo/sample_and_score_dual.py \
  --pdb_dir "$PDB_VALID" \
  --agg_ckpt "$AGG_CKPT" \
  --agg_config "$AGG_CONFIG" \
  --stab_ckpt "$STAB_CKPT" \
  --stab_config "$STAB_CONFIG" \
  --stability_csv "$STABILITY_CSV" \
  --output "$VAL_DUAL" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --agg_score_gap_delta "$AGG_SCORE_GAP_DELTA" \
  --stab_score_gap_delta "$STAB_SCORE_GAP_DELTA" \
  --stability_gate_mode "$STABILITY_GATE_MODE" \
  --stability_gate_min "$STABILITY_GATE_MIN" \
  --stability_gate_margin "$STABILITY_GATE_MARGIN" \
  --predictor_batch_size "$PREDICTOR_BATCH_SIZE" \
  --max_pdbs "$MAX_VALID_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 3/4: Dual-objective DPO fine-tuning"
echo "============================================================"
python src/dpo/dpo_train.py \
  --agg_pairs_path "$TRAIN_DUAL" \
  --stab_pairs_path "$TRAIN_DUAL" \
  --val_agg_pairs_path "$VAL_DUAL" \
  --val_stab_pairs_path "$VAL_DUAL" \
  --output_dir "$DPO_OUTPUT" \
  --device "$DEVICE" \
  --epochs "$DPO_EPOCHS" \
  --lr "$DPO_LR" \
  --beta "$DPO_BETA" \
  --batch_size "$DPO_BATCH_SIZE" \
  --dual_agg_ratio "$DUAL_AGG_RATIO" \
  --dual_stab_ratio "$DUAL_STAB_RATIO" \
  --patience "$DPO_PATIENCE" \
  --agg_score_gap_delta "$AGG_SCORE_GAP_DELTA" \
  --stab_score_gap_delta "$STAB_SCORE_GAP_DELTA" \
  --seed "$SEED"

BEST_CKPT="$DPO_OUTPUT/mpnn_dpo_best.pt"
if [ ! -f "$BEST_CKPT" ]; then
  BEST_CKPT=$(ls -t "$DPO_OUTPUT"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
fi

echo ""
echo "============================================================"
echo "Step 4/4: Evaluate on test backbones"
echo "============================================================"
python src/dpo/evaluate.py \
  --pdb_dir "$PDB_TEST" \
  --dpo_mpnn_ckpt "$BEST_CKPT" \
  --proagg_ckpt "$AGG_CKPT" \
  --proagg_config "$AGG_CONFIG" \
  --stab_ckpt "$STAB_CKPT" \
  --stab_config "$STAB_CONFIG" \
  --output "$EVAL_JSON" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --proagg_batch_size "$PREDICTOR_BATCH_SIZE" \
  --max_pdbs "$MAX_TEST_PDBS" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "Done. Results: $DPO_OUTPUT"
