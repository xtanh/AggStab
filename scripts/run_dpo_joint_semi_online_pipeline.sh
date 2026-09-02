#!/bin/bash

set -euo pipefail

AGG_CKPT=${1:?"Usage: $0 <agg_ckpt> <stab_ckpt> [device]"}
STAB_CKPT=${2:?"Usage: $0 <agg_ckpt> <stab_ckpt> [device]"}
DEVICE=${3:-"cuda:0"}

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

PDB_TRAIN=${PDB_TRAIN:-"data/dpo/representative_pdbs/train"}
PDB_VALID=${PDB_VALID:-"data/dpo/representative_pdbs/valid"}
PDB_TEST=${PDB_TEST:-"data/dpo/representative_pdbs/test"}
SELECT_PDB_VALID=${SELECT_PDB_VALID:-"$PDB_VALID"}

MAX_TRAIN_PDBS=${MAX_TRAIN_PDBS:-80}
MAX_VALID_PDBS=${MAX_VALID_PDBS:-10}
MAX_TEST_PDBS=${MAX_TEST_PDBS:-10}
SELECT_MAX_VALID_PDBS=${SELECT_MAX_VALID_PDBS:-$MAX_VALID_PDBS}

NUM_SAMPLES=${NUM_SAMPLES:-16}
TEMPERATURE=${TEMPERATURE:-0.5}
SEED=${SEED:-42}
PREDICTOR_BATCH_SIZE=${PREDICTOR_BATCH_SIZE:-16}

AGG_CONFIG=${AGG_CONFIG:-"configs/proagg_final_candidate.yaml"}
STAB_CONFIG=${STAB_CONFIG:-"configs/proagg_deltaG_only.yaml"}
STABILITY_CSV=${STABILITY_CSV:-"data/rocklin/Metagenomic_dG.csv"}

ROUNDS=${ROUNDS:-2}
ROUND_EPOCHS=${ROUND_EPOCHS:-2}
DPO_LR=${DPO_LR:-1e-5}
DPO_BETA=${DPO_BETA:-0.1}
DPO_BATCH_SIZE=${DPO_BATCH_SIZE:-32}
DPO_PATIENCE=${DPO_PATIENCE:-2}

AGG_SCORE_GAP_DELTA=${AGG_SCORE_GAP_DELTA:-0.20}
STAB_SCORE_GAP_DELTA=${STAB_SCORE_GAP_DELTA:-0.20}
STABILITY_GATE_MODE=${STABILITY_GATE_MODE:-"wt_absolute"}
STABILITY_GATE_MIN=${STABILITY_GATE_MIN:-0.0}
STABILITY_GATE_MARGIN=${STABILITY_GATE_MARGIN:-1.0}

START_MPNN_CKPT=${START_MPNN_CKPT:-""}
RUN_TAG=${RUN_TAG:-"semi_online_joint_r${ROUNDS}_e${ROUND_EPOCHS}_n${NUM_SAMPLES}"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/${RUN_TAG}"}
ORIGINAL_RESULTS_CACHE=${ORIGINAL_RESULTS_CACHE:-""}
VALID_ORIGINAL_RESULTS_CACHE=${VALID_ORIGINAL_RESULTS_CACHE:-""}
VALID_SELECTION_ENABLED=${VALID_SELECTION_ENABLED:-1}
VALID_SELECTION_METRIC=${VALID_SELECTION_METRIC:-joint_sum}
VALID_SELECTION_MAX_PENALTY=${VALID_SELECTION_MAX_PENALTY:-0.1}
ROUND_TRAIN_SPLIT_MODE=${ROUND_TRAIN_SPLIT_MODE:-"none"}
ROUND_TRAIN_SPLIT_SEED=${ROUND_TRAIN_SPLIT_SEED:-$SEED}

mkdir -p "$OUTPUT_ROOT"

if [ -n "$START_MPNN_CKPT" ]; then
  CURRENT_CKPT="$START_MPNN_CKPT"
else
  CURRENT_CKPT=""
fi

echo "============================================================"
echo "Semi-online joint DPO pipeline"
echo "  rounds=$ROUNDS  round_epochs=$ROUND_EPOCHS"
echo "  train/valid/test backbones: $MAX_TRAIN_PDBS / $MAX_VALID_PDBS / $MAX_TEST_PDBS"
echo "  N=$NUM_SAMPLES  T=$TEMPERATURE  beta=$DPO_BETA"
echo "  agg_gap=$AGG_SCORE_GAP_DELTA  stab_gap=$STAB_SCORE_GAP_DELTA"
echo "  stability_gate=$STABILITY_GATE_MODE  margin=$STABILITY_GATE_MARGIN"
echo "  valid selection enabled=$VALID_SELECTION_ENABLED  valid_pdb_dir=$SELECT_PDB_VALID"
echo "  round train split mode=$ROUND_TRAIN_SPLIT_MODE"
echo "  output_root=$OUTPUT_ROOT"
echo "============================================================"

ROUND_LIST_DIR="${OUTPUT_ROOT}/round_train_lists"
if [ "$ROUND_TRAIN_SPLIT_MODE" = "halves" ] || [ "$ROUND_TRAIN_SPLIT_MODE" = "thirds" ] || [ "$ROUND_TRAIN_SPLIT_MODE" = "quarters" ]; then
  mkdir -p "$ROUND_LIST_DIR"
  python - <<PY
import glob, os, random
pdb_dir = "${PDB_TRAIN}"
output_dir = "${ROUND_LIST_DIR}"
seed = int("${ROUND_TRAIN_SPLIT_SEED}")
rounds = int("${ROUNDS}")
files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
random.Random(seed).shuffle(files)
mode = "${ROUND_TRAIN_SPLIT_MODE}"
expected_rounds = {
    "halves": 2,
    "thirds": 3,
    "quarters": 4,
}
if mode not in expected_rounds:
    raise SystemExit(f"Unsupported ROUND_TRAIN_SPLIT_MODE: {mode}")
num_splits = expected_rounds[mode]
if rounds != num_splits:
    raise SystemExit(f"ROUND_TRAIN_SPLIT_MODE={mode} requires ROUNDS={num_splits}")

base, rem = divmod(len(files), num_splits)
splits = []
start = 0
for i in range(num_splits):
    size = base + (1 if i < rem else 0)
    end = start + size
    splits.append(files[start:end])
    start = end

for i, split in enumerate(splits):
    out = os.path.join(output_dir, f"round{i}.txt")
    with open(out, "w") as f:
        for path in split:
            f.write(os.path.basename(path) + "\\n")
print(f"Wrote split lists to {output_dir} with sizes {[len(s) for s in splits]}")
PY
fi

for (( ROUND=0; ROUND<ROUNDS; ROUND++ )); do
  ROUND_DIR="${OUTPUT_ROOT}/round${ROUND}"
  mkdir -p "$ROUND_DIR"

  TRAIN_PAIRS="${ROUND_DIR}/train_pairs.pt"
  VAL_PAIRS="${ROUND_DIR}/val_pairs.pt"
  DPO_OUTPUT="${ROUND_DIR}/dpo"
  EVAL_JSON="${ROUND_DIR}/eval_results.json"

  echo ""
  echo "============================================================"
  echo "Round ${ROUND}/${ROUNDS}"
  echo "  starting_ckpt=${CURRENT_CKPT:-original_ProteinMPNN}"
  echo "============================================================"

  JOINT_COMMON_ARGS=(
    --agg_ckpt "$AGG_CKPT"
    --agg_config "$AGG_CONFIG"
    --stab_ckpt "$STAB_CKPT"
    --stab_config "$STAB_CONFIG"
    --stability_csv "$STABILITY_CSV"
    --num_samples "$NUM_SAMPLES"
    --temperature "$TEMPERATURE"
    --agg_score_gap_delta "$AGG_SCORE_GAP_DELTA"
    --stab_score_gap_delta "$STAB_SCORE_GAP_DELTA"
    --stability_gate_mode "$STABILITY_GATE_MODE"
    --stability_gate_min "$STABILITY_GATE_MIN"
    --stability_gate_margin "$STABILITY_GATE_MARGIN"
    --predictor_batch_size "$PREDICTOR_BATCH_SIZE"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [ -n "$CURRENT_CKPT" ]; then
    JOINT_COMMON_ARGS+=(--mpnn_ckpt "$CURRENT_CKPT")
  fi

  ROUND_TRAIN_ARGS=()
  if [ "$ROUND_TRAIN_SPLIT_MODE" = "halves" ] || [ "$ROUND_TRAIN_SPLIT_MODE" = "thirds" ] || [ "$ROUND_TRAIN_SPLIT_MODE" = "quarters" ]; then
    ROUND_LIST_FILE="${ROUND_LIST_DIR}/round${ROUND}.txt"
    ROUND_TRAIN_ARGS+=(--pdb_list_file "$ROUND_LIST_FILE")
    echo "  using round-specific train list: $ROUND_LIST_FILE"
  fi

  echo "Step 1: Build round train pairs"
  python src/dpo/sample_and_score_joint.py \
    --pdb_dir "$PDB_TRAIN" \
    --output "$TRAIN_PAIRS" \
    --max_pdbs "$MAX_TRAIN_PDBS" \
    "${ROUND_TRAIN_ARGS[@]}" \
    "${JOINT_COMMON_ARGS[@]}"

  echo "Step 2: Build round val pairs"
  python src/dpo/sample_and_score_joint.py \
    --pdb_dir "$PDB_VALID" \
    --output "$VAL_PAIRS" \
    --max_pdbs "$MAX_VALID_PDBS" \
    "${JOINT_COMMON_ARGS[@]}"

  echo "Step 3: Train round DPO"
  TRAIN_ARGS=(
    --pairs_path "$TRAIN_PAIRS"
    --val_pairs_path "$VAL_PAIRS"
    --output_dir "$DPO_OUTPUT"
    --device "$DEVICE"
    --epochs "$ROUND_EPOCHS"
    --lr "$DPO_LR"
    --beta "$DPO_BETA"
    --batch_size "$DPO_BATCH_SIZE"
    --patience "$DPO_PATIENCE"
    --score_gap_delta "$AGG_SCORE_GAP_DELTA"
    --seed "$SEED"
  )
  if [ -n "$CURRENT_CKPT" ]; then
    TRAIN_ARGS+=(--mpnn_ckpt "$CURRENT_CKPT")
  fi
  python src/dpo/dpo_train.py "${TRAIN_ARGS[@]}"

  BEST_CKPT="${DPO_OUTPUT}/mpnn_dpo_best.pt"
  if [ ! -f "$BEST_CKPT" ]; then
    BEST_CKPT=$(ls -t "$DPO_OUTPUT"/mpnn_dpo_epoch*.pt 2>/dev/null | head -1)
  fi

  if [ "$VALID_SELECTION_ENABLED" = "1" ]; then
    echo "Step 3b: Select best checkpoint on validation set"
    SELECT_DIR="${ROUND_DIR}/valid_epoch_selection"
    SELECT_ARGS=(
      --ckpt_dir "$DPO_OUTPUT"
      --pdb_dir "$SELECT_PDB_VALID"
      --agg_ckpt "$AGG_CKPT"
      --agg_config "$AGG_CONFIG"
      --stab_ckpt "$STAB_CKPT"
      --stab_config "$STAB_CONFIG"
      --output_dir "$SELECT_DIR"
      --num_samples "$NUM_SAMPLES"
      --temperature "$TEMPERATURE"
      --batch_size "$PREDICTOR_BATCH_SIZE"
      --max_pdbs "$SELECT_MAX_VALID_PDBS"
      --device "$DEVICE"
      --seed "$SEED"
      --selection_metric "$VALID_SELECTION_METRIC"
      --max_penalty "$VALID_SELECTION_MAX_PENALTY"
    )
    if [ -n "$VALID_ORIGINAL_RESULTS_CACHE" ]; then
      SELECT_ARGS+=(--original_results_cache "$VALID_ORIGINAL_RESULTS_CACHE")
    fi
    python scripts/select_best_joint_checkpoint.py "${SELECT_ARGS[@]}"
    BEST_CKPT=$(python - <<PY
import json
with open("${SELECT_DIR}/best_checkpoint.json") as f:
    payload=json.load(f)
print(payload["checkpoint"])
PY
)
  fi

  CURRENT_CKPT="$BEST_CKPT"

  echo "Step 4: Evaluate round checkpoint"
  if [ -n "$ORIGINAL_RESULTS_CACHE" ]; then
    python src/dpo/evaluate.py \
      --pdb_dir "$PDB_TEST" \
      --dpo_mpnn_ckpt "$CURRENT_CKPT" \
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
      --seed "$SEED" \
      --original_results_cache "$ORIGINAL_RESULTS_CACHE"
  else
    python src/dpo/evaluate.py \
      --pdb_dir "$PDB_TEST" \
      --dpo_mpnn_ckpt "$CURRENT_CKPT" \
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
  fi

  python scripts/analyze_dual_eval.py "$EVAL_JSON" --output_dir "$ROUND_DIR"
done

printf "final_checkpoint=%s\n" "$CURRENT_CKPT" > "${OUTPUT_ROOT}/final_checkpoint.txt"
echo "Done. Final checkpoint: $CURRENT_CKPT"
