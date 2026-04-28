#!/usr/bin/env bash

set -euo pipefail

PDB_NAME="${1:?Usage: $0 <pdb_name> <topk_rank> [gpu_id] [simulation_time_ns] [python_bin]}"
TOPK_RANK="${2:?Usage: $0 <pdb_name> <topk_rank> [gpu_id] [simulation_time_ns] [python_bin]}"
GPU_ID="${3:-0}"
SIM_NS="${4:-200}"
PYTHON_BIN="${5:-/home/xy_th/miniconda3/envs/md/bin/python}"

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

INPUT_CIF="results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/chai_top3_joint/${PDB_NAME}__top${TOPK_RANK}/best.cif"
if [ ! -f "$INPUT_CIF" ]; then
  echo "Missing input CIF: $INPUT_CIF" >&2
  exit 1
fi

CASE_DIRS=(
  "results/md/${PDB_NAME}_case"
  "results/md/${PDB_NAME}_case_rep2"
  "results/md/${PDB_NAME}_case_rep3"
)
SEEDS=(42 43 44)

for idx in 0 1 2; do
  out_dir="${CASE_DIRS[$idx]}"
  seed="${SEEDS[$idx]}"
  mkdir -p "$out_dir"

  echo "Launching MoMPNN MD replicate $((idx + 1))"
  echo "  input:  $INPUT_CIF"
  echo "  output: ${out_dir}/mompnn"
  echo "  seed:   $seed"
  echo "  gpu:    $GPU_ID"

  nohup bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} ${PROJ_DIR}/scripts/md.py --input ${PROJ_DIR}/${INPUT_CIF} --output ${PROJ_DIR}/${out_dir}/mompnn --time ${SIM_NS} --seed ${seed}" \
    > "${PROJ_DIR}/${out_dir}/mompnn.log" 2>&1 &
done

echo "Submitted 3 MoMPNN MD runs for ${PDB_NAME} top${TOPK_RANK}."
echo "Logs:"
echo "  ${PROJ_DIR}/results/md/${PDB_NAME}_case/mompnn.log"
echo "  ${PROJ_DIR}/results/md/${PDB_NAME}_case_rep2/mompnn.log"
echo "  ${PROJ_DIR}/results/md/${PDB_NAME}_case_rep3/mompnn.log"
