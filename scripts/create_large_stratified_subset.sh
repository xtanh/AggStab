#!/bin/bash
# Create larger stratified DPO subsets
# Usage: bash scripts/create_large_stratified_subset.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Creating Large Stratified DPO Subsets"
echo "=========================================="

# 方案1: 最大规模 (用满所有可用数据)
echo ""
echo "[方案1] 最大规模 - 用满所有可用数据"
python scripts/select_dpo_subset.py \
    --representatives_csv data/dpo/representative_pdbs/representatives.csv \
    --pdb_root data/dpo/representative_pdbs \
    --output_dir data/dpo/subsets/stratified_tr4000_va600_te1200 \
    --train_size 4000 \
    --valid_size 600 \
    --test_size 1200 \
    --seed 42

echo ""
echo "[方案2] 平衡规模 - 兼顾训练效率和覆盖度"
python scripts/select_dpo_subset.py \
    --representatives_csv data/dpo/representative_pdbs/representatives.csv \
    --pdb_root data/dpo/representative_pdbs \
    --output_dir data/dpo/subsets/stratified_tr2000_va300_te600 \
    --train_size 2000 \
    --valid_size 300 \
    --test_size 600 \
    --seed 42

echo ""
echo "[方案3] 中等规模 - 快速迭代用"
python scripts/select_dpo_subset.py \
    --representatives_csv data/dpo/representative_pdbs/representatives.csv \
    --pdb_root data/dpo/representative_pdbs \
    --output_dir data/dpo/subsets/stratified_tr1200_va150_te300 \
    --train_size 1200 \
    --valid_size 150 \
    --test_size 300 \
    --seed 42

echo ""
echo "=========================================="
echo "All subsets created!"
echo "=========================================="
echo ""
echo "Available subsets:"
ls -la data/dpo/subsets/
