"""Select a DPO subset from representative backbones.

The input is expected to be the existing representative set
(`data/dpo/representative_pdbs/representatives.csv`), which already has
cluster-balanced train/valid/test representatives. This script performs a
second-stage subset selection to build a safer DPO training subset:

  - split-preserving
  - one representative per cluster (inherited from representatives.csv)
  - either stratified by sequence length + aggregation label
  - or capped-bin sampling in absolute label space
  - or threshold-balanced sampling around a chosen label cutoff

Output:
  - symlinked PDB subset directories: <output_dir>/{train,valid,test}/
  - manifest CSV with bins and selected status

Usage:
  python scripts/select_dpo_subset.py \
    --representatives_csv data/dpo/representative_pdbs/representatives.csv \
    --pdb_root data/dpo/representative_pdbs \
    --output_dir data/dpo/subsets/stratified_tr500_va100_te100 \
    --train_size 500 --valid_size 100 --test_size 100

  python scripts/select_dpo_subset.py \
    --mode capped_bin \
    --representatives_csv data/dpo/representative_pdbs/representatives.csv \
    --pdb_root data/dpo/representative_pdbs \
    --output_dir data/dpo/subsets/capped_bin_train \
    --bin_width 0.5 \
    --max_per_bin 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LENGTH_BINS = "0,50,60,72"
DEFAULT_LABEL_COL = "log2_fold_change_75_clip"
DEFAULT_LABEL_BINS = 5
DEFAULT_MODE = "stratified"


def parse_bins(spec: str) -> list[float]:
    bins = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(bins) < 2:
        raise ValueError(f"Need at least two bin edges, got: {spec}")
    if sorted(bins) != bins:
        raise ValueError(f"Bin edges must be sorted ascending: {spec}")
    return bins


def assign_label_bins(split_df: pd.DataFrame, label_col: str, n_bins: int) -> pd.Series:
    # qcut may collapse bins when values are duplicated; that's acceptable.
    return pd.qcut(
        split_df[label_col],
        q=n_bins,
        labels=[f"q{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )


def allocate_per_stratum(counts: pd.Series, target_n: int) -> pd.Series:
    """Allocate target_n samples across strata.

    Strategy:
      - if target_n >= n_strata, assign at least 1 to each non-empty stratum
      - allocate the remainder proportionally to stratum size
      - cap by available counts
      - distribute leftover greedily by largest fractional need
    """
    counts = counts[counts > 0].sort_index()
    if target_n >= len(counts):
        base = pd.Series(1, index=counts.index, dtype=int)
        remaining_target = target_n - int(base.sum())
        remaining_counts = counts - base
    else:
        base = pd.Series(0, index=counts.index, dtype=int)
        remaining_target = target_n
        remaining_counts = counts.copy()

    if remaining_target <= 0:
        return base.clip(upper=counts)

    proportions = remaining_counts / remaining_counts.sum()
    raw = proportions * remaining_target
    extra = np.floor(raw).astype(int)
    alloc = (base + extra).clip(upper=counts)

    leftover = target_n - int(alloc.sum())
    if leftover <= 0:
        return alloc

    frac = (raw - np.floor(raw)).sort_values(ascending=False)
    for stratum in frac.index:
        if leftover == 0:
            break
        if alloc[stratum] < counts[stratum]:
            alloc[stratum] += 1
            leftover -= 1

    if leftover > 0:
        slack = (counts - alloc).sort_values(ascending=False)
        for stratum, room in slack.items():
            if leftover == 0:
                break
            if room <= 0:
                continue
            take = min(int(room), leftover)
            alloc[stratum] += take
            leftover -= take

    return alloc


def sample_split(split_df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if target_n >= len(split_df):
        out = split_df.copy()
        out["selected"] = True
        return out

    counts = split_df["stratum"].value_counts().sort_index()
    alloc = allocate_per_stratum(counts, target_n)

    sampled = []
    for idx, (stratum, n_take) in enumerate(alloc.items()):
        if n_take <= 0:
            continue
        group = split_df[split_df["stratum"] == stratum]
        sampled.append(group.sample(n=int(n_take), random_state=seed + idx))

    selected = pd.concat(sampled, axis=0).copy()
    selected["selected"] = True
    return selected


def sample_split_capped_bin(
    split_df: pd.DataFrame,
    label_col: str,
    bin_width: float,
    max_per_bin: int,
    seed: int,
) -> pd.DataFrame:
    values = split_df[label_col]
    min_val = float(np.floor(values.min()))
    max_val = float(np.ceil(values.max()))
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([min_val, min_val + bin_width])

    binned = split_df.copy()
    binned["value_bin"] = pd.cut(binned[label_col], bins=bins, include_lowest=True)

    sampled = []
    non_empty_bins = [bin_name for bin_name, group in binned.groupby("value_bin", observed=False) if len(group) > 0]
    for idx, bin_name in enumerate(non_empty_bins):
        group = binned[binned["value_bin"] == bin_name]
        if len(group) > max_per_bin:
            group = group.sample(n=max_per_bin, random_state=seed + idx)
        sampled.append(group)

    selected = pd.concat(sampled, axis=0).copy()
    selected["selected"] = True
    selected["value_bin"] = selected["value_bin"].astype(str)
    return selected


def sample_split_threshold_balanced(
    split_df: pd.DataFrame,
    label_col: str,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    left = split_df[split_df[label_col] < threshold].copy()
    right_pool = split_df[split_df[label_col] >= threshold].copy()
    if len(left) == 0 or len(right_pool) == 0:
        raise ValueError(
            f"Cannot threshold-balance split with threshold={threshold}: "
            f"left={len(left)}, right={len(right_pool)}"
        )

    n_right = min(len(left), len(right_pool))
    right = right_pool.sample(n=n_right, random_state=seed)
    selected = pd.concat([left, right], axis=0).copy()
    selected["selected"] = True
    selected["threshold_group"] = np.where(selected[label_col] < threshold, "lt_threshold", "ge_threshold")
    return selected


def create_symlinks(selected_df: pd.DataFrame, pdb_root: Path, output_dir: Path) -> None:
    for split in ["train", "valid", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_df = selected_df[selected_df["split"] == split]
        for _, row in split_df.iterrows():
            src = (pdb_root / split / row["pdb_file"]).resolve()
            dst = split_dir / row["pdb_file"]
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a DPO subset.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stratified", "capped_bin", "threshold_balance"],
        default=DEFAULT_MODE,
        help="Subset construction strategy",
    )
    parser.add_argument(
        "--representatives_csv",
        type=str,
        default="data/dpo/representative_pdbs/representatives.csv",
    )
    parser.add_argument(
        "--pdb_root",
        type=str,
        default="data/dpo/representative_pdbs",
        help="Root containing train/valid/test representative PDB symlink dirs",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--valid_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--label_bins", type=int, default=DEFAULT_LABEL_BINS)
    parser.add_argument("--length_bins", type=str, default=DEFAULT_LENGTH_BINS)
    parser.add_argument(
        "--bin_width",
        type=float,
        default=0.5,
        help="Absolute-value bin width for --mode capped_bin",
    )
    parser.add_argument(
        "--max_per_bin",
        type=int,
        default=100,
        help="Maximum samples kept per value bin for --mode capped_bin",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Label threshold for --mode threshold_balance",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reps = pd.read_csv(args.representatives_csv).copy()
    required = {"name", "protein_sequence", "split", "cluster", "pdb_file", args.label_col}
    missing = required - set(reps.columns)
    if missing:
        raise ValueError(f"Missing required columns in representatives CSV: {missing}")

    reps["seq_len"] = reps["protein_sequence"].str.len()

    sampled_splits = []
    if args.mode == "stratified":
        length_bins = parse_bins(args.length_bins)
        length_labels = [f"len_{int(length_bins[i])}_{int(length_bins[i+1]) - 1}" for i in range(len(length_bins) - 1)]
        reps["length_bin"] = pd.cut(
            reps["seq_len"],
            bins=length_bins,
            labels=length_labels,
            include_lowest=True,
            right=False,
        )

        size_map = {"train": args.train_size, "valid": args.valid_size, "test": args.test_size}
        for split, target_n in size_map.items():
            split_df = reps[reps["split"] == split].copy()
            split_df["label_bin"] = assign_label_bins(split_df, args.label_col, args.label_bins).astype(str)
            split_df["stratum"] = split_df["length_bin"].astype(str) + "__" + split_df["label_bin"].astype(str)
            selected = sample_split(split_df, target_n=target_n, seed=args.seed)
            sampled_splits.append(selected)
    elif args.mode == "capped_bin":
        for split in ["train", "valid", "test"]:
            split_df = reps[reps["split"] == split].copy()
            selected = sample_split_capped_bin(
                split_df,
                label_col=args.label_col,
                bin_width=args.bin_width,
                max_per_bin=args.max_per_bin,
                seed=args.seed,
            )
            sampled_splits.append(selected)
    else:
        for split in ["train", "valid", "test"]:
            split_df = reps[reps["split"] == split].copy()
            selected = sample_split_threshold_balanced(
                split_df,
                label_col=args.label_col,
                threshold=args.threshold,
                seed=args.seed,
            )
            sampled_splits.append(selected)

    selected_df = pd.concat(sampled_splits, axis=0).sort_values(["split", "cluster", "name"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_symlinks(selected_df, Path(args.pdb_root), output_dir)

    manifest_path = output_dir / "subset_manifest.csv"
    selected_df.to_csv(manifest_path, index=False)

    print(f"Subset selection complete. mode={args.mode}")
    for split in ["train", "valid", "test"]:
        split_df = selected_df[selected_df["split"] == split]
        if args.mode == "stratified":
            detail = f"strata={split_df['stratum'].nunique()}"
        elif args.mode == "capped_bin":
            detail = f"value_bins={split_df['value_bin'].nunique()}"
        else:
            counts = split_df["threshold_group"].value_counts().to_dict()
            detail = f"threshold_groups={counts}"
        print(f"[{split}] selected={len(split_df)}  {detail}  path={output_dir / split}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
