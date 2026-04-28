#!/usr/bin/env python
"""Select a shared pilot backbone subset from baseline and DPO candidate tables.

This script:
  1) finds common backbone names between two candidate CSVs
  2) samples a fixed number of backbone IDs with a fixed seed
  3) writes filtered CSVs for baseline and DPO
  4) writes matching FASTA files for downstream structure prediction
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_fasta(df: pd.DataFrame, fasta_path: Path, tag: str) -> None:
    with fasta_path.open("w") as f:
        for row in df.itertuples(index=False):
            header = (
                f">{row.pdb_name}|{tag}|top{row.topk_rank}|"
                f"rank{row.candidate_rank}|proagg={row.proagg_score:.4f}|"
                f"stage={row.selection_stage}"
            )
            f.write(header + "\n")
            f.write(row.sequence + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=Path, required=True)
    parser.add_argument("--dpo_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_backbones", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline_csv)
    dpo_df = pd.read_csv(args.dpo_csv)

    common = sorted(set(baseline_df["pdb_name"]).intersection(dpo_df["pdb_name"]))
    if len(common) < args.num_backbones:
        raise ValueError(
            f"Requested {args.num_backbones} backbones but only found {len(common)} common ones."
        )

    selected = (
        pd.Series(common)
        .sample(n=args.num_backbones, random_state=args.seed, replace=False)
        .sort_values()
        .tolist()
    )

    baseline_pilot = baseline_df[baseline_df["pdb_name"].isin(selected)].copy()
    dpo_pilot = dpo_df[dpo_df["pdb_name"].isin(selected)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_txt = args.output_dir / "selected_backbones.txt"
    selected_txt.write_text("\n".join(selected) + "\n")

    baseline_csv = args.output_dir / "baseline_top3_pilot.csv"
    dpo_csv = args.output_dir / "dpo_top3_pilot.csv"
    baseline_fasta = args.output_dir / "baseline_top3_pilot.fasta"
    dpo_fasta = args.output_dir / "dpo_top3_pilot.fasta"

    baseline_pilot.to_csv(baseline_csv, index=False)
    dpo_pilot.to_csv(dpo_csv, index=False)
    write_fasta(baseline_pilot, baseline_fasta, "baseline")
    write_fasta(dpo_pilot, dpo_fasta, "dpo")

    print(f"Selected backbones: {len(selected)}")
    print(f"Baseline rows:      {len(baseline_pilot)}")
    print(f"DPO rows:           {len(dpo_pilot)}")
    print(f"Saved: {selected_txt}")
    print(f"Saved: {baseline_csv}")
    print(f"Saved: {dpo_csv}")
    print(f"Saved: {baseline_fasta}")
    print(f"Saved: {dpo_fasta}")


if __name__ == "__main__":
    main()
