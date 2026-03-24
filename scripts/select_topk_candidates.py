#!/usr/bin/env python
"""Select per-backbone top-k candidates with pathology filtering.

This script is intended for downstream structure-prediction selection.
It:
  1) computes simple pathology stats per sequence
  2) filters out obviously bad sequences
  3) ranks the remaining candidates by ProAgg score and keeps top-k

If a backbone has no candidates after strict filtering, the script falls back in stages:
  - no_filter
The fallback stage used is recorded in the output.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CHARGED = set("DEKRH")
HOMOPOLYMER_RUN_RE = re.compile(r"(.)\1{5,}")  # >= 6


def seq_stats(seq: str) -> dict[str, float | int]:
    counts = Counter(seq)
    length = len(seq)
    if length == 0:
        return {
            "seq_len": 0,
            "top_frac": 0.0,
            "charged_frac": 0.0,
            "has_run": 0,
        }
    return {
        "seq_len": length,
        "top_frac": counts.most_common(1)[0][1] / length,
        "charged_frac": sum(counts.get(aa, 0) for aa in CHARGED) / length,
        "has_run": int(HOMOPOLYMER_RUN_RE.search(seq) is not None),
    }


def select_group(
    group: pd.DataFrame,
    top_k: int,
    max_top_frac: float,
    max_charged_frac: float,
) -> pd.DataFrame:
    group = group.copy()

    strict = group[
        (group["has_run"] == 0)
        & (group["top_frac"] <= max_top_frac)
        & (group["charged_frac"] <= max_charged_frac)
    ].copy()
    if len(strict) >= top_k:
        strict["selection_stage"] = "strict"
        return strict.sort_values(["proagg_score", "mpnn_logprob"], ascending=[False, False]).head(top_k)

    no_filter = group.copy()
    no_filter["selection_stage"] = "no_filter"
    return no_filter.sort_values(["proagg_score", "mpnn_logprob"], ascending=[False, False]).head(top_k)


def write_fasta(df: pd.DataFrame, fasta_path: Path, tag: str) -> None:
    with fasta_path.open("w") as f:
        for row in df.itertuples(index=False):
            header = (
                f">{row.pdb_name}|{tag}|top{row.topk_rank}|"
                f"rank{row.candidate_rank}|proagg={row.proagg_score:.4f}|"
                f"logprob={row.mpnn_logprob:.4f}|stage={row.selection_stage}"
            )
            f.write(header + "\n")
            f.write(row.sequence + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_fasta", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_top_frac", type=float, default=0.40)
    parser.add_argument("--max_charged_frac", type=float, default=0.50)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    stat_df = pd.DataFrame([seq_stats(seq) for seq in df["sequence"]])
    df = pd.concat([df, stat_df], axis=1)

    selected = []
    for _, group in df.groupby("pdb_name", sort=True):
        picked = select_group(
            group=group,
            top_k=args.top_k,
            max_top_frac=args.max_top_frac,
            max_charged_frac=args.max_charged_frac,
        ).copy()
        picked["topk_rank"] = list(range(1, len(picked) + 1))
        selected.append(picked)

    out = pd.concat(selected, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    write_fasta(out, args.output_fasta, args.tag)

    stage_counts = out.groupby("selection_stage")["pdb_name"].nunique().to_dict()
    print(f"Saved CSV:   {args.output_csv}")
    print(f"Saved FASTA: {args.output_fasta}")
    print(f"Backbones:   {out['pdb_name'].nunique()}")
    print(f"Rows:        {len(out)}")
    print(f"Stage usage: {stage_counts}")


if __name__ == "__main__":
    main()
