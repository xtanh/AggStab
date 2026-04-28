#!/usr/bin/env python
"""Select per-backbone top-k candidates for joint property validation.

Selection is intentionally conservative and staged:
  1) pathology-safe + ProAgg>0 + deltaG_minus_wt>0
  2) pathology-safe + ProAgg>0
  3) pathology-safe
  4) no filter

This preserves coverage while recording how much fallback was needed.
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


def rank_candidates(df: pd.DataFrame, top_k: int, stage: str) -> pd.DataFrame:
    ranked = df.copy()
    ranked["selection_stage"] = stage
    ranked["deltaG_minus_wt_for_sort"] = ranked["deltaG_minus_wt"].fillna(float("-inf"))
    if "mpnn_logprob" not in ranked.columns:
        ranked["mpnn_logprob"] = float("nan")
    ranked["mpnn_logprob_for_sort"] = ranked["mpnn_logprob"].fillna(float("-inf"))
    ranked = ranked.sort_values(
        ["proagg_score", "deltaG_minus_wt_for_sort", "mpnn_logprob_for_sort"],
        ascending=[False, False, False],
    ).head(top_k)
    ranked = ranked.drop(columns=["deltaG_minus_wt_for_sort", "mpnn_logprob_for_sort"])
    return ranked


def select_group(
    group: pd.DataFrame,
    top_k: int,
    max_top_frac: float,
    max_charged_frac: float,
) -> pd.DataFrame:
    group = group.copy()
    pathology = group[
        (group["has_run"] == 0)
        & (group["top_frac"] <= max_top_frac)
        & (group["charged_frac"] <= max_charged_frac)
    ].copy()

    strict_joint = pathology[
        (pathology["proagg_score"] > 0)
        & (pathology["deltaG_minus_wt"].notna())
        & (pathology["deltaG_minus_wt"] > 0)
    ].copy()
    if len(strict_joint) >= top_k:
        return rank_candidates(strict_joint, top_k, "strict_joint")

    strict_proagg = pathology[pathology["proagg_score"] > 0].copy()
    if len(strict_proagg) >= top_k:
        return rank_candidates(strict_proagg, top_k, "strict_proagg")

    if len(pathology) >= top_k:
        return rank_candidates(pathology, top_k, "pathology_only")

    return rank_candidates(group, top_k, "no_filter")


def write_fasta(df: pd.DataFrame, fasta_path: Path, tag: str) -> None:
    with fasta_path.open("w") as f:
        for row in df.itertuples(index=False):
            delta_g = "nan" if pd.isna(row.deltaG) else f"{row.deltaG:.4f}"
            delta_g_minus_wt = "nan" if pd.isna(row.deltaG_minus_wt) else f"{row.deltaG_minus_wt:.4f}"
            header = (
                f">{row.pdb_name}|{tag}|top{row.topk_rank}|rank{row.candidate_rank}|"
                f"proagg={row.proagg_score:.4f}|dg={delta_g}|dgmwt={delta_g_minus_wt}|"
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
