"""Analyze DPO v2 evaluation for sequence pathologies and preference-pair bias.

This script focuses on diagnosing the failure mode observed in `results/dpo_v2/`,
especially composition collapse (low-complexity / homopolymer runs) and how it
relates to `proagg_max` degradation.

Usage:
  # 1) Pure-python analysis (no torch required)
  python scripts/analyze_dpo_v2_pathology.py \
    --eval_json results/dpo_v2/eval_results.json \
    --summary_csv results/dpo_v2/dpo_eval_summary.csv

  # 2) Also analyze training/val preference pairs (requires torch)
  bash -lc 'eval "$(conda shell.bash hook)"; conda activate SaProt; \
    python scripts/analyze_dpo_v2_pathology.py \
      --eval_json results/dpo_v2/eval_results.json \
      --summary_csv results/dpo_v2/dpo_eval_summary.csv \
      --train_pairs_pt data/dpo/train_pairs.pt \
      --val_pairs_pt data/dpo/val_pairs.pt'
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CHARGED = set("DEKRH")
HYDROPHOBIC = set("AILMFWV")
HOMOPOLYMER_RUN_RE = re.compile(r"(.)\1{5,}")  # >= 6


@dataclass(frozen=True)
class SeqStats:
    top_frac: float
    charged_frac: float
    hydrophobic_frac: float
    has_homopolymer_run: int


def _seq_stats(seq: str) -> SeqStats:
    c = Counter(seq)
    L = len(seq)
    if L == 0:
        return SeqStats(0.0, 0.0, 0.0, 0)
    top_frac = c.most_common(1)[0][1] / L
    charged_frac = sum(c.get(a, 0) for a in CHARGED) / L
    hydrophobic_frac = sum(c.get(a, 0) for a in HYDROPHOBIC) / L
    has_run = 1 if HOMOPOLYMER_RUN_RE.search(seq) else 0
    return SeqStats(top_frac, charged_frac, hydrophobic_frac, has_run)


def _quantiles(x: np.ndarray, qs=(0.0, 0.5, 0.9, 1.0)) -> dict[float, float]:
    out = {}
    for q in qs:
        out[q] = float(np.quantile(x, q))
    return out


def analyze_eval(eval_json: Path, summary_csv: Path, output_dir: Path) -> Path:
    data = json.loads(eval_json.read_text())
    df = pd.read_csv(summary_csv)

    by_o = {r["pdb_name"]: r for r in data["original"]}
    by_d = {r["pdb_name"]: r for r in data["dpo"]}

    common = sorted(set(by_o).intersection(by_d))
    if len(common) == 0:
        raise RuntimeError("No overlapping pdb_name between original and dpo results.")

    df_map = df.set_index("pdb_name")

    rows = []
    for name in common:
        srow = df_map.loc[name]
        d = by_d[name]
        o = by_o[name]
        sd = _seq_stats(d["best_seq"])
        so = _seq_stats(o["best_seq"])
        rows.append(
            {
                "pdb_name": name,
                "delta_max": float(srow["delta_max"]),
                "delta_mean": float(srow["delta_mean"]),
                "diversity_dpo": float(srow["diversity_dpo"]),
                "n_unique_dpo": int(d.get("n_unique", 0)),
                "top_frac_dpo": sd.top_frac,
                "charged_frac_dpo": sd.charged_frac,
                "hydrophobic_frac_dpo": sd.hydrophobic_frac,
                "has_run_dpo": sd.has_homopolymer_run,
                "top_frac_orig": so.top_frac,
                "charged_frac_orig": so.charged_frac,
                "hydrophobic_frac_orig": so.hydrophobic_frac,
                "has_run_orig": so.has_homopolymer_run,
            }
        )

    adf = pd.DataFrame(rows)
    out_csv = output_dir / "dpo_v2_pathology_enriched.csv"
    adf.to_csv(out_csv, index=False)

    # Print a concise report
    print("=" * 60)
    print("DPO v2 pathology report")
    print("=" * 60)
    print(f"eval_json:   {eval_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"N: {len(adf)}")
    print()

    print("--- Pathology rates (DPO best_seq) ---")
    print(f"has_run(>=6): {adf.has_run_dpo.mean():.4%}")
    for col in ["top_frac_dpo", "charged_frac_dpo", "hydrophobic_frac_dpo", "diversity_dpo"]:
        arr = adf[col].to_numpy()
        qs = _quantiles(arr, qs=(0.0, 0.5, 0.9, 1.0))
        print(
            f"{col}: mean={arr.mean():.4f} "
            f"p50={qs[0.5]:.4f} p90={qs[0.9]:.4f} min={qs[0.0]:.4f} max={qs[1.0]:.4f}"
        )
    print()

    print("--- Correlations vs delta_max (Pearson) ---")
    for col in ["top_frac_dpo", "charged_frac_dpo", "hydrophobic_frac_dpo", "diversity_dpo", "n_unique_dpo"]:
        corr = float(np.corrcoef(adf["delta_max"], adf[col])[0, 1])
        print(f"corr(delta_max, {col}) = {corr:+.4f}")
    print()

    print(f"Saved enriched CSV: {out_csv}")
    print("=" * 60)
    return out_csv


def _pair_bias(pairs_pt: Path, label: str) -> None:
    try:
        import torch  # noqa: F401
        import torch as _torch
    except Exception as e:
        raise RuntimeError(
            f"torch is required to analyze {pairs_pt}. "
            f"Run inside an environment with torch. Error: {e}"
        ) from e

    data = _torch.load(pairs_pt, map_location="cpu")
    pairs = data["pairs"]

    winners = [p["seq_winner"] for p in pairs]
    losers = [p["seq_loser"] for p in pairs]

    def agg(seqs):
        c = Counter()
        n = 0
        for s in seqs:
            c.update(s)
            n += len(s)
        return c, n

    def group(c, n):
        return {
            "charged(DEKRH)": sum(c.get(a, 0) for a in "DEKRH") / n,
            "hydrophobic(AILMFWV)": sum(c.get(a, 0) for a in "AILMFWV") / n,
        }

    cw, nw = agg(winners)
    cl, nl = agg(losers)

    print()
    print("--- Pair bias:", label, "---")
    print(f"pairs: {len(pairs)} ({pairs_pt})")
    print("winner groups:", group(cw, nw))
    print("loser  groups:", group(cl, nl))

    aas = sorted(set(cw) | set(cl))
    diffs = []
    for a in aas:
        fw = cw.get(a, 0) / nw
        fl = cl.get(a, 0) / nl
        diffs.append((a, fw - fl, fw, fl))
    diffs.sort(key=lambda x: -abs(x[1]))
    print("top abs diffs (winner - loser):")
    for a, d, fw, fl in diffs[:10]:
        print(f"  {a}: {d:+.4f} (w={fw:.4f}, l={fl:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", type=Path, required=True)
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--train_pairs_pt", type=Path, default=None)
    p.add_argument("--val_pairs_pt", type=Path, default=None)
    args = p.parse_args()

    output_dir = args.output_dir or args.eval_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_eval(args.eval_json, args.summary_csv, output_dir)

    if args.train_pairs_pt:
        _pair_bias(args.train_pairs_pt, "train")
    if args.val_pairs_pt:
        _pair_bias(args.val_pairs_pt, "val")


if __name__ == "__main__":
    main()

