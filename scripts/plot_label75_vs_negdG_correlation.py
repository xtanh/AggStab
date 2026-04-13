#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot correlation between log2_fold_change_75_clip and -deltaG."
    )
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=Path("data/rocklin/rawdata/data.csv"),
    )
    parser.add_argument(
        "--dg_csv",
        type=Path,
        default=Path("data/rocklin/Metagenomic_dG.csv"),
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/log2fc75_vs_negdG"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    dg = pd.read_csv(args.dg_csv)[["name", "deltaG"]]
    df = raw.merge(dg, on="name", how="inner", validate="one_to_one").copy()
    df["neg_deltaG"] = -df["deltaG"]

    x = df["log2_fold_change_75_clip"].astype(float)
    y = df["neg_deltaG"].astype(float)
    pcc, pcc_p = pearsonr(x, y)
    scc, scc_p = spearmanr(x, y)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_csv = args.output_prefix.with_name(args.output_prefix.name + "_merged.csv")
    stats_txt = args.output_prefix.with_name(args.output_prefix.name + "_stats.txt")
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")

    df[["name", "cluster", "split", "log2_fold_change_75_clip", "deltaG", "neg_deltaG"]].to_csv(
        merged_csv, index=False
    )

    with open(stats_txt, "w") as handle:
        handle.write(f"n={len(df)}\n")
        handle.write(f"pearson_r={pcc:.6f}\n")
        handle.write(f"pearson_p={pcc_p:.6e}\n")
        handle.write(f"spearman_r={scc:.6f}\n")
        handle.write(f"spearman_p={scc_p:.6e}\n")

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    ax.scatter(x, y, s=14, alpha=0.22, linewidths=0, color="#7c3aed", rasterized=True)
    coeffs = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(x.min()), float(x.max()), 200)
    yy = coeffs[0] * xx + coeffs[1]
    ax.plot(xx, yy, color="#4c1d95", linewidth=1.6, alpha=0.95)
    ax.set_xlabel("log2_fold_change_75_clip")
    ax.set_ylabel("-deltaG")
    ax.set_title("Aggregation label vs. negated stability label")

    text = (
        f"n = {len(df)}\n"
        f"PCC = {pcc:.4f}\n"
        f"SCC = {scc:.4f}"
    )
    add_stats_box(ax, text)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG:   {svg_path}")
    print(f"Saved PDF:   {pdf_path}")
    print(f"Saved CSV:   {merged_csv}")
    print(f"Saved stats: {stats_txt}")
    print(f"PCC={pcc:.6f}, SCC={scc:.6f}")


if __name__ == "__main__":
    main()
