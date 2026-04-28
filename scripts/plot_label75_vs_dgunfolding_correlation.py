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
        description="Plot correlation between log2_fold_change_75_clip and ΔG_unfolding."
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
        default=Path("results/figures/log2fc75_vs_dGunfolding"),
    )
    parser.add_argument(
        "--highlight_names",
        type=str,
        nargs="*",
        default=[
            "rocklin_batch2_236560",
            "rocklin_batch2_249883",
            "rocklin_batch2_481438",
            "rocklin_batch2_126725",
        ],
        help="Specific samples to highlight with star markers.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    dg = pd.read_csv(args.dg_csv)[["name", "deltaG"]].rename(columns={"deltaG": "dG_unfolding"})
    df = raw.merge(dg, on="name", how="inner", validate="one_to_one").copy()

    x = df["log2_fold_change_75_clip"].astype(float)
    y = df["dG_unfolding"].astype(float)
    pcc, pcc_p = pearsonr(x, y)
    scc, scc_p = spearmanr(x, y)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_csv = args.output_prefix.with_name(args.output_prefix.name + "_merged.csv")
    stats_txt = args.output_prefix.with_name(args.output_prefix.name + "_stats.txt")
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")

    df[["name", "cluster", "split", "log2_fold_change_75_clip", "dG_unfolding"]].to_csv(
        merged_csv, index=False
    )

    with open(stats_txt, "w") as handle:
        handle.write(f"n={len(df)}\n")
        handle.write(f"pearson_r={pcc:.6f}\n")
        handle.write(f"pearson_p={pcc_p:.6e}\n")
        handle.write(f"spearman_r={scc:.6f}\n")
        handle.write(f"spearman_p={scc_p:.6e}\n")

    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=300)
    ax.scatter(x, y, s=14, alpha=0.22, linewidths=0, color="#0f766e", rasterized=True)
    coeffs = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(x.min()), float(x.max()), 200)
    yy = coeffs[0] * xx + coeffs[1]
    ax.plot(xx, yy, color="#115e59", linewidth=1.6, alpha=0.95)
    ax.set_xlabel("Anti-aggregation label (log2_fold_change_75_clip)")
    ax.set_ylabel("Folding stability label (ΔG_unfolding)")
    ax.set_title("Anti-aggregation label vs. folding stability")

    highlight_df = df[df["name"].isin(args.highlight_names)].copy()
    if not highlight_df.empty:
        ax.scatter(
            highlight_df["log2_fold_change_75_clip"].astype(float),
            highlight_df["dG_unfolding"].astype(float),
            s=140,
            marker="*",
            color="#b91c1c",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        for _, row in highlight_df.iterrows():
            short_name = row["name"].replace("rocklin_batch2_", "")
            x0 = float(row["log2_fold_change_75_clip"])
            y0 = float(row["dG_unfolding"])
            dx = 0.08 if x0 < x.median() else -0.08
            ha = "left" if dx > 0 else "right"
            ax.text(
                x0 + dx,
                y0 + 0.06,
                short_name,
                fontsize=8.8,
                color="#7f1d1d",
                ha=ha,
                va="bottom",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

    text = f"n = {len(df)}\nPCC = {pcc:.4f}\nSCC = {scc:.4f}"
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
