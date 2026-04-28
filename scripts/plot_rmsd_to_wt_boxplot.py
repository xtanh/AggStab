#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "ESM-IF", "MoMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "MoMPNN": "#F0C987",
    "AggStab-DPO": "#E8B2A7",
    "AggStab-SFT": "#A7C4BC",
}


def load_backbone_mean_df(path: Path, method_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"method", "pdb_name", "rmsd_aligned"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df[df["method"].isin(method_order)].copy()
    out = (
        df.groupby(["method", "pdb_name"], as_index=False)["rmsd_aligned"]
        .mean()
        .rename(columns={"method": "Method", "rmsd_aligned": "mean_rmsd"})
    )
    return out


def add_median_labels(ax, data: pd.DataFrame, method_order: list[str]) -> None:
    medians = data.groupby("Method")["mean_rmsd"].median().reindex(method_order)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    for idx, value in enumerate(medians):
        ax.text(
            idx,
            value + 0.02 * yrange,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RMSD-to-WT distribution across methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=["ProteinMPNN", "SolubleMPNN", "ESM-IF", "MoMPNN", "AggStab-SFT", "AggStab-DPO"],
        help="Methods to include.",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("results/figures/rmsd_to_wt_all_methods.csv"),
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/rmsd_to_wt_boxplot"),
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=4.0,
        help="Upper y-limit for the main panel to suppress extreme tails visually.",
    )
    args = parser.parse_args()

    method_order = args.methods
    df = load_backbone_mean_df(args.input_csv, method_order)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    long_csv = args.output_prefix.with_name(args.output_prefix.name + "_long.csv")
    df.to_csv(long_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=300)
    palette = [METHOD_COLORS[m] for m in method_order]

    sns.violinplot(
        data=df,
        x="Method",
        y="mean_rmsd",
        hue="Method",
        order=method_order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.9,
        saturation=1.0,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="Method",
        y="mean_rmsd",
        order=method_order,
        width=0.22,
        showcaps=True,
        boxprops=dict(facecolor="white", edgecolor="#374151", linewidth=1.0, zorder=3),
        whiskerprops=dict(color="#374151", linewidth=1.0),
        capprops=dict(color="#374151", linewidth=1.0),
        medianprops=dict(color="#111827", linewidth=1.2),
        showfliers=False,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Backbone-level mean-of-top3 RMSD to WT (Å)")
    ax.set_title("Structural deviation from WT")
    ax.set_ylim(0, args.ylim)
    style_axis(ax)
    add_median_labels(ax, df, method_order)

    fig.tight_layout()
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    summary = (
        df.groupby("Method")["mean_rmsd"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved long CSV: {long_csv}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
