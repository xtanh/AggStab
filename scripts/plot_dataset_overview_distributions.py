#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


COLORS = {
    "agg_fill": "#B2E6FD",
    "agg_line": "#4A86B8",
    "dg_fill": "#B8D2CC",
    "dg_line": "#4D8A72",
    "len_fill": "#E8B2A7",
    "len_line": "#C97D6D",
}


def _save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVG: {prefix.with_suffix('.svg')}")
    print(f"Saved PDF: {prefix.with_suffix('.pdf')}")


def _plot_hist_kde(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str,
    hist_color: str,
    line_color: str,
    bins: int,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=300)
    ax.hist(
        values,
        bins=bins,
        density=True,
        color=hist_color,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.8,
        zorder=1,
    )
    xs = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=line_color, linewidth=2.2, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    style_axis(ax)

    text = (
        f"n = {len(values):,}\n"
        f"mean = {values.mean():.3f}\n"
        f"median = {np.median(values):.3f}\n"
        f"std = {values.std(ddof=1):.3f}"
    )
    add_stats_box(ax, text)
    _save(fig, output_prefix)


def _plot_length_bar(
    values: np.ndarray,
    *,
    xlabel: str,
    title: str,
    bar_color: str,
    edge_color: str,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=300)
    values = values.astype(int)
    lengths, counts = np.unique(values, return_counts=True)
    ax.bar(lengths, counts, width=0.85, color=bar_color, edgecolor=bar_color, linewidth=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    main_lengths = lengths[counts >= max(5, int(0.002 * len(values)))]
    if len(main_lengths) > 0:
        ax.set_xlim(main_lengths.min() - 0.8, main_lengths.max() + 0.8)
    visible = lengths[(lengths >= ax.get_xlim()[0]) & (lengths <= ax.get_xlim()[1])]
    ax.set_xticks(visible[::2] if len(visible) > 18 else visible)
    style_axis(ax)

    text = (
        f"n = {len(values):,}\n"
        f"min = {values.min()}\n"
        f"max = {values.max()}\n"
        f"median = {int(np.median(values))}\n"
        f"rare shorter outliers omitted"
    )
    add_stats_box(ax, text)
    _save(fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset overview distributions for the main figure.")
    parser.add_argument("--raw_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--dg_csv", type=Path, default=Path("data/rocklin/Metagenomic_dG.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv, usecols=["name", "protein_sequence", "log2_fold_change_75_clip"])
    dg = pd.read_csv(args.dg_csv, usecols=["name", "deltaG"])

    agg_values = raw["log2_fold_change_75_clip"].dropna().astype(float).to_numpy()
    dg_values = raw[["name"]].merge(dg, on="name", how="inner", validate="one_to_one")["deltaG"].dropna().astype(float).to_numpy()
    len_values = raw["protein_sequence"].astype(str).str.len().astype(float).to_numpy()

    _plot_hist_kde(
        agg_values,
        xlabel="Anti-aggregation label (`log2_fold_change_75_clip`)",
        title="Distribution of anti-aggregation labels",
        hist_color=COLORS["agg_fill"],
        line_color=COLORS["agg_line"],
        bins=48,
        output_prefix=args.output_dir / "dataset_overview_log2fc75_distribution",
    )
    _plot_hist_kde(
        dg_values,
        xlabel="Stability label (`ΔG_unfolding`)",
        title="Distribution of stability labels",
        hist_color=COLORS["dg_fill"],
        line_color=COLORS["dg_line"],
        bins=48,
        output_prefix=args.output_dir / "dataset_overview_dgunfolding_distribution",
    )
    _plot_length_bar(
        len_values,
        xlabel="Protein length (aa)",
        title="Distribution of protein lengths",
        bar_color=COLORS["len_fill"],
        edge_color=COLORS["len_line"],
        output_prefix=args.output_dir / "dataset_overview_length_distribution",
    )


if __name__ == "__main__":
    main()
