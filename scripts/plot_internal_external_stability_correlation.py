#!/usr/bin/env python

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def corr_stats(df: pd.DataFrame):
    pcc = df["deltaG_minus_wt"].corr(df["external_stability_gain"], method="pearson")
    scc = df["deltaG_minus_wt"].corr(df["external_stability_gain"], method="spearman")
    return pcc, scc


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)


def add_stats(ax, title, n, pcc, scc):
    ax.set_title(title, fontsize=13, pad=8)
    ax.text(
        0.03,
        0.97,
        f"n = {n}\nPCC = {pcc:.3f}\nSCC = {scc:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.95),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="results/figures/aggstab_top3_main4_predictions.csv",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/figures/internal_external_stability_correlation",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["deltaG_minus_wt", "predicted_ddG"]).copy()
    df["external_stability_gain"] = -df["predicted_ddG"]

    row_df = df.copy()
    backbone_df = (
        df.groupby(["method", "pdb_name"], as_index=False)
        .agg(
            deltaG_minus_wt=("deltaG_minus_wt", "mean"),
            external_stability_gain=("external_stability_gain", "mean"),
        )
    )

    row_pcc, row_scc = corr_stats(row_df)
    bb_pcc, bb_scc = corr_stats(backbone_df)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=300)

    ax = axes[0]
    ax.scatter(
        row_df["deltaG_minus_wt"],
        row_df["external_stability_gain"],
        s=8,
        alpha=0.28,
        color="#8DA6C1",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel("Internal stability gain (ΔG - WT)", fontsize=12)
    ax.set_ylabel("External stability gain (-predicted ΔΔG)", fontsize=12)
    style_axes(ax)
    add_stats(ax, "Row-Level Correlation", len(row_df), row_pcc, row_scc)

    ax = axes[1]
    ax.scatter(
        backbone_df["deltaG_minus_wt"],
        backbone_df["external_stability_gain"],
        s=12,
        alpha=0.40,
        color="#E0A18D",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel("Backbone mean internal stability gain", fontsize=12)
    ax.set_ylabel("Backbone mean external stability gain", fontsize=12)
    style_axes(ax)
    add_stats(ax, "Backbone-Level Correlation", len(backbone_df), bb_pcc, bb_scc)

    fig.tight_layout()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close(fig)

    row_df.to_csv(f"{output_prefix}_row_long.csv", index=False)
    backbone_df.to_csv(f"{output_prefix}_backbone_long.csv", index=False)
    pd.DataFrame(
        [
            {"level": "row", "n": len(row_df), "pcc": row_pcc, "scc": row_scc},
            {"level": "backbone", "n": len(backbone_df), "pcc": bb_pcc, "scc": bb_scc},
        ]
    ).to_csv(f"{output_prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
