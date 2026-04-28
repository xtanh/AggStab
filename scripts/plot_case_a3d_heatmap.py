#!/usr/bin/env python

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["WT", "ProteinMPNN", "SolubleMPNN", "ESM-IF", "AggStab-DPO"]
SUMMARY_COLORS = {
    "WT": "#7A7A7A",
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "AggStab-DPO": "#E8B2A7",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results/agg3d_case/a3d_case_long.csv")
    parser.add_argument("--summary_csv", default="results/agg3d_case/a3d_case_summary.csv")
    parser.add_argument("--output_prefix", default="results/agg3d_case/a3d_case_heatmap")
    parser.add_argument("--heatmap_only", action="store_true")
    args = parser.parse_args()

    long_df = pd.read_csv(args.input_csv)
    summary_df = pd.read_csv(args.summary_csv)

    long_df["method"] = pd.Categorical(long_df["method"], categories=METHOD_ORDER, ordered=True)
    summary_df["Method"] = pd.Categorical(summary_df["Method"], categories=METHOD_ORDER, ordered=True)
    summary_df = summary_df.sort_values("Method")

    heatmap_df = (
        long_df.pivot(index="method", columns="residue", values="score")
        .reindex(METHOD_ORDER)
    )

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    if args.heatmap_only:
        fig, ax = plt.subplots(1, 1, figsize=(11.0, 2.9), dpi=300)
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.0, 6.6),
            dpi=300,
            gridspec_kw={"height_ratios": [2.4, 1.0]},
        )
        ax = axes[0]

    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "A3D score"},
        linewidths=0.15,
        linecolor="white",
    )
    ax.set_xlabel("Residue index", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Per-residue aggregation propensity (A3D)", fontsize=13, pad=8)
    ax.tick_params(axis="y", labelrotation=0)
    ax.text(
        0.0,
        1.07,
        "More negative = less aggregation-prone; more positive = more aggregation-prone",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )

    if not args.heatmap_only:
        ax = axes[1]
        x = range(len(summary_df))
        ax.bar(
            x,
            summary_df["A3D_mean"],
            color=[SUMMARY_COLORS[m] for m in summary_df["Method"]],
            edgecolor="none",
            width=0.72,
            zorder=3,
        )
        ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(summary_df["Method"], rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("Mean A3D score", fontsize=12)
        ax.set_title("Sequence-level mean aggregation propensity", fontsize=12, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
        ax.set_axisbelow(True)

    fig.tight_layout()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
