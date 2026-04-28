#!/usr/bin/env python

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


METHOD_ORDER = ["ESM-IF","ProteinMPNN", "SolubleMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "AggStab-DPO": "#E8B2A7",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="results/figures/aggstab_top3_main4_predictions.csv",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/figures/external_ddg_validation",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["external_stability_gain"] = -df["predicted_ddG"]
    df["external_improved"] = df["predicted_ddG"] < 0

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=300)

    ax = axes[0]
    grouped = [df.loc[df["method"] == method, "external_stability_gain"].to_numpy() for method in METHOD_ORDER]
    vp = ax.violinplot(grouped, positions=range(len(METHOD_ORDER)), widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body, method in zip(vp["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor(METHOD_COLORS[method])
        body.set_alpha(0.75)

    medians = [pd.Series(vals).median() for vals in grouped]
    ax.scatter(range(len(METHOD_ORDER)), medians, color="white", edgecolors="#374151", s=36, zorder=4, linewidths=1.0)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("External stability gain (-predicted ΔΔG)", fontsize=12)
    ax.set_title("External Stability Score", fontsize=13, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)

    ax = axes[1]
    summary = (
        df.groupby("method", observed=True)["external_improved"]
        .mean()
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    bars = ax.bar(
        summary["method"],
        summary["external_improved"],
        color=[METHOD_COLORS[m] for m in summary["method"]],
        edgecolor="none",
        width=0.72,
        zorder=3,
    )
    for bar, value in zip(bars, summary["external_improved"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )
    ax.set_ylabel("Predicted stabilizing fraction", fontsize=12)
    ax.set_title("External Stability Improvement Rate", fontsize=13, pad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(0.75, summary["external_improved"].max() + 0.08))
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(10)
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
