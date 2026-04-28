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


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)


def violin_panel(ax, grouped, ylabel, title):
    vp = ax.violinplot(
        grouped,
        positions=range(len(METHOD_ORDER)),
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, method in zip(vp["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor(METHOD_COLORS[method])
        body.set_alpha(0.75)
    medians = [pd.Series(vals).median() for vals in grouped]
    ax.scatter(
        range(len(METHOD_ORDER)),
        medians,
        color="white",
        edgecolors="#374151",
        s=34,
        zorder=4,
        linewidths=1.0,
    )
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    style_axes(ax)


def bar_panel(ax, series, ylabel, title):
    series = series.reindex(METHOD_ORDER)
    bars = ax.bar(
        series.index,
        series.values,
        color=[METHOD_COLORS[m] for m in series.index],
        edgecolor="none",
        width=0.72,
        zorder=3,
    )
    for bar, value in zip(bars, series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#374151",
        )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(0.9, series.max() + 0.08))
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(10)
    style_axes(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="results/figures/aggstab_top3_main4_predictions.csv",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/figures/external_ddg_validation_main4_2x2",
    )
    args = parser.parse_args()

    row = pd.read_csv(args.input_csv)
    row = row[row["method"].isin(METHOD_ORDER)].copy()
    row["external_stability_gain"] = -row["predicted_ddG"]
    row["external_improved"] = row["predicted_ddG"] < 0

    backbone = (
        row.groupby(["method", "pdb_name"], as_index=False)
        .agg(
            external_stability_gain_mean=("external_stability_gain", "mean"),
            external_improved_frac=("external_improved", "mean"),
        )
    )
    backbone["any_external_improved"] = backbone["external_improved_frac"] > 0

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4), dpi=300)

    row_grouped = [
        row.loc[row["method"] == method, "external_stability_gain"].to_numpy()
        for method in METHOD_ORDER
    ]
    violin_panel(
        axes[0, 0],
        row_grouped,
        "External stability gain (-predicted ΔΔG)",
        "Row-Level Stability Distribution",
    )

    row_frac = row.groupby("method", observed=True)["external_improved"].mean()
    bar_panel(
        axes[0, 1],
        row_frac,
        "Predicted stabilizing fraction",
        "Row-Level Stabilizing Fraction",
    )

    backbone_grouped = [
        backbone.loc[backbone["method"] == method, "external_stability_gain_mean"].to_numpy()
        for method in METHOD_ORDER
    ]
    violin_panel(
        axes[1, 0],
        backbone_grouped,
        "Backbone mean external stability gain",
        "Backbone-Level Mean Stability Gain",
    )

    backbone_cov = backbone.groupby("method", observed=True)["any_external_improved"].mean()
    bar_panel(
        axes[1, 1],
        backbone_cov,
        "Backbone fraction with any stabilizing top3 design",
        "Backbone-Level Stabilizing Coverage",
    )

    fig.tight_layout()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
