#!/usr/bin/env python

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["WT", "ProteinMPNN", "SolubleMPNN", "ESM-IF", "AggStab-DPO"]
METHOD_COLORS = {
    "WT": "#7A7A7A",
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "AggStab-DPO": "#E8B2A7",
}


def rolling_mean(values, window=5):
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results/agg3d_case/a3d_case_long.csv")
    parser.add_argument("--summary_csv", default="results/agg3d_case/a3d_case_summary.csv")
    parser.add_argument("--output_prefix", default="results/agg3d_case/a3d_case_profiles")
    parser.add_argument("--smooth_window", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    summary = pd.read_csv(args.summary_csv)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    summary["Method"] = pd.Categorical(summary["Method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("Method")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.8, 7.2),
        dpi=300,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    ax = axes[0]
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("residue")
        x = sub["residue"].to_numpy()
        y = sub["score"].to_numpy()
        y_smooth = rolling_mean(y, window=args.smooth_window)
        ax.plot(
            x,
            y_smooth,
            color=METHOD_COLORS[method],
            linewidth=2.2 if method != "WT" else 2.4,
            alpha=0.98,
            label=method,
        )
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_xlim(df["residue"].min(), df["residue"].max())
    ax.set_xlabel("Residue index", fontsize=12)
    ax.set_ylabel("A3D score", fontsize=12)
    ax.set_title("Per-residue aggregation propensity profiles", fontsize=13, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.text(
        0.01,
        0.03,
        "More negative scores indicate lower aggregation propensity",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )

    ax = axes[1]
    x = range(len(summary))
    ax.bar(
        x,
        summary["A3D_mean"],
        width=0.72,
        color=[METHOD_COLORS[m] for m in summary["Method"]],
        edgecolor="none",
        zorder=3,
    )
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["Method"], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Mean A3D score", fontsize=12)
    ax.set_title("Sequence-level summary", fontsize=12, pad=8)
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
