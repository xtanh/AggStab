#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "ESM-IF", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "AggStab-DPO": "#E8B2A7",
}
THRESHOLD_ORDER = [
    "No structure filter",
    "pLDDT>0.75\npTM>0.60",
    "pLDDT>0.80\npTM>0.70",
    "pLDDT>0.85\npTM>0.80",
]


def _plot_panel(ax, df: pd.DataFrame, value_col: str, title: str, ylabel: str, ylim: tuple[float, float]) -> None:
    x = np.arange(len(THRESHOLD_ORDER))
    for method in METHOD_ORDER:
        sub = (
            df[df["Method"] == method]
            .set_index("Threshold")
            .reindex(THRESHOLD_ORDER)
        )
        y = sub[value_col].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=6.5,
            linewidth=2.3,
            color=METHOD_COLORS[method],
            label=method,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=1.1,
            solid_capstyle="round",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(THRESHOLD_ORDER, fontsize=10)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Structure filtering criterion", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(colors="#374151", width=0.8, length=4)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structure-only and joint threshold sensitivity in one figure.")
    parser.add_argument(
        "--structure_csv",
        type=Path,
        default=Path("results/figures/structure_threshold_sensitivity_lines_summary.csv"),
    )
    parser.add_argument(
        "--joint_csv",
        type=Path,
        default=Path("results/figures/threshold_sensitivity_lines_summary.csv"),
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/dual_threshold_sensitivity_panels"),
    )
    args = parser.parse_args()

    structure_df = pd.read_csv(args.structure_csv)
    joint_df = pd.read_csv(args.joint_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=300, sharex=False)
    _plot_panel(
        axes[0],
        structure_df,
        value_col="Structure success",
        title="Structure success",
        ylabel="Structure success rate",
        ylim=(0.35, 1.02),
    )
    _plot_panel(
        axes[1],
        joint_df,
        value_col="Joint success",
        title="Joint success",
        ylabel="Joint success rate",
        ylim=(0.10, 0.86),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove() if axes[0].legend_ else None
    axes[1].legend_.remove() if axes[1].legend_ else None
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
