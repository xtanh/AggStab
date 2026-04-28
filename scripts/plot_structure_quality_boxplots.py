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
RUNS = {
    "ProteinMPNN": Path("results/joint_structure_vs_wt_analysis_label75/backbone_mean_of_top3_vs_wt.csv"),
    "SolubleMPNN": Path("results/joint_structure_vs_wt_analysis_label75_soluble/backbone_mean_of_top3_vs_wt.csv"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/chai_top3_joint"),
    "MoMPNN": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/chai_top3_joint"),
    "AggStab-SFT": Path("results/joint_structure_vs_wt_analysis_label75_sft/backbone_mean_of_top3_vs_wt.csv"),
    "AggStab-DPO": Path("results/joint_structure_vs_wt_analysis_label75_dpo_sft10/backbone_mean_of_top3_vs_wt.csv"),
}


def load_long_df(method_order: list[str]) -> pd.DataFrame:
    frames = []
    for method in method_order:
        path = RUNS[method]
        if method == "ProteinMPNN":
            df = pd.read_csv(path, usecols=["pdb_name", "base_mean_plddt", "base_mean_ptm"]).copy()
            df = df.rename(
                columns={
                    "base_mean_plddt": "mean_plddt",
                    "base_mean_ptm": "mean_ptm",
                }
            )
        elif method in {"ESM-IF", "MoMPNN"}:
            summary_frames = [pd.read_csv(f, usecols=["pdb_name", "best_plddt", "best_ptm"]) for f in sorted(path.glob("summary_shard*.csv"))]
            df = pd.concat(summary_frames, ignore_index=True)
            df = (
                df.groupby("pdb_name", as_index=False)
                .agg(
                    mean_plddt=("best_plddt", "mean"),
                    mean_ptm=("best_ptm", "mean"),
                )
            )
        else:
            df = pd.read_csv(path, usecols=["pdb_name", "best_mean_plddt", "best_mean_ptm"]).copy()
            df = df.rename(
                columns={
                    "best_mean_plddt": "mean_plddt",
                    "best_mean_ptm": "mean_ptm",
                }
            )
        df["Method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_median_labels(ax, data: pd.DataFrame, y_col: str, method_order: list[str]) -> None:
    medians = data.groupby("Method")[y_col].median().reindex(method_order)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    for idx, value in enumerate(medians):
        ax.text(
            idx,
            value + 0.03 * yrange,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structure-quality boxplots across methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=list(RUNS.keys()),
        help="Methods to plot.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/structure_quality_boxplots"),
    )
    args = parser.parse_args()

    method_order = args.methods
    df = load_long_df(method_order)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    long_csv = args.output_prefix.with_name(args.output_prefix.name + "_long.csv")
    df.to_csv(long_csv, index=False)

    palette = [METHOD_COLORS[m] for m in method_order]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    sns.boxplot(
        data=df,
        x="Method",
        y="mean_plddt",
        order=method_order,
        palette=palette,
        width=0.6,
        fliersize=1.8,
        linewidth=1.0,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Backbone-level mean-of-top3 pLDDT")
    axes[0].set_title("Structure confidence")
    style_axis(axes[0])
    add_median_labels(axes[0], df, "mean_plddt", method_order)

    sns.boxplot(
        data=df,
        x="Method",
        y="mean_ptm",
        order=method_order,
        palette=palette,
        width=0.6,
        fliersize=1.8,
        linewidth=1.0,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Backbone-level mean-of-top3 pTM")
    axes[1].set_title("Global fold agreement")
    style_axis(axes[1])
    add_median_labels(axes[1], df, "mean_ptm", method_order)

    fig.tight_layout()
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved long CSV: {long_csv}")


if __name__ == "__main__":
    main()
