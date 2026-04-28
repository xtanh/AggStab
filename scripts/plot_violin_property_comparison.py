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
    "MoMPNN[Sol+TM]": "#DCC7AA",
    "MoMPNN[Sol+IG+ESM]": "#F0C987",
    "MoMPNN-t0.5": "#D9A441",
}
RUNS = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/full_candidates.csv"),
    "MoMPNN": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/full_candidates.csv"),
    "AggStab-SFT": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "AggStab-DPO": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "MoMPNN[Sol+TM]": Path("results/mompnn_protsol_tm_fulltest_n16_joint/full_candidates.csv"),
    "MoMPNN[Sol+IG+ESM]": Path("results/mompnn_protsol_ig_esm_fulltest_n16_joint/full_candidates.csv"),
    "MoMPNN-t0.5": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/full_candidates.csv"),
}


def load_long_df(method_order: list[str]) -> pd.DataFrame:
    frames = []
    for method in method_order:
        path = RUNS[method]
        df = pd.read_csv(path, usecols=["pdb_name", "proagg_score", "deltaG"])
        grouped = (
            df.groupby("pdb_name", as_index=False)
            .agg(
                proagg_score=("proagg_score", "mean"),
                deltaG=("deltaG", "mean"),
            )
        )
        grouped["Method"] = method
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def add_median_labels(ax, data: pd.DataFrame, y_col: str, method_order: list[str]) -> None:
    grouped = data.groupby("Method")[y_col].median().reindex(method_order)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    for idx, (method, value) in enumerate(grouped.items()):
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
    parser = argparse.ArgumentParser(description="Plot violin comparison for aggregation and stability scores.")
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
        default=Path("results/figures/violin_property_comparison"),
    )
    args = parser.parse_args()

    method_order = args.methods
    df = load_long_df(method_order)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_long.csv"), index=False)

    palette = [METHOD_COLORS[m] for m in method_order]

    # Stability
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=300)
    sns.violinplot(
        data=df,
        x="Method",
        y="deltaG",
        order=method_order,
        palette=palette,
        inner="box",
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("ΔG_unfolding")
    ax.set_title("Stability score distribution")
    style_axis(ax)
    add_median_labels(ax, df, "deltaG", method_order)
    fig.tight_layout()
    stability_svg = args.output_prefix.with_name(args.output_prefix.name + "_stability.svg")
    stability_pdf = args.output_prefix.with_name(args.output_prefix.name + "_stability.pdf")
    fig.savefig(stability_svg, format="svg", bbox_inches="tight")
    fig.savefig(stability_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Aggregation
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=300)
    sns.violinplot(
        data=df,
        x="Method",
        y="proagg_score",
        order=method_order,
        palette=palette,
        inner="box",
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Anti-aggregation score")
    ax.set_title("Aggregation score distribution")
    style_axis(ax)
    add_median_labels(ax, df, "proagg_score", method_order)
    fig.tight_layout()
    agg_svg = args.output_prefix.with_name(args.output_prefix.name + "_aggregation.svg")
    agg_pdf = args.output_prefix.with_name(args.output_prefix.name + "_aggregation.pdf")
    fig.savefig(agg_svg, format="svg", bbox_inches="tight")
    fig.savefig(agg_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved stability SVG: {stability_svg}")
    print(f"Saved stability PDF: {stability_pdf}")
    print(f"Saved aggregation SVG: {agg_svg}")
    print(f"Saved aggregation PDF: {agg_pdf}")
    print(f"Saved long CSV: {args.output_prefix.with_name(args.output_prefix.name + '_long.csv')}")


if __name__ == "__main__":
    main()
