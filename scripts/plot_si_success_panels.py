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


METHOD_ORDER = ["ESM-IF","ProteinMPNN", "SolubleMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ESM-IF": "#B8D2CC",
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "AggStab-DPO": "#E8B2A7",
}
FULL_CANDIDATE_FILES = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/full_candidates.csv"),
    "AggStab-DPO": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
}
THRESHOLD_ORDER = [
    "No structure filter",
    "pLDDT>0.75\npTM>0.60",
    "pLDDT>0.80\npTM>0.70",
    "pLDDT>0.85\npTM>0.80",
]


def load_property_success(raw_data_csv: Path, wt_agg_column: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_data_csv, usecols=["name", wt_agg_column]).rename(
        columns={"name": "pdb_name", wt_agg_column: "wt_proagg_score"}
    )
    rows = []
    for method in METHOD_ORDER:
        df = pd.read_csv(FULL_CANDIDATE_FILES[method])
        df = df.merge(raw, on="pdb_name", how="left", validate="many_to_one")
        rows.append(
            {
                "Method": method,
                "AntiAgg > WT": float((df["proagg_score"] > df["wt_proagg_score"]).mean()),
                "ΔG > WT": float((df["deltaG_minus_wt"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_bar(ax, summary: pd.DataFrame, value_col: str, title: str, ylabel: str) -> None:
    x = np.arange(len(METHOD_ORDER))
    vals = summary.set_index("Method").loc[METHOD_ORDER, value_col].to_numpy(dtype=float)
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    bars = ax.bar(x, vals, width=0.64, color=colors, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, min(1.0, max(vals) * 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(colors="#374151", width=0.8, length=4)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )


def _plot_line(ax, df: pd.DataFrame, value_col: str, title: str, ylabel: str, ylim: tuple[float, float]) -> None:
    x = np.arange(len(THRESHOLD_ORDER))
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].set_index("Threshold").reindex(THRESHOLD_ORDER)
        y = sub[value_col].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=6,
            linewidth=2.2,
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
    parser = argparse.ArgumentParser(description="Plot SI success panels for sequence and structure criteria.")
    parser.add_argument("--raw_data_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--wt_agg_column", type=str, default="log2_fold_change_75_clip")
    parser.add_argument(
        "--joint_threshold_csv",
        type=Path,
        default=Path("results/figures/threshold_sensitivity_lines_summary.csv"),
    )
    parser.add_argument(
        "--structure_threshold_csv",
        type=Path,
        default=Path("results/figures/structure_threshold_sensitivity_lines_summary.csv"),
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/si_success_panels"),
    )
    args = parser.parse_args()

    prop_summary = load_property_success(args.raw_data_csv, args.wt_agg_column)
    joint_df = pd.read_csv(args.joint_threshold_csv)
    structure_df = pd.read_csv(args.structure_threshold_csv)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prop_summary.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_property_summary.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.0), dpi=300)

    _plot_bar(
        axes[0, 0],
        prop_summary,
        value_col="AntiAgg > WT",
        title="Anti-aggregation success",
        ylabel="Fraction of generated sequences",
    )
    _plot_bar(
        axes[0, 1],
        prop_summary,
        value_col="ΔG > WT",
        title="Stability success",
        ylabel="Fraction of generated sequences",
    )
    _plot_line(
        axes[1, 0],
        joint_df,
        value_col="Joint success",
        title="Joint success under structure thresholds",
        ylabel="Joint success rate",
        ylim=(0.10, 0.86),
    )
    _plot_line(
        axes[1, 1],
        structure_df,
        value_col="Structure success",
        title="Structure-only success under thresholds",
        ylabel="Structure success rate",
        ylim=(0.35, 1.02),
    )

    panel_labels = ["a", "b", "c", "d"]
    for label, ax in zip(panel_labels, axes.flatten()):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    for ax in axes.flatten():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved property summary CSV: {args.output_prefix.with_name(args.output_prefix.name + '_property_summary.csv')}")


if __name__ == "__main__":
    main()
