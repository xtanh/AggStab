#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "SFT", "DPO+SFT", "ESM-IF"]
METHOD_COLORS = {
    "ProteinMPNN": "#6b7280",
    "SolubleMPNN": "#2563eb",
    "SFT": "#059669",
    "DPO+SFT": "#dc2626",
    "ESM-IF": "#7c3aed",
}

RUNS = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "SFT": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "DPO+SFT": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/full_candidates.csv"),
}


def compute_summary(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    tmp = df.copy()
    tmp["joint_positive"] = (tmp["proagg_score"] > 0) & (tmp["deltaG_minus_wt"] > 0)
    backbone = tmp.groupby("pdb_name").agg(
        proagg_mean_over16=("proagg_score", "mean"),
        dgunfolding_minus_wt_mean_over16=("deltaG_minus_wt", "mean"),
        any_joint_positive=("joint_positive", "any"),
    )
    mpnn_logprob_metric = (
        float(df.groupby("pdb_name")["mpnn_logprob"].mean().mean())
        if "mpnn_logprob" in df.columns and df["mpnn_logprob"].notna().any()
        else float("nan")
    )
    return {
        "Mean anti-aggregation score": float(backbone["proagg_mean_over16"].mean()),
        "Mean ΔG_unfolding improvement over WT": float(backbone["dgunfolding_minus_wt_mean_over16"].mean()),
        "Mean ProteinMPNN log-likelihood": mpnn_logprob_metric,
        "Fraction with ≥1 joint-positive candidate": float(backbone["any_joint_positive"].mean()),
    }


def add_value_labels(ax, bars, values, fmt: str) -> None:
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value >= 0:
            y = value + 0.02 * yrange
            va = "bottom"
        else:
            y = value - 0.03 * yrange
            va = "top"
        ax.text(x, y, format(value, fmt), ha="center", va=va, fontsize=9, color="#111827")


def plot_panel(ax, methods: list[str], values: list[float], title: str, ylabel: str, fmt: str) -> None:
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    add_value_labels(ax, bars, values, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generated sequence property summaries across methods.")
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/generated_sequence_properties_over16"),
    )
    args = parser.parse_args()

    summary = {method: compute_summary(path) for method, path in RUNS.items()}
    summary_df = pd.DataFrame(summary).T.loc[METHOD_ORDER]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary_df.to_csv(summary_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), dpi=300)
    axes = axes.flatten()

    metrics = [
        ("Mean anti-aggregation score", "Generated sequence anti-aggregation", ".4f"),
        ("Mean ΔG_unfolding improvement over WT", "Generated sequence stability", ".4f"),
        ("Mean ProteinMPNN log-likelihood", "ProteinMPNN prior consistency", ".4f"),
        ("Fraction with ≥1 joint-positive candidate", "Joint-positive backbone coverage", ".3f"),
    ]

    for ax, (metric, title, fmt) in zip(axes, metrics):
        vals = [float(summary_df.loc[m, metric]) for m in METHOD_ORDER]
        plot_panel(ax, METHOD_ORDER, vals, title, metric, fmt)

    fig.tight_layout()
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
