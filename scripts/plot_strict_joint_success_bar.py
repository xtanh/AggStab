#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "ESM-IF", "MoMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "MoMPNN": "#F0C987",
    "AggStab-DPO": "#E8B2A7",
    "AggStab-SFT": "#A7C4BC",
}
REPORTS = {
    "ProteinMPNN": Path("results/joint_structure_vs_wt_analysis_label75/joint_structure_vs_wt_report.json"),
    "SolubleMPNN": Path("results/joint_structure_vs_wt_analysis_label75_soluble/joint_structure_vs_wt_report.json"),
    "AggStab-SFT": Path("results/joint_structure_vs_wt_analysis_label75_sft/joint_structure_vs_wt_report.json"),
    "AggStab-DPO": Path("results/joint_structure_vs_wt_analysis_label75_dpo_sft10/joint_structure_vs_wt_report.json"),
}


def load_summary(method_order: list[str], threshold_summary_csv: Path | None) -> pd.DataFrame:
    rows = []
    if threshold_summary_csv is not None and threshold_summary_csv.exists():
        summary = pd.read_csv(threshold_summary_csv)
        strict = summary[summary["Threshold"] == "pLDDT>0.85\npTM>0.80"].copy()
        strict = strict.set_index("Method")
        for method in method_order:
            rows.append({"Method": method, "Strict joint success": float(strict.loc[method, "Joint success"])})
        return pd.DataFrame(rows)

    for method in method_order:
        path = REPORTS[method]
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if method == "ProteinMPNN":
            value = float(report["backbone_mean_of_top3"]["baseline"]["joint_success_rate"])
        else:
            value = float(report["backbone_mean_of_top3"]["best"]["joint_success_rate"])
        rows.append({"Method": method, "Strict joint success": value})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strict joint success bar chart across methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=["ProteinMPNN", "SolubleMPNN", "ESM-IF", "MoMPNN", "AggStab-SFT", "AggStab-DPO"],
        help="Methods to include.",
    )
    parser.add_argument(
        "--threshold_summary_csv",
        type=Path,
        default=Path("results/figures/threshold_sensitivity_lines_summary.csv"),
        help="Optional summary CSV to read strict joint success from.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/strict_joint_success_bar"),
    )
    args = parser.parse_args()

    method_order = args.methods
    summary = load_summary(method_order, args.threshold_summary_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    x = np.arange(len(method_order))
    vals = summary["Strict joint success"].to_numpy(dtype=float)
    colors = [METHOD_COLORS[m] for m in method_order]
    bars = ax.bar(x, vals, color=colors, width=0.64)

    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    ax.set_ylabel("Strict joint success rate")
    ax.set_ylim(0, max(vals) * 1.18)
    style_axis(ax)

    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )

    fig.tight_layout()
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
