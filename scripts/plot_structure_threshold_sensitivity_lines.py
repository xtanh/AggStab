#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
RUNS = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/chai_top3_joint"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/chai_top3_joint"),
    "MoMPNN": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/chai_top3_joint"),
    "AggStab-SFT": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint"),
    "AggStab-DPO": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint"),
}
THRESHOLDS = [
    ("No structure filter", None, None),
    ("pLDDT>0.75\npTM>0.60", 0.75, 0.60),
    ("pLDDT>0.80\npTM>0.70", 0.80, 0.70),
    ("pLDDT>0.85\npTM>0.80", 0.85, 0.80),
]


def load_summary_dir(path: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(path.glob("summary_shard*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No summary_shard*.csv found in {path}")
    return pd.concat(frames, ignore_index=True)


def compute_structure_success_curve(df: pd.DataFrame) -> list[float]:
    values = []
    for _, plddt_thr, ptm_thr in THRESHOLDS:
        if plddt_thr is None:
            structure_pass = pd.Series(True, index=df.index)
        else:
            structure_pass = (df["best_plddt"] > plddt_thr) & (df["best_ptm"] > ptm_thr)
        per_backbone = structure_pass.groupby(df["pdb_name"]).mean()
        values.append(float(per_backbone.mean()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structure-only threshold sensitivity lines.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=list(RUNS.keys()),
        help="Methods to include.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/structure_threshold_sensitivity_lines"),
    )
    args = parser.parse_args()

    method_order = args.methods
    summary_rows = []
    xlabels = [name for name, _, _ in THRESHOLDS]

    for method in method_order:
        df = load_summary_dir(RUNS[method])
        vals = compute_structure_success_curve(df)
        for label, value in zip(xlabels, vals):
            summary_rows.append(
                {
                    "Method": method,
                    "Threshold": label,
                    "Structure success": value,
                }
            )

    summary = pd.DataFrame(summary_rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    x = np.arange(len(xlabels))
    for method in method_order:
        sub = summary[summary["Method"] == method].set_index("Threshold").reindex(xlabels)
        y = sub["Structure success"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.6,
            color=METHOD_COLORS[method],
            label=method,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=1.2,
            solid_capstyle="round",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("Structure success rate", fontsize=12)
    ax.set_xlabel("Structure filtering criterion", fontsize=12)
    ax.set_title("Sensitivity to structure-quality thresholds", fontsize=13, pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.35, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(colors="#374151", width=0.8, length=4)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6, color="#9ca3af")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(4, len(method_order)),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
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


if __name__ == "__main__":
    main()
