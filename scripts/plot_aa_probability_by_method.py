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

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "ESM-IF", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "AggStab-DPO": "#E8B2A7",
    "AggStab-SFT": "#A7C4BC",
    "MoMPNN[Sol+TM]": "#DCC7AA",
    "MoMPNN[Sol+IG+ESM]": "#F0C987",
}
RUNS = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/full_candidates.csv"),
    "AggStab-SFT": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "AggStab-DPO": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "MoMPNN[Sol+TM]": Path("results/mompnn_protsol_tm_fulltest_n16_joint/full_candidates.csv"),
    "MoMPNN[Sol+IG+ESM]": Path("results/mompnn_protsol_ig_esm_fulltest_n16_joint/full_candidates.csv"),
}


def aa_frequency_from_sequences(sequences: list[str]) -> pd.Series:
    counts = pd.Series(0.0, index=AA_ORDER)
    total = 0
    for seq in sequences:
        for aa in seq:
            if aa in counts.index:
                counts[aa] += 1
                total += 1
    if total == 0:
        return counts
    return counts / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot amino-acid probability across generated sequences for each method.")
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
        default=Path("results/figures/aa_probability_fig3_main_methods"),
    )
    args = parser.parse_args()

    method_order = args.methods
    freq_table = {}
    for method in method_order:
        csv_path = RUNS[method]
        df = pd.read_csv(csv_path)
        seqs = df["sequence"].dropna().astype(str).tolist()
        freq_table[method] = aa_frequency_from_sequences(seqs)

    freq_df = pd.DataFrame(freq_table).T.loc[method_order]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    freq_csv = args.output_prefix.with_name(args.output_prefix.name + "_freq.csv")
    freq_df.to_csv(freq_csv)

    anchor_method = method_order[0]
    sort_idx = np.argsort(freq_df.loc[anchor_method, AA_ORDER].to_numpy(dtype=float))[::-1]
    sorted_aas = [AA_ORDER[i] for i in sort_idx]
    sorted_df = freq_df.loc[method_order, sorted_aas]

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    x = np.arange(len(sorted_aas))
    bar_w = 0.75 / len(method_order)
    n_methods = len(method_order)

    for i, method in enumerate(method_order):
        vals = sorted_df.loc[method].to_numpy(dtype=float)
        offset = (i - n_methods / 2 + 0.5) * bar_w
        ax.bar(
            x + offset,
            vals,
            width=bar_w,
            color=METHOD_COLORS[method],
            alpha=0.85,
            label=method,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_aas, fontsize=10)
    ax.set_xlabel("Amino acid", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Amino acid probability distribution", fontsize=13, pad=8)

    fig.tight_layout()

    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved frequency CSV: {freq_csv}")
    print(sorted_df.to_string())


if __name__ == "__main__":
    main()
