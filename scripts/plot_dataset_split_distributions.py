#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


SPLIT_ORDER = ["train", "valid", "test"]
SPLIT_COLORS = {
    "train": "#5B8DB8",
    "valid": "#BE6DB7",
    "test": "#E07B39",
}


def _save(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVG: {prefix.with_suffix('.svg')}")
    print(f"Saved PDF: {prefix.with_suffix('.pdf')}")


def _plot_split_kdes(
    split_to_values: dict[str, np.ndarray],
    *,
    xlabel: str,
    title: str,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=300)

    all_values = np.concatenate([split_to_values[s] for s in SPLIT_ORDER if len(split_to_values[s]) > 1])
    x_min, x_max = float(all_values.min()), float(all_values.max())
    xs = np.linspace(x_min, x_max, 500)

    for split in SPLIT_ORDER:
        values = split_to_values[split]
        color = SPLIT_COLORS[split]
        kde = gaussian_kde(values)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=2.0, label=f"{split} (n={len(values):,})")
        ax.fill_between(xs, ys, color=color, alpha=0.18)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)

    text = "\n".join(
        f"{split}: median={np.median(split_to_values[split]):.3f}"
        for split in SPLIT_ORDER
    )
    add_stats_box(ax, text, x=0.03, y=0.97)
    _save(fig, output_prefix)


def _plot_split_length_bars(
    split_to_values: dict[str, np.ndarray],
    *,
    xlabel: str,
    title: str,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=300)
    lengths = sorted(set(int(v) for values in split_to_values.values() for v in values))
    x = np.arange(len(lengths))
    width = 0.24

    for idx, split in enumerate(SPLIT_ORDER):
        values = split_to_values[split].astype(int)
        counts = pd.Series(values).value_counts().reindex(lengths, fill_value=0).to_numpy()
        ax.bar(
            x + (idx - 1) * width,
            counts,
            width=width,
            color=SPLIT_COLORS[split],
            alpha=0.9,
            label=f"{split} (n={len(values):,})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    tick_idx = x[::2] if len(x) > 18 else x
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([lengths[i] for i in tick_idx])
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    _save(fig, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/valid/test distributions for SI.")
    parser.add_argument("--raw_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--dg_csv", type=Path, default=Path("data/rocklin/Metagenomic_dG.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv, usecols=["name", "protein_sequence", "log2_fold_change_75_clip", "split"])
    dg = pd.read_csv(args.dg_csv, usecols=["name", "deltaG"])
    merged = raw.merge(dg, on="name", how="inner", validate="one_to_one")
    raw["length"] = raw["protein_sequence"].astype(str).str.len().astype(float)

    agg = {split: raw.loc[raw["split"] == split, "log2_fold_change_75_clip"].dropna().astype(float).to_numpy() for split in SPLIT_ORDER}
    dgun = {split: merged.loc[merged["split"] == split, "deltaG"].dropna().astype(float).to_numpy() for split in SPLIT_ORDER}
    length = {split: raw.loc[raw["split"] == split, "length"].dropna().astype(float).to_numpy() for split in SPLIT_ORDER}

    _plot_split_kdes(
        agg,
        xlabel="Anti-aggregation label (`log2_fold_change_75_clip`)",
        title="Split-wise distribution of anti-aggregation labels",
        output_prefix=args.output_dir / "si_split_log2fc75_distribution",
    )
    _plot_split_kdes(
        dgun,
        xlabel="Stability label (`ΔG_unfolding`)",
        title="Split-wise distribution of stability labels",
        output_prefix=args.output_dir / "si_split_dgunfolding_distribution",
    )
    _plot_split_length_bars(
        length,
        xlabel="Protein length (aa)",
        title="Split-wise distribution of protein lengths",
        output_prefix=args.output_dir / "si_split_length_distribution",
    )


if __name__ == "__main__":
    main()
