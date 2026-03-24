"""
Plot label distribution histograms for train/val/test separately.

Usage:
    python scripts/plot_label_distribution.py \
        --output_dir results/datasets_distribution_plots
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12


def load_data():
    """Load train/val/test data."""
    data_dir = Path("/home/xy_th/Project_protein_aggregation/data/rocklin")

    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    test = pd.read_csv(data_dir / "test.csv")

    return train, valid, test


def plot_single_histogram(data, name, output_path, color):
    """Plot a single histogram for one split."""

    label_col = "log2_fold_change_75_clip"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    bins = np.linspace(-6.5, 1.5, 50)
    counts, bin_edges, patches = ax.hist(
        data[label_col],
        bins=bins,
        color=color,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add threshold lines
    ax.axvline(x=-1.0, color='red', linestyle='--', linewidth=2, label='threshold=-1 (high agg)')
    ax.axvline(x=0, color='darkgreen', linestyle='-', linewidth=2, label='threshold=0 (low agg)')

    # Add statistics text box
    stats_text = (
        f"n = {len(data):,}\n"
        f"min = {data[label_col].min():.3f}\n"
        f"max = {data[label_col].max():.3f}\n"
        f"mean = {data[label_col].mean():.3f}\n"
        f"median = {data[label_col].median():.3f}\n"
        f"std = {data[label_col].std():.3f}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Add percentage annotations for each region
    high_agg = (data[label_col] < -1).sum()
    medium_agg = ((data[label_col] >= -1) & (data[label_col] < 0)).sum()
    low_agg = (data[label_col] >= 0).sum()
    total = len(data)

    # Position text annotations
    y_max = ax.get_ylim()[1]
    ax.text(
        -3.5, y_max * 0.85,
        f'High agg (<-1)\n{high_agg} ({high_agg/total*100:.1f}%)',
        ha='center', fontsize=11, color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8)
    )
    ax.text(
        -0.5, y_max * 0.85,
        f'Medium agg (-1~0)\n{medium_agg} ({medium_agg/total*100:.1f}%)',
        ha='center', fontsize=11, color='darkorange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    ax.text(
        0.5, y_max * 0.85,
        f'Low agg (>0)\n{low_agg} ({low_agg/total*100:.1f}%)',
        ha='center', fontsize=11, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    )

    ax.set_xlabel("log2_fold_change_75_clip", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"{name} Set Label Distribution (n={len(data):,})", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(-6.5, 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot label distribution histograms")
    parser.add_argument("--output_dir", type=str,
                        default="/home/xy_th/Project_protein_aggregation/results/datasets_distribution_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    train, valid, test = load_data()

    print(f"Train: {len(train):,} samples")
    print(f"Valid: {len(valid):,} samples")
    print(f"Test:  {len(test):,} samples")

    print("\nGenerating histograms...")

    plot_single_histogram(
        train, "Train",
        os.path.join(args.output_dir, "train_distribution.png"),
        color="#1f77b4"  # blue
    )

    plot_single_histogram(
        valid, "Validation",
        os.path.join(args.output_dir, "valid_distribution.png"),
        color="#ff7f0e"  # orange
    )

    plot_single_histogram(
        test, "Test",
        os.path.join(args.output_dir, "test_distribution.png"),
        color="#2ca02c"  # green
    )

    print(f"\nAll plots saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - train_distribution.png")
    print("  - valid_distribution.png")
    print("  - test_distribution.png")


if __name__ == "__main__":
    main()
