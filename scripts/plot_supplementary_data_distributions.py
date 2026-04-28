#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


RAW_COLOR = "#2563eb"
DG_COLOR = "#059669"
REP_COLOR = "#6366f1"
SUBSET_COLOR = "#dc2626"
TRAIN_COLOR = "#111827"
THRESHOLD = -1.0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def _save(fig: plt.Figure, prefix: Path) -> None:
    svg = prefix.with_suffix('.svg')
    pdf = prefix.with_suffix('.pdf')
    fig.tight_layout()
    fig.savefig(svg, format='svg', bbox_inches='tight')
    fig.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved SVG: {svg}')
    print(f'Saved PDF: {pdf}')



def _plot_hist_with_kde(ax, values: np.ndarray, *, bins: int, hist_color: str, line_color: str, xlabel: str, title: str) -> None:
    ax.hist(values, bins=bins, density=True, color=hist_color, alpha=0.32, edgecolor='white', linewidth=0.5)
    xs = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=line_color, linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    style_axis(ax)



def plot_raw_aggregation(raw: pd.DataFrame, out_prefix: Path) -> None:
    values = raw['log2_fold_change_75_clip'].dropna().astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    _plot_hist_with_kde(
        ax,
        values,
        bins=48,
        hist_color='#93c5fd',
        line_color=RAW_COLOR,
        xlabel='Anti-aggregation label (log2_fold_change_75_clip)',
        title='Distribution of anti-aggregation labels in the raw dataset',
    )
    text = (
        f'n = {len(values):,}\n'
        f'mean = {values.mean():.3f}\n'
        f'median = {np.median(values):.3f}\n'
        f'std = {values.std(ddof=1):.3f}'
    )
    add_stats_box(ax, text)
    _save(fig, out_prefix)



def plot_raw_delta_g(raw: pd.DataFrame, dg: pd.DataFrame, out_prefix: Path) -> None:
    merged = raw[['name']].merge(dg[['name', 'deltaG']], on='name', how='inner', validate='one_to_one')
    values = merged['deltaG'].dropna().astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=300)
    _plot_hist_with_kde(
        ax,
        values,
        bins=48,
        hist_color='#a7f3d0',
        line_color=DG_COLOR,
        xlabel='Stability label (ΔG)',
        title='Distribution of stability labels in the raw dataset',
    )
    text = (
        f'n = {len(values):,}\n'
        f'mean = {values.mean():.3f}\n'
        f'median = {np.median(values):.3f}\n'
        f'std = {values.std(ddof=1):.3f}'
    )
    add_stats_box(ax, text)
    _save(fig, out_prefix)



def _kde_line(ax, values: np.ndarray, color: str, label: str) -> None:
    xs = np.linspace(values.min() - 0.2, values.max() + 0.2, 500)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=color, linewidth=2.2, label=label)



def plot_sampling_distribution(raw: pd.DataFrame, reps: pd.DataFrame, subset: pd.DataFrame, out_prefix: Path) -> None:
    raw_train = raw[raw['split'] == 'train'].copy()
    reps_train = reps[reps['split'] == 'train'].copy()
    subset_train = subset[subset['split'] == 'train'].copy()

    raw_vals = raw_train['log2_fold_change_75_clip'].dropna().astype(float).to_numpy()
    reps_vals = reps_train['log2_fold_change_75_clip'].dropna().astype(float).to_numpy()
    subset_vals = subset_train['log2_fold_change_75_clip'].dropna().astype(float).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=300)
    ax = axes[0]
    _kde_line(ax, raw_vals, TRAIN_COLOR, f'Raw train (n={len(raw_vals):,})')
    _kde_line(ax, reps_vals, REP_COLOR, f'Representative train (n={len(reps_vals):,})')
    _kde_line(ax, subset_vals, SUBSET_COLOR, f'Threshold-balanced train (n={len(subset_vals):,})')
    ax.axvline(THRESHOLD, color='#6b7280', linestyle='--', linewidth=1.5)
    ax.text(THRESHOLD + 0.05, ax.get_ylim()[1] * 0.92, 'threshold = -1.0', color='#4b5563', fontsize=10)
    ax.set_xlabel('Anti-aggregation label (log2_fold_change_75_clip)')
    ax.set_ylabel('Density')
    ax.set_title('Effect of subset sampling on the training-label distribution')
    ax.legend(frameon=False, loc='upper left')
    style_axis(ax)

    ax = axes[1]
    stages = ['Raw train', 'Representative train', 'Threshold-balanced train']
    lt_counts = [int((raw_vals < THRESHOLD).sum()), int((reps_vals < THRESHOLD).sum()), int((subset_vals < THRESHOLD).sum())]
    ge_counts = [int((raw_vals >= THRESHOLD).sum()), int((reps_vals >= THRESHOLD).sum()), int((subset_vals >= THRESHOLD).sum())]
    x = np.arange(len(stages))
    width = 0.34
    ax.bar(x - width / 2, lt_counts, width=width, color='#ef4444', label='label < -1.0')
    ax.bar(x + width / 2, ge_counts, width=width, color='#3b82f6', label='label ≥ -1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=10, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Threshold-group counts before and after sampling')
    ax.legend(frameon=False, loc='upper right')
    style_axis(ax)

    stats_df = pd.DataFrame(
        {
            'stage': stages,
            'n_total': [len(raw_vals), len(reps_vals), len(subset_vals)],
            'n_lt_threshold': lt_counts,
            'n_ge_threshold': ge_counts,
            'mean_label': [raw_vals.mean(), reps_vals.mean(), subset_vals.mean()],
            'median_label': [np.median(raw_vals), np.median(reps_vals), np.median(subset_vals)],
        }
    )
    stats_path = out_prefix.with_name(out_prefix.name + '_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f'Saved stats: {stats_path}')
    _save(fig, out_prefix)



def main() -> None:
    parser = argparse.ArgumentParser(description='Create supplementary data distribution figures.')
    parser.add_argument('--raw_csv', type=Path, default=Path('data/rocklin/rawdata/data.csv'))
    parser.add_argument('--dg_csv', type=Path, default=Path('data/rocklin/Metagenomic_dG.csv'))
    parser.add_argument('--representatives_csv', type=Path, default=Path('data/dpo/representative_pdbs/representatives.csv'))
    parser.add_argument('--subset_csv', type=Path, default=Path('data/dpo/subsets/threshold_balance_ltneg1_eq/subset_manifest.csv'))
    parser.add_argument('--output_dir', type=Path, default=Path('results/figures'))
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    dg = pd.read_csv(args.dg_csv)
    reps = pd.read_csv(args.representatives_csv)
    subset = pd.read_csv(args.subset_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_raw_aggregation(raw, args.output_dir / 'raw_log2fc75_distribution')
    plot_raw_delta_g(raw, dg, args.output_dir / 'raw_deltaG_distribution')
    plot_sampling_distribution(raw, reps, subset, args.output_dir / 'dpo_sampling_distribution')


if __name__ == '__main__':
    main()
