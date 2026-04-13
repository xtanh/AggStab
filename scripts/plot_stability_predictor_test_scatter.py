import argparse
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis


def main():
    configure_matplotlib()

    parser = argparse.ArgumentParser(description='Plot stability predictor test scatter.')
    parser.add_argument('--input_csv', type=Path, required=True)
    parser.add_argument('--x_col', type=str, default='target')
    parser.add_argument('--y_col', type=str, default='prediction')
    parser.add_argument('--output_prefix', type=Path, required=True)
    parser.add_argument('--xlabel', type=str, default='True deltaG')
    parser.add_argument('--ylabel', type=str, default='Predicted deltaG')
    parser.add_argument('--title', type=str, default='Stability predictor on test set')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    plot_df = df[[args.x_col, args.y_col]].dropna().astype(float)
    x = plot_df[args.x_col]
    y = plot_df[args.y_col]

    pcc = pearsonr(x, y)[0]
    scc = spearmanr(x, y)[0]

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
    ax.scatter(x, y, s=14, alpha=0.24, color='#059669', edgecolors='none', rasterized=True)

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())
    pad = 0.05 * (upper - lower)
    lo, hi = lower - pad, upper + pad
    diag_x = np.linspace(lo, hi, 200)
    ax.plot(diag_x, diag_x, linestyle='--', linewidth=1.4, color='#111827', alpha=0.85, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    style_axis(ax)

    stats_text = f'n = {len(plot_df)}\nPCC = {pcc:.4f}\nSCC = {scc:.4f}'
    add_stats_box(ax, stats_text)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')
    plot_df.to_csv(out_prefix.with_name(out_prefix.name + '_points.csv'), index=False)
    with open(out_prefix.with_name(out_prefix.name + '_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'input_csv: {args.input_csv}\n')
        f.write(f'n: {len(plot_df)}\n')
        f.write(f'pcc: {pcc:.10f}\n')
        f.write(f'scc: {scc:.10f}\n')


if __name__ == '__main__':
    main()
