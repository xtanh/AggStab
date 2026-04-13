import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser(description='Plot stability predictor distribution from paired CSV.')
    parser.add_argument('--input_csv', type=Path, required=True)
    parser.add_argument('--target_col', type=str, default='deltaG')
    parser.add_argument('--pred_col', type=str, default='prediction')
    parser.add_argument('--output_prefix', type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[[args.target_col, args.pred_col]].dropna().astype(float)
    true_vals = df[args.target_col].to_numpy()
    pred_vals = df[args.pred_col].to_numpy()
    pcc = pearsonr(pred_vals, true_vals)[0]
    scc = spearmanr(pred_vals, true_vals)[0]

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    bins = 45
    ax.hist(true_vals, bins=bins, density=True, alpha=0.35, color='#059669', label='True $\\Delta G$')
    ax.hist(pred_vals, bins=bins, density=True, alpha=0.35, color='#2563eb', label='Predicted $\\Delta G$')
    x_lo = min(true_vals.min(), pred_vals.min())
    x_hi = max(true_vals.max(), pred_vals.max())
    xs = np.linspace(x_lo, x_hi, 400)
    ax.plot(xs, gaussian_kde(true_vals)(xs), color='#047857', linewidth=2.0)
    ax.plot(xs, gaussian_kde(pred_vals)(xs), color='#1d4ed8', linewidth=2.0)
    ax.set_xlabel('$\\Delta G$')
    ax.set_ylabel('Density')
    ax.set_title('Stability predictor on test set')
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)
    stats_text = f'n = {len(df)}\nPCC = {pcc:.4f}\nSCC = {scc:.4f}'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, va='top', ha='right', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#d1d5db', alpha=0.95))
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')
    with open(out_prefix.with_name(out_prefix.name + '_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'n: {len(df)}\n')
        f.write(f'pcc: {pcc:.10f}\n')
        f.write(f'scc: {scc:.10f}\n')


if __name__ == '__main__':
    main()
