import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from src.dpo.sample_and_score_dual import load_predictor, score_sequences_with_predictor


def main():
    parser = argparse.ArgumentParser(description='Plot stability predictor distribution on test set.')
    parser.add_argument('--input_csv', type=Path, default=Path('data/rocklin/rawdata/data.csv'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/proagg_deltaG_only.yaml'))
    parser.add_argument('--output_prefix', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df['split'] == 'test'].copy()
    df = df[['name', 'protein_sequence', 'sa_sequence_foldseek', 'deltaG']].dropna().reset_index(drop=True)

    device = torch.device(args.device)
    model, tokenizer, output_key = load_predictor(str(args.checkpoint), str(args.config), device)
    preds = score_sequences_with_predictor(
        predictor_model=model,
        output_key=output_key,
        sequences=df['protein_sequence'].tolist(),
        struct_tokens=df['sa_sequence_foldseek'].tolist(),
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    ).cpu().numpy()

    df['prediction'] = preds
    df['target'] = df['deltaG'].astype(float)

    pcc = pearsonr(df['prediction'], df['target'])[0]
    scc = spearmanr(df['prediction'], df['target'])[0]

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_name(out_prefix.name + '_points.csv'), index=False)
    with open(out_prefix.with_name(out_prefix.name + '_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(f'n: {len(df)}\n')
        f.write(f'pcc: {pcc:.10f}\n')
        f.write(f'scc: {scc:.10f}\n')

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)

    true_vals = df['target'].to_numpy()
    pred_vals = df['prediction'].to_numpy()

    bins = 45
    ax.hist(true_vals, bins=bins, density=True, alpha=0.35, color='#059669', label='True $\\Delta G$')
    ax.hist(pred_vals, bins=bins, density=True, alpha=0.35, color='#2563eb', label='Predicted $\\Delta G$')

    x_lo = min(true_vals.min(), pred_vals.min())
    x_hi = max(true_vals.max(), pred_vals.max())
    xs = torch.linspace(float(x_lo), float(x_hi), 400).numpy()
    ax.plot(xs, gaussian_kde(true_vals)(xs), color='#047857', linewidth=2.0)
    ax.plot(xs, gaussian_kde(pred_vals)(xs), color='#1d4ed8', linewidth=2.0)

    ax.set_xlabel('$\\Delta G$')
    ax.set_ylabel('Density')
    ax.set_title('Stability predictor on test set')
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)

    stats_text = f'n = {len(df)}\nPCC = {pcc:.4f}\nSCC = {scc:.4f}'
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va='top',
        ha='right',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#d1d5db', alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
