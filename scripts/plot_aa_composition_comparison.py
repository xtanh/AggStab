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
from matplotlib.colors import LinearSegmentedColormap

AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
AA_GROUPS = {
    'Hydrophobic': set('AVILM'),
    'Aromatic': set('FWY'),
    'Polar': set('STNQ'),
    'Positive': set('KRH'),
    'Negative': set('DE'),
    'Special': set('CGP'),
}


def normalize_counts(sequences: list[str]) -> pd.Series:
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


def group_freq(freq: pd.Series) -> pd.Series:
    out = {}
    for name, aas in AA_GROUPS.items():
        out[name] = float(freq.loc[[aa for aa in AA_ORDER if aa in aas]].sum())
    return pd.Series(out)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot amino-acid composition comparison across methods.')
    parser.add_argument('--wt_csv', type=Path, default=Path('data/dpo/representative_pdbs/representatives.csv'))
    parser.add_argument('--baseline_csv', type=Path, default=Path('results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv'))
    parser.add_argument('--dpo_csv', type=Path, default=Path('results/semi_joint_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv'))
    parser.add_argument('--sft_csv', type=Path, default=Path('results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv'))
    parser.add_argument('--hybrid_csv', type=Path, default=Path('results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv'))
    parser.add_argument('--output_prefix', type=Path, default=Path('results/figures/aa_composition_top3_methods'))
    args = parser.parse_args()

    wt_df = pd.read_csv(args.wt_csv)
    wt_df = wt_df[wt_df['split'] == 'test'].copy()
    datasets = {
        'WT test set': wt_df['protein_sequence'].dropna().astype(str).tolist(),
        'ProteinMPNN': pd.read_csv(args.baseline_csv)['sequence'].dropna().astype(str).tolist(),
        'Pure DPO': pd.read_csv(args.dpo_csv)['sequence'].dropna().astype(str).tolist(),
        'Pure SFT': pd.read_csv(args.sft_csv)['sequence'].dropna().astype(str).tolist(),
        'DPO+SFT (1.0)': pd.read_csv(args.hybrid_csv)['sequence'].dropna().astype(str).tolist(),
    }

    aa_freq = pd.DataFrame({name: normalize_counts(seqs) for name, seqs in datasets.items()}).T
    group_freqs = pd.DataFrame({name: group_freq(aa_freq.loc[name]) for name in aa_freq.index}).T
    wt = aa_freq.loc['WT test set'].to_numpy()
    js = pd.Series({name: js_divergence(aa_freq.loc[name].to_numpy(), wt) for name in aa_freq.index if name != 'WT test set'})

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    aa_freq.to_csv(args.output_prefix.with_name(args.output_prefix.name + '_aa_freq.csv'))
    group_freqs.to_csv(args.output_prefix.with_name(args.output_prefix.name + '_group_freq.csv'))
    js.to_frame('js_divergence_vs_wt').to_csv(args.output_prefix.with_name(args.output_prefix.name + '_js.csv'))

    centered = aa_freq.subtract(aa_freq.loc['WT test set'], axis=1)
    plot_df = centered.loc[['ProteinMPNN', 'Pure DPO', 'Pure SFT', 'DPO+SFT (1.0)']]
    cmap = LinearSegmentedColormap.from_list('custom_div', ['#1d4ed8', '#ffffff', '#b91c1c'])

    fig = plt.figure(figsize=(12.4, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.25, 1.25, 0.9], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(plot_df.values, aspect='auto', cmap=cmap, vmin=-0.035, vmax=0.035)
    ax0.set_xticks(np.arange(len(AA_ORDER)))
    ax0.set_xticklabels(AA_ORDER)
    ax0.set_yticks(np.arange(len(plot_df.index)))
    ax0.set_yticklabels(plot_df.index)
    ax0.set_title('Amino-acid frequency shift relative to WT test set')
    ax0.set_xlabel('Amino acid')
    for spine in ax0.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label('Frequency difference')

    ax1 = fig.add_subplot(gs[0, 1])
    grp_plot = group_freqs.loc[['WT test set', 'ProteinMPNN', 'Pure DPO', 'Pure SFT', 'DPO+SFT (1.0)']]
    x = np.arange(len(AA_GROUPS))
    width = 0.16
    colors = ['#111827', '#2563eb', '#7c3aed', '#059669', '#dc2626']
    for idx, row_name in enumerate(grp_plot.index):
        ax1.bar(x + (idx - 2) * width, grp_plot.loc[row_name].values, width=width, label=row_name, color=colors[idx])
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(AA_GROUPS.keys()), rotation=22, ha='right')
    ax1.set_ylabel('Fraction of residues')
    ax1.set_title('Residue-property composition')
    ax1.legend(frameon=False, fontsize=8, loc='upper right')
    style_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    js_plot = js.loc[['ProteinMPNN', 'Pure DPO', 'Pure SFT', 'DPO+SFT (1.0)']]
    ax2.barh(np.arange(len(js_plot)), js_plot.values, color=['#2563eb', '#7c3aed', '#059669', '#dc2626'])
    ax2.set_yticks(np.arange(len(js_plot)))
    ax2.set_yticklabels(js_plot.index)
    ax2.invert_yaxis()
    ax2.set_xlabel('JS divergence')
    ax2.set_title('Distance to WT composition')
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix('.svg'), format='svg', bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved SVG: {args.output_prefix.with_suffix('.svg')}")
    print(f"Saved PDF: {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved AA freq CSV: {args.output_prefix.with_name(args.output_prefix.name + '_aa_freq.csv')}")
    print(f"Saved group freq CSV: {args.output_prefix.with_name(args.output_prefix.name + '_group_freq.csv')}")
    print(f"Saved JS CSV: {args.output_prefix.with_name(args.output_prefix.name + '_js.csv')}")


if __name__ == '__main__':
    main()
