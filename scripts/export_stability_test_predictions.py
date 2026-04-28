import argparse
from pathlib import Path

import os
import sys

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('scripts')]
sys.path.insert(0, PROJ_DIR)

from src.dpo.sample_and_score_dual import load_predictor
from src.ln.lightning_data import _sa_seq_to_spaced


def main():
    parser = argparse.ArgumentParser(description='Export stability predictor test-set predictions.')
    parser.add_argument('--input_csv', type=Path, default=Path('data/rocklin/rawdata/data.csv'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/proagg_deltaG_only.yaml'))
    parser.add_argument('--stability_csv', type=Path, default=Path('data/rocklin/Metagenomic_dG.csv'))
    parser.add_argument('--output_csv', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df['split'] == 'test'].copy()
    dg_df = pd.read_csv(args.stability_csv)[['name', 'deltaG']].dropna()
    df = df.merge(dg_df, on='name', how='inner')
    df = df[['name', 'protein_sequence', 'sa_sequence_foldseek', 'deltaG']].dropna().reset_index(drop=True)

    device = torch.device(args.device)
    model, tokenizer, output_key = load_predictor(str(args.checkpoint), str(args.config), device)
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start:start + args.batch_size]
            spaced_sequences = [_sa_seq_to_spaced(sa) for sa in batch_df['sa_sequence_foldseek'].tolist()]
            encoded = tokenizer.batch_encode_plus(spaced_sequences, return_tensors='pt', padding=True)
            batch = {
                'input_ids': encoded['input_ids'].to(device),
                'attention_mask': encoded['attention_mask'].to(device),
            }
            outputs = model(batch)
            all_preds.append(outputs[output_key].detach().cpu().flatten())
    preds = torch.cat(all_preds, dim=0).numpy()

    df['prediction'] = preds
    df['target'] = df['deltaG'].astype(float)
    df['abs_error'] = (df['prediction'] - df['target']).abs()
    df['signed_error'] = df['prediction'] - df['target']

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f'n: {len(df)}')
    print(f'pcc: {pearsonr(df["prediction"], df["target"])[0]:.10f}')
    print(f'scc: {spearmanr(df["prediction"], df["target"])[0]:.10f}')
    print(f'saved: {args.output_csv}')


if __name__ == '__main__':
    main()
