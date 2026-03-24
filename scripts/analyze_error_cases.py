"""Analyze error cases on train and validation sets for ProAgg v33.

This script loads a trained model and analyzes:
1. Largest absolute errors
2. Ranking inconsistencies
3. Error distribution by score ranges
4. Hotspot attention analysis for error cases
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get project root (parent of scripts directory)
PROJ_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.models.factory import build_proagg_model
from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_ranking import LightningProAggModelRanking
import pytorch_lightning as pl


def compute_spearman(pred, tgt):
    """Compute Spearman correlation."""
    from scipy.stats import spearmanr
    return spearmanr(pred.cpu().numpy(), tgt.cpu().numpy())[0]


def analyze_split(model, dataloader, device, split_name, cfg):
    """Analyze errors for a data split."""
    model.eval()

    all_results = []

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward with interpretation
            out = model(batch, return_interpretation=True)

            pred = out['score'].flatten()
            tgt = batch['score'].flatten()
            sa_sequences = batch.get('sa_sequence', [None] * len(pred))

            # Per-sample results
            for i in range(len(pred)):
                result = {
                    'sa_sequence': sa_sequences[i] if isinstance(sa_sequences, list) else sa_sequences[i].cpu().numpy(),
                    'protein_sequence': batch.get('protein_sequence', [None])[i] if isinstance(batch.get('protein_sequence'), list) else None,
                    'predicted': pred[i].item(),
                    'target': tgt[i].item(),
                    'absolute_error': abs(pred[i].item() - tgt[i].item()),
                    'squared_error': (pred[i].item() - tgt[i].item()) ** 2,
                }

                # Add interpretation info if available
                if 'interpretation' in out:
                    interp = out['interpretation'][i] if isinstance(out['interpretation'], list) else out['interpretation']
                    result['hotspot_indices'] = interp.get('hotspot_indices', [])
                    result['hotspot_scores'] = interp.get('hotspot_scores', [])
                    result['top_attributed_indices'] = interp.get('top_attributed_indices', [])

                all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    df['error'] = df['predicted'] - df['target']

    print(f"\n{'='*70}")
    print(f"Error Analysis: {split_name} Set")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"MSE: {df['squared_error'].mean():.4f}")
    print(f"MAE: {df['absolute_error'].mean():.4f}")
    print(f"Spearman: {compute_spearman(torch.tensor(df['predicted'].values), torch.tensor(df['target'].values)):.4f}")

    # 1. Largest Absolute Errors
    print(f"\n--- Top 10 Largest Absolute Errors ---")
    worst = df.nlargest(10, 'absolute_error')
    for idx, row in worst.iterrows():
        print(f"  True: {row['target']:+.2f}, Pred: {row['predicted']:+.2f}, Error: {row['absolute_error']:.2f}")
        seq = row['sa_sequence']
        if seq and len(seq) > 20:
            seq_str = str(seq)[:50] + "..." if len(str(seq)) > 50 else str(seq)
            print(f"    Seq: {seq_str}")
        if 'hotspot_indices' in row and row['hotspot_indices']:
            print(f"    Hotspots: {row['hotspot_indices'][:5]}")

    # 2. Error by Target Range
    print(f"\n--- Error Distribution by Target Range ---")
    bins = [-float('inf'), -3, -2, -1, 0, float('inf')]
    labels = ['<-3', '-3~-2', '-2~-1', '-1~0', '>0']
    df['target_bin'] = pd.cut(df['target'], bins=bins, labels=labels)

    for label in labels:
        subset = df[df['target_bin'] == label]
        if len(subset) > 0:
            print(f"  {label:8s}: n={len(subset):4d}, MAE={subset['absolute_error'].mean():.3f}, "
                  f"Spearman={compute_spearman(torch.tensor(subset['predicted'].values), torch.tensor(subset['target'].values)):.3f}")

    # 3. Ranking Errors (inversions)
    print(f"\n--- Ranking Analysis ---")
    pred_tensor = torch.tensor(df['predicted'].values)
    tgt_tensor = torch.tensor(df['target'].values)

    # Compute pairwise inversions
    tgt_diff = tgt_tensor.unsqueeze(0) - tgt_tensor.unsqueeze(1)
    pred_diff = pred_tensor.unsqueeze(0) - pred_tensor.unsqueeze(1)

    # Valid pairs (significant difference)
    valid_pairs = torch.abs(tgt_diff) > 0.3
    n_valid = valid_pairs.sum().item() - len(df)  # Exclude diagonal

    # Inversions (sign mismatch)
    inversions = (tgt_diff * pred_diff < 0) & valid_pairs
    n_inversions = inversions.sum().item()

    print(f"  Valid pairs (|diff|>0.3): {n_valid}")
    print(f"  Ranking inversions: {n_inversions} ({100*n_inversions/max(n_valid,1):.1f}%)")

    # Largest ranking errors
    print(f"\n--- Top 10 Ranking Errors (largest |tgt_diff| with wrong sign) ---")
    inversion_scores = torch.abs(tgt_diff) * inversions.float()
    top_inversions = torch.topk(inversion_scores.flatten(), k=min(10, n_inversions))

    for rank, (flat_idx, score) in enumerate(zip(top_inversions.indices, top_inversions.values)):
        if score == 0:
            break
        i = int(flat_idx // len(df))
        j = int(flat_idx % len(df))
        print(f"  {rank+1}. Sample {i} (tgt={df.iloc[i]['target']:+.2f}) vs Sample {j} (tgt={df.iloc[j]['target']:+.2f})")
        print(f"     Pred: {df.iloc[i]['predicted']:+.2f} vs {df.iloc[j]['predicted']:+.2f} (should be opposite)")

    # 4. Systematic bias analysis
    print(f"\n--- Systematic Bias ---")
    print(f"  Mean error (pred - tgt): {df['error'].mean():+.3f}")
    print(f"  Std error: {df['error'].std():.3f}")

    # Bias by range
    for label in labels:
        subset = df[df['target_bin'] == label]
        if len(subset) > 0:
            print(f"  {label:8s}: mean_error={subset['error'].mean():+.3f} (positive=overestimate)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze error cases for ProAgg v33")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/proagg_v33_interpretable.yaml",
                        help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    # Load config
    cfg = load_yaml_config(args.config)

    # Load model using Lightning
    print(f"Loading checkpoint: {args.ckpt}")
    pl_module = LightningProAggModelRanking.load_from_checkpoint(args.ckpt, cfg=cfg)
    model = pl_module.model
    model = model.to(args.device)
    model.eval()

    # Data module
    dm = ProAggDataModule(cfg)
    dm.setup(stage='fit')

    # Analyze train set
    train_loader = dm.train_dataloader()
    train_df = analyze_split(model, train_loader, args.device, "Train", cfg)

    # Analyze val set
    val_loader = dm.val_dataloader()
    val_df = analyze_split(model, val_loader, args.device, "Validation", cfg)

    # Save to CSV if requested
    if args.output:
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        combined = pd.concat([train_df, val_df], ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
