"""Analyze error cases for version_60."""
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.models.factory import build_proagg_model
from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_ranking import LightningProAggModelRanking
import pytorch_lightning as pl
from omegaconf import OmegaConf


def compute_spearman(pred, tgt):
    from scipy.stats import spearmanr
    return spearmanr(pred.cpu().numpy(), tgt.cpu().numpy())[0]


def analyze_split(model, dataloader, device, split_name):
    model.eval()
    all_results = []

    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            out = model(batch)
            pred = out['score'].flatten()
            tgt = batch['score'].flatten()

            for i in range(len(pred)):
                result = {
                    'predicted': pred[i].item(),
                    'target': tgt[i].item(),
                    'absolute_error': abs(pred[i].item() - tgt[i].item()),
                    'squared_error': (pred[i].item() - tgt[i].item()) ** 2,
                }
                all_results.append(result)

    df = pd.DataFrame(all_results)
    df['error'] = df['predicted'] - df['target']

    print(f"\n{'='*70}")
    print(f"Error Analysis: {split_name} Set")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"MSE: {df['squared_error'].mean():.4f}")
    print(f"MAE: {df['absolute_error'].mean():.4f}")
    print(f"Spearman: {compute_spearman(torch.tensor(df['predicted'].values), torch.tensor(df['target'].values)):.4f}")

    # Error by Target Range
    print(f"\n--- Error Distribution by Target Range ---")
    bins = [-float('inf'), -3, -2, -1, 0, float('inf')]
    labels = ['<-3', '-3~-2', '-2~-1', '-1~0', '>0']
    df['target_bin'] = pd.cut(df['target'], bins=bins, labels=labels)

    for label in labels:
        subset = df[df['target_bin'] == label]
        if len(subset) > 0:
            spearman = compute_spearman(torch.tensor(subset['predicted'].values), torch.tensor(subset['target'].values))
            print(f"  {label:8s}: n={len(subset):4d}, MAE={subset['absolute_error'].mean():.3f}, "
                  f"MeanErr={subset['error'].mean():+.3f}, Spearman={spearman:.3f}")

    # Largest Absolute Errors
    print(f"\n--- Top 10 Largest Absolute Errors ---")
    worst = df.nlargest(10, 'absolute_error')
    for idx, row in worst.iterrows():
        print(f"  True: {row['target']:+.2f}, Pred: {row['predicted']:+.2f}, Error: {row['absolute_error']:.2f} ({row['error']:+.2f})")

    # Ranking analysis
    print(f"\n--- Ranking Analysis ---")
    pred_tensor = torch.tensor(df['predicted'].values)
    tgt_tensor = torch.tensor(df['target'].values)

    tgt_diff = tgt_tensor.unsqueeze(0) - tgt_tensor.unsqueeze(1)
    pred_diff = pred_tensor.unsqueeze(0) - pred_tensor.unsqueeze(1)

    valid_pairs = torch.abs(tgt_diff) > 0.3
    n_valid = valid_pairs.sum().item() - len(df)

    inversions = (tgt_diff * pred_diff < 0) & valid_pairs
    n_inversions = inversions.sum().item()

    print(f"  Valid pairs (|diff|>0.3): {n_valid}")
    print(f"  Ranking inversions: {n_inversions} ({100*n_inversions/max(n_valid,1):.1f}%)")

    return df


def main():
    ckpt_path = "results/lightning_logs/version_60/checkpoints/best_epoch=07_val_spearman=0.7635.ckpt"
    device = "cuda:0"

    # Load checkpoint to get config
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = checkpoint['hyper_parameters']['cfg']
    cfg = OmegaConf.create(cfg_dict)

    print(f"Loading checkpoint: {ckpt_path}")
    pl_module = LightningProAggModelRanking.load_from_checkpoint(ckpt_path, cfg=cfg)
    model = pl_module.model
    model = model.to(device)
    model.eval()

    # Data module
    dm = ProAggDataModule(cfg)
    dm.setup(stage='fit')

    # Analyze train set
    train_loader = dm.train_dataloader()
    train_df = analyze_split(model, train_loader, device, "Train")

    # Analyze val set
    val_loader = dm.val_dataloader()
    val_df = analyze_split(model, val_loader, device, "Validation")

    # Save results
    output_path = "results/lightning_logs/version_60/error_analysis.csv"
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
