"""Ensemble evaluation for ProAgg models."""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_ranking import LightningProAggModelRanking
from src.config.utils import load_yaml_config


def ensemble_predict(models, dataloader, device):
    """Ensemble prediction by averaging."""
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Ensemble predicting"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Collect predictions from all models
        preds = []
        with torch.no_grad():
            for model in models:
                model.eval()
                out = model(batch)
                pred = out["score"].float().flatten()
                preds.append(pred.cpu().numpy())

        # Average predictions
        avg_pred = np.mean(preds, axis=0)

        all_preds.append(avg_pred)
        all_targets.append(batch["score"].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser(description='Ensemble evaluation for ProAgg')
    parser.add_argument('--config', type=str, default='configs/proagg_v13.yaml')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='List of checkpoint paths to ensemble')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Optional weights for each model (default: equal)')
    parser.add_argument('--output', type=str, default='ensemble_results.txt')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Load data
    data_module = ProAggDataModule(cfg)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()

    # Load models
    device = torch.device(f'cuda:{cfg.hardware.devices[0]}' if torch.cuda.is_available() else 'cpu')
    models = []

    print(f"Loading {len(args.checkpoints)} models for ensemble...")
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping...")
            continue
        model = LightningProAggModelRanking.load_from_checkpoint(ckpt_path, cfg=cfg)
        model.eval()
        model.to(device)
        models.append(model)
        print(f"  Loaded: {os.path.basename(ckpt_path)}")

    if len(models) == 0:
        print("No models loaded!")
        return

    # Ensemble prediction
    print(f"\nRunning ensemble inference with {len(models)} models...")
    all_preds = []
    all_targets = []

    for batch in tqdm(test_loader, desc="Predicting"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        preds = []
        with torch.no_grad():
            for model in models:
                out = model(batch)
                pred = out["score"].float().flatten()
                preds.append(pred.cpu().numpy())

        # Weighted or equal average
        if args.weights is not None and len(args.weights) == len(models):
            weights = np.array(args.weights) / sum(args.weights)
            avg_pred = np.average(preds, axis=0, weights=weights)
        else:
            avg_pred = np.mean(preds, axis=0)

        all_preds.append(avg_pred)
        all_targets.append(batch["score"].cpu().numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Calculate metrics
    from scipy.stats import pearsonr, spearmanr

    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    mse = np.mean((predictions - targets) ** 2)

    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)
    print(f"Models: {len(models)}")
    if args.weights:
        print(f"Weights: {args.weights}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test Pearson: {pearson_r:.6f}")
    print(f"Test Spearman: {spearman_r:.6f}")
    print("=" * 60)

    # Save results
    with open(args.output, 'w') as f:
        f.write("Ensemble Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Models used: {len(models)}\n")
        for i, ckpt in enumerate(args.checkpoints):
            f.write(f"  [{i+1}] {ckpt}\n")
        if args.weights:
            f.write(f"Weights: {args.weights}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test Pearson: {pearson_r:.6f}\n")
        f.write(f"Test Spearman: {spearman_r:.6f}\n")
        f.write("=" * 60 + "\n")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
