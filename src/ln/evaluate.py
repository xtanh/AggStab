import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from scipy.stats import spearmanr, pearsonr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model import LightningProAggModel
from src.config.utils import load_yaml_config


def evaluate(cfg, checkpoint_path, output_dir=None):
    pl.seed_everything(42)

    model = LightningProAggModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    model.cuda()

    data_module = ProAggDataModule(cfg)

    splits = {"test": data_module.test_dataset, "valid": data_module.valid_dataset}
    collate_fn = data_module.collate_fn

    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    for split_name, dataset in splits.items():
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.train.batch_size,
            collate_fn=collate_fn, num_workers=0, shuffle=False,
        )

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(batch)
                pred = out["score"].float().flatten().cpu().numpy()
                tgt = batch["score"].float().flatten().cpu().numpy()
                all_preds.append(pred)
                all_targets.append(tgt)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        scc, _ = spearmanr(all_preds, all_targets)
        pcc, _ = pearsonr(all_preds, all_targets)
        mse = np.mean((all_preds - all_targets) ** 2)

        print(f"\n{'='*60}")
        print(f"  {split_name.upper()} set ({len(all_preds)} samples)")
        print(f"{'='*60}")
        print(f"  Spearman:  {scc:.4f}")
        print(f"  Pearson:   {pcc:.4f}")
        print(f"  MSE:       {mse:.4f}")
        print(f"{'='*60}")

        df_out = pd.DataFrame({
            "target": all_targets,
            "prediction": all_preds,
        })
        csv_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"  Predictions saved to: {csv_path}")

    summary_path = os.path.join(output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        for split_name, dataset in splits.items():
            csv_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
            df = pd.read_csv(csv_path)
            scc, _ = spearmanr(df["prediction"], df["target"])
            pcc, _ = pearsonr(df["prediction"], df["target"])
            mse = np.mean((df["prediction"].values - df["target"].values) ** 2)
            f.write(f"[{split_name.upper()}] Spearman={scc:.4f}  Pearson={pcc:.4f}  MSE={mse:.4f}  N={len(df)}\n")

    print(f"\n  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProAgg checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save predictions")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    evaluate(cfg, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
