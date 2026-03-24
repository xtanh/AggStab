import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[: FILE_DIR.index("scripts")]
sys.path.append(PROJ_DIR)

from src.config.utils import load_yaml_config
from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_ranking import LightningProAggModelRanking


def infer_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def load_model(checkpoint_path: str, cfg, device: torch.device):
    model = LightningProAggModelRanking.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
    model.eval()
    model.to(device)
    return model


def export_split(model, dataset, collate_fn, batch_size: int, device: torch.device, split_name: str, output_dir: str):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    rows = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch)
            bsz = len(batch["score"])
            raw = dataset.df.iloc[offset : offset + bsz].copy()
            offset += bsz

            pred = out["score"].float().flatten().detach().cpu().numpy()
            gate = out.get("gate_weights", None)
            experts = out.get("expert_scores", None)

            raw["prediction"] = pred
            raw["target"] = raw["log2_fold_change_75_clip"].to_numpy(dtype=np.float32)
            raw["abs_error"] = np.abs(raw["prediction"] - raw["target"])
            raw["signed_error"] = raw["prediction"] - raw["target"]

            if gate is not None:
                gate_np = gate.detach().cpu().numpy()
                for i in range(gate_np.shape[1]):
                    raw[f"gate_w_{i}"] = gate_np[:, i]
                raw["gate_argmax"] = gate_np.argmax(axis=1)

            if experts is not None:
                experts_np = experts.detach().cpu().numpy()
                for i in range(experts_np.shape[1]):
                    raw[f"expert_score_{i}"] = experts_np[:, i]

            rows.append(raw)

    df = pd.concat(rows, ignore_index=True)
    out_csv = os.path.join(output_dir, f"{split_name}_moe_behavior.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "split": split_name,
        "n": len(df),
        "spearman": float(spearmanr(df["prediction"], df["target"]).statistic),
        "pearson": float(pearsonr(df["prediction"], df["target"]).statistic),
        "mae": float(df["abs_error"].mean()),
        "mse": float(((df["prediction"] - df["target"]) ** 2).mean()),
        "path": out_csv,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export soft-MoE gate and expert behavior for a checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    device = infer_device(args.device)
    model = load_model(args.checkpoint, cfg, device)
    dm = ProAggDataModule(cfg)
    split_to_dataset = {
        "train": dm.train_dataset,
        "valid": dm.valid_dataset,
        "test": dm.test_dataset,
    }

    summaries = []
    for split in args.splits:
        summaries.append(
            export_split(model, split_to_dataset[split], dm.collate_fn, cfg.train.batch_size, device, split, output_dir)
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(output_dir, "moe_behavior_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
