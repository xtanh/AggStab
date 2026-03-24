import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.append(PROJ_DIR)

from src.config.utils import load_yaml_config
from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model import LightningProAggModel
from src.ln.lightning_model_cl import LightningProAggModelCL
from src.ln.lightning_model_ranking import LightningProAggModelRanking


MODEL_TYPES = {
    "base": LightningProAggModel,
    "cl": LightningProAggModelCL,
    "ranking": LightningProAggModelRanking,
}


def _infer_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _load_lightning_model(model_type: str, checkpoint_path: str, cfg, device: torch.device):
    model_cls = MODEL_TYPES[model_type]
    model = model_cls.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
    model.eval()
    model.to(device)
    return model


def _build_detailed_frame(raw_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out_df = raw_df.copy()
    target = out_df["log2_fold_change_75_clip"].to_numpy(dtype=np.float32)
    signed_error = preds - target
    out_df["prediction"] = preds
    out_df["target"] = target
    out_df["abs_error"] = np.abs(signed_error)
    out_df["signed_error"] = signed_error
    out_df["seq_len"] = out_df["protein_sequence"].str.len()
    return out_df


def evaluate_split(model, dataset, collate_fn, batch_size: int, device: torch.device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = model(batch)
            pred = out["score"].float().flatten().detach().cpu().numpy()
            all_preds.append(pred)

    return np.concatenate(all_preds, axis=0)


def evaluate(cfg, checkpoint_path: str, model_type: str, splits, output_dir: str, device_str: str):
    torch.manual_seed(42)
    np.random.seed(42)

    device = _infer_device(device_str)
    model = _load_lightning_model(model_type, checkpoint_path, cfg, device)
    data_module = ProAggDataModule(cfg)

    split_to_dataset = {
        "train": data_module.train_dataset,
        "valid": data_module.valid_dataset,
        "test": data_module.test_dataset,
    }

    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for split_name in splits:
        dataset = split_to_dataset[split_name]
        preds = evaluate_split(model, dataset, data_module.collate_fn, cfg.train.batch_size, device)
        detailed_df = _build_detailed_frame(dataset.df, preds)

        csv_path = os.path.join(output_dir, f"{split_name}_predictions_detailed.csv")
        detailed_df.to_csv(csv_path, index=False)

        target = detailed_df["target"].to_numpy()
        prediction = detailed_df["prediction"].to_numpy()
        summary_rows.append({
            "split": split_name,
            "n": len(detailed_df),
            "spearman": float(spearmanr(prediction, target).statistic),
            "pearson": float(pearsonr(prediction, target).statistic),
            "mae": float(np.mean(np.abs(prediction - target))),
            "mse": float(np.mean((prediction - target) ** 2)),
            "path": csv_path,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "detailed_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Export detailed split predictions for a checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", choices=sorted(MODEL_TYPES), required=True)
    parser.add_argument("--splits", nargs="+", default=["valid"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    evaluate(cfg, args.checkpoint, args.model_type, args.splits, output_dir, args.device)


if __name__ == "__main__":
    main()
