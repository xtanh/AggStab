"""Winner-only SFT fine-tuning of ProteinMPNN from preference pairs.

This provides a minimal SFT baseline against joint/semi-online DPO:
- use the same preference-pair construction
- keep only winner sequences
- optimize standard teacher-forcing log-likelihood
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

from src.dpo.dpo_train import seq_to_indices
from src.mpnn.mpnn_wrapper import (
    MPNN_ALPHABET,
    featurize_pdb,
    load_mpnn_model,
    protein_mpnn_scores,
)

from src.utils.seed import set_global_seed


class SFTWinnerDataset(Dataset):
    def __init__(self, pairs, device: str = "cpu"):
        self.pairs = pairs
        self.device = device
        self._feat_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _get_feat(self, pdb_path):
        if pdb_path not in self._feat_cache:
            self._feat_cache[pdb_path] = featurize_pdb(pdb_path, device=self.device)
        return self._feat_cache[pdb_path]

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "pdb_path": pair["pdb_path"],
            "seq": pair["seq_winner"],
            "objective": pair.get("objective", "joint"),
        }


def compute_seq_nll(model, feat, S, device, randn=None):
    X = feat["X"]
    mask = feat["mask"]
    chain_M = feat["chain_M"] * feat["chain_M_pos"]
    chain_encoding_all = feat["chain_encoding_all"]
    residue_idx = feat["residue_idx"]

    if S.dim() == 1:
        S = S.unsqueeze(0)
    if randn is None:
        randn = torch.zeros(chain_M.shape, device=device)

    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    neg_scores = protein_mpnn_scores(S, log_probs, mask * chain_M)
    return neg_scores.squeeze()


@torch.no_grad()
def evaluate_sft(model, dataset, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        feat = dataset._get_feat(item["pdb_path"])
        L = feat["X"].shape[1]
        S = seq_to_indices(item["seq"], L, device).unsqueeze(0)
        randn = torch.zeros((1, L), device=device)
        loss = compute_seq_nll(model, feat, S, device=device, randn=randn)
        total_loss += loss.item()
        n += 1
    model.train()
    return {"loss": total_loss / max(n, 1)}


def _save_checkpoint(model, epoch, args, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_edges": 48,
        "noise_level": 0.0,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def train_sft(args):
    device = args.device
    set_global_seed(args.seed)

    print(f"Loading SFT train pairs from {args.pairs_path}...", flush=True)
    train_data = torch.load(args.pairs_path, map_location="cpu")
    train_pairs = train_data["pairs"]
    print(f"  {len(train_pairs)} train pairs loaded", flush=True)

    val_dataset = None
    if args.val_pairs_path:
        print(f"Loading SFT val pairs from {args.val_pairs_path}...", flush=True)
        val_data = torch.load(args.val_pairs_path, map_location="cpu")
        val_pairs = val_data["pairs"]
        val_dataset = SFTWinnerDataset(val_pairs, device=device)
        print(f"  {len(val_pairs)} val pairs loaded", flush=True)

    dataset = SFTWinnerDataset(train_pairs, device=device)

    print("Loading ProteinMPNN (policy)...", flush=True)
    model = load_mpnn_model(checkpoint_path=args.mpnn_ckpt, device=device, backbone_noise=0.0)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "mpnn_sft_best.pt")
    patience_counter = 0

    accum_steps = args.batch_size

    for epoch in range(args.epochs):
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps = 0

        for step, idx in enumerate(indices):
            item = dataset[idx]
            feat = dataset._get_feat(item["pdb_path"])
            L = feat["X"].shape[1]
            S = seq_to_indices(item["seq"], L, device).unsqueeze(0)
            randn = torch.zeros((1, L), device=device)
            loss = compute_seq_nll(model, feat, S, device=device, randn=randn)
            (loss / accum_steps).backward()

            epoch_loss += loss.item()
            n_steps += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_steps, 1)
        record = {"epoch": epoch + 1, "train_loss": avg_loss}
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f}", flush=True)

        if val_dataset is not None:
            val_metrics = evaluate_sft(model, val_dataset, device=device)
            record["val_loss"] = val_metrics["loss"]
            print(f"           | val_loss={val_metrics['loss']:.4f}", flush=True)

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                _save_checkpoint(model, epoch + 1, args, best_path)
                print(f"  -> New best model saved (val_loss={best_val_loss:.4f})", flush=True)
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.patience})", flush=True)

            if args.patience > 0 and patience_counter >= args.patience:
                history.append(record)
                break

        history.append(record)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"mpnn_sft_epoch{epoch+1}.pt")
            _save_checkpoint(model, epoch + 1, args, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. History saved to {history_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Winner-only SFT baseline for ProteinMPNN")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, default=None)
    parser.add_argument("--mpnn_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/sft/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sft(args)


if __name__ == "__main__":
    main()
