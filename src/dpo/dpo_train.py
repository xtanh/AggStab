"""DPO fine-tuning of ProteinMPNN using ProAgg preference pairs.

Usage:
    python src/dpo/dpo_train.py \
        --pairs_path data/dpo_pairs.pt \
        --output_dir results/dpo/ \
        --device cuda:3 \
        --epochs 10 \
        --lr 1e-5 \
        --beta 0.1
"""

import os
import sys
import copy
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

from src.mpnn.mpnn_wrapper import (
    load_mpnn_model,
    featurize_pdb,
    MPNN_ALPHABET,
)

MPNN_DIR = "/home/xy_th/ProteinMPNN"
sys.path.insert(0, MPNN_DIR)
from protein_mpnn_utils import _scores


# =====================================================================
# Dataset
# =====================================================================

class DPOPairDataset(Dataset):
    """Dataset of (backbone, seq_winner, seq_loser) triples."""

    def __init__(self, pairs, device="cpu"):
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
            "seq_winner": pair["seq_winner"],
            "seq_loser": pair["seq_loser"],
            "score_winner": pair["score_winner"],
            "score_loser": pair["score_loser"],
        }


def seq_to_indices(seq, L, device):
    """Convert sequence string to index tensor, padded to length L."""
    indices = torch.zeros(L, dtype=torch.long, device=device)
    for j, aa in enumerate(seq):
        if j < L:
            indices[j] = MPNN_ALPHABET.index(aa) if aa in MPNN_ALPHABET else 20
    return indices


# =====================================================================
# DPO Loss
# =====================================================================

def compute_seq_log_prob(model, feat, S, device):
    """Compute mean log-probability of sequence S under the model.

    Returns scalar: mean log p(s_i | context) over designed positions.
    """
    X = feat["X"]
    mask = feat["mask"]
    chain_M = feat["chain_M"] * feat["chain_M_pos"]
    chain_encoding_all = feat["chain_encoding_all"]
    residue_idx = feat["residue_idx"]

    if S.dim() == 1:
        S = S.unsqueeze(0)

    randn = torch.randn(chain_M.shape, device=device)
    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)

    neg_scores = _scores(S, log_probs, mask * chain_M)
    return -neg_scores.squeeze()


def dpo_loss(
    model_theta, model_ref, feat, S_w, S_l, beta, device,
):
    """Compute the DPO loss for a single (backbone, winner, loser) triple.

    loss = -log sigmoid(beta * (
        (log pi_theta(y_w|x) - log pi_ref(y_w|x))
      - (log pi_theta(y_l|x) - log pi_ref(y_l|x))
    ))
    """
    log_pi_theta_w = compute_seq_log_prob(model_theta, feat, S_w, device)
    log_pi_theta_l = compute_seq_log_prob(model_theta, feat, S_l, device)

    with torch.no_grad():
        log_pi_ref_w = compute_seq_log_prob(model_ref, feat, S_w, device)
        log_pi_ref_l = compute_seq_log_prob(model_ref, feat, S_l, device)

    log_ratio_w = log_pi_theta_w - log_pi_ref_w
    log_ratio_l = log_pi_theta_l - log_pi_ref_l

    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))

    with torch.no_grad():
        reward_margin = (log_ratio_w - log_ratio_l).item()
        accuracy = float((log_ratio_w > log_ratio_l).item())

    return loss, {
        "reward_margin": reward_margin,
        "accuracy": accuracy,
        "log_pi_theta_w": log_pi_theta_w.item(),
        "log_pi_theta_l": log_pi_theta_l.item(),
    }


# =====================================================================
# Training loop
# =====================================================================

def train_dpo(args):
    device = args.device

    print("Loading DPO pairs...", flush=True)
    data = torch.load(args.pairs_path, map_location="cpu")
    pairs = data["pairs"]
    print(f"  {len(pairs)} pairs loaded", flush=True)

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
        print(f"  Using first {len(pairs)} pairs", flush=True)

    print("Loading ProteinMPNN (policy)...", flush=True)
    model_theta = load_mpnn_model(
        checkpoint_path=args.mpnn_ckpt, device=device, backbone_noise=0.0,
    )
    model_theta.train()

    print("Loading ProteinMPNN (reference, frozen)...", flush=True)
    model_ref = load_mpnn_model(
        checkpoint_path=args.mpnn_ckpt, device=device, backbone_noise=0.0,
    )
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model_theta.parameters(), lr=args.lr)

    dataset = DPOPairDataset(pairs, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    history = []

    for epoch in range(args.epochs):
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_steps = 0

        for step, idx in enumerate(indices):
            item = dataset[idx]

            feat = dataset._get_feat(item["pdb_path"])
            L = feat["X"].shape[1]

            S_w = seq_to_indices(item["seq_winner"], L, device).unsqueeze(0)
            S_l = seq_to_indices(item["seq_loser"], L, device).unsqueeze(0)

            loss, metrics = dpo_loss(
                model_theta, model_ref, feat, S_w, S_l,
                beta=args.beta, device=device,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model_theta.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += metrics["accuracy"]
            epoch_margin += metrics["reward_margin"]
            n_steps += 1

            if (step + 1) % args.log_every == 0:
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(indices)} | "
                    f"loss={loss.item():.4f} acc={metrics['accuracy']:.0f} "
                    f"margin={metrics['reward_margin']:.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_acc = epoch_acc / max(n_steps, 1)
        avg_margin = epoch_margin / max(n_steps, 1)

        record = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": avg_acc,
            "reward_margin": avg_margin,
        }
        history.append(record)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={avg_loss:.4f} acc={avg_acc:.3f} margin={avg_margin:.4f}",
            flush=True,
        )

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"mpnn_dpo_epoch{epoch+1}.pt")
            torch.save({
                "model_state_dict": model_theta.state_dict(),
                "num_edges": 48,
                "noise_level": 0.0,
                "epoch": epoch + 1,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")

    return model_theta


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning of ProteinMPNN")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--mpnn_ckpt", type=str, default=None,
                        help="ProteinMPNN checkpoint (default: v_48_020)")
    parser.add_argument("--output_dir", type=str, default="results/dpo/")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature parameter")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    train_dpo(args)


if __name__ == "__main__":
    main()
