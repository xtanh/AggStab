"""DPO fine-tuning of ProteinMPNN using ProAgg preference pairs.

Usage:
    python src/dpo/dpo_train.py \
        --pairs_path data/dpo/train_pairs.pt \
        --val_pairs_path data/dpo/val_pairs.pt \
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

from src.utils.seed import set_global_seed


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

def compute_seq_log_prob(model, feat, S, device, randn=None):
    """Compute mean log-probability of sequence S under the model.

    Returns scalar: mean log p(s_i | context) over designed positions.

    Args:
        randn: Fixed decoding-order tensor. If None, uses zeros (fixed order).
               Pass the same randn to policy and reference to eliminate
               decoding-order noise from the DPO log-ratio.
    """
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

    A single fixed randn (zeros) is shared across all four log-prob calls so
    that the decoding order is identical for policy and reference.  This
    eliminates random decoding-order noise from the log-ratio, making the
    DPO gradient deterministic and meaningful.
    """
    chain_M = feat["chain_M"] * feat["chain_M_pos"]
    randn = torch.zeros(chain_M.shape, device=device)

    log_pi_theta_w = compute_seq_log_prob(model_theta, feat, S_w, device, randn)
    log_pi_theta_l = compute_seq_log_prob(model_theta, feat, S_l, device, randn)

    with torch.no_grad():
        log_pi_ref_w = compute_seq_log_prob(model_ref, feat, S_w, device, randn)
        log_pi_ref_l = compute_seq_log_prob(model_ref, feat, S_l, device, randn)

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

@torch.no_grad()
def evaluate_dpo(model_theta, model_ref, dataset, beta, device):
    """Evaluate DPO loss and accuracy on a dataset."""
    model_theta.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    n = 0

    for idx in range(len(dataset)):
        item = dataset[idx]
        feat = dataset._get_feat(item["pdb_path"])
        L = feat["X"].shape[1]

        S_w = seq_to_indices(item["seq_winner"], L, device).unsqueeze(0)
        S_l = seq_to_indices(item["seq_loser"], L, device).unsqueeze(0)

        loss, metrics = dpo_loss(
            model_theta, model_ref, feat, S_w, S_l,
            beta=beta, device=device,
        )

        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        total_margin += metrics["reward_margin"]
        n += 1

    model_theta.train()
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "reward_margin": total_margin / max(n, 1),
    }


def _save_checkpoint(model, epoch, args, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_edges": 48,
        "noise_level": 0.0,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def train_dpo(args):
    device = args.device
    set_global_seed(args.seed)

    print("Loading DPO training pairs...", flush=True)
    data = torch.load(args.pairs_path, map_location="cpu")
    pairs = data["pairs"]
    print(f"  {len(pairs)} training pairs loaded", flush=True)

    if args.score_gap_delta > 0.0:
        n_before = len(pairs)
        pairs = [p for p in pairs if p.get("score_gap", float("inf")) > args.score_gap_delta]
        print(f"  score_gap_delta={args.score_gap_delta}: {n_before} -> {len(pairs)} pairs", flush=True)

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
        print(f"  Using first {len(pairs)} pairs", flush=True)

    val_dataset = None
    if args.val_pairs_path:
        print("Loading DPO validation pairs...", flush=True)
        val_data = torch.load(args.val_pairs_path, map_location="cpu")
        val_pairs = val_data["pairs"]
        val_dataset = DPOPairDataset(val_pairs, device=device)
        print(f"  {len(val_pairs)} validation pairs loaded", flush=True)

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
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    accum_steps = args.batch_size

    for epoch in range(args.epochs):
        model_theta.train()
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_steps = 0

        optimizer.zero_grad()

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

            (loss / accum_steps).backward()

            epoch_loss += loss.item()
            epoch_acc += metrics["accuracy"]
            epoch_margin += metrics["reward_margin"]
            n_steps += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model_theta.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

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
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "train_reward_margin": avg_margin,
        }

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_loss:.4f} train_acc={avg_acc:.3f} "
            f"train_margin={avg_margin:.4f}",
            flush=True,
        )

        # --- Validation ---
        if val_dataset is not None:
            val_metrics = evaluate_dpo(
                model_theta, model_ref, val_dataset,
                beta=args.beta, device=device,
            )
            record["val_loss"] = val_metrics["loss"]
            record["val_accuracy"] = val_metrics["accuracy"]
            record["val_reward_margin"] = val_metrics["reward_margin"]

            print(
                f"           | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_margin={val_metrics['reward_margin']:.4f}",
                flush=True,
            )

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
                best_path = os.path.join(args.output_dir, "mpnn_dpo_best.pt")
                _save_checkpoint(model_theta, epoch + 1, args, best_path)
                print(f"  -> New best model saved (val_loss={best_val_loss:.4f}, "
                      f"val_acc={best_val_acc:.3f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{args.patience})")

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f})")
                history.append(record)
                break

        history.append(record)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"mpnn_dpo_epoch{epoch+1}.pt")
            _save_checkpoint(model_theta, epoch + 1, args, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if val_dataset is not None:
        print(f"\nBest model: val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f}")
        print(f"Best checkpoint: {os.path.join(args.output_dir, 'mpnn_dpo_best.pt')}")

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")

    return model_theta


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning of ProteinMPNN")
    parser.add_argument("--pairs_path", type=str, required=True,
                        help="Path to training preference pairs (.pt)")
    parser.add_argument("--val_pairs_path", type=str, default=None,
                        help="Path to validation preference pairs (.pt). "
                             "If provided, enables best-checkpoint saving and early stopping.")
    parser.add_argument("--mpnn_ckpt", type=str, default=None,
                        help="ProteinMPNN checkpoint (default: v_48_020)")
    parser.add_argument("--output_dir", type=str, default="results/dpo/")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="DPO temperature parameter (KL penalty). "
                             "Higher = stay closer to reference, less reward hacking.")
    parser.add_argument("--score_gap_delta", type=float, default=0.0,
                        help="Filter out pairs whose score_gap <= delta. "
                             "Removes noisy pairs without re-sampling (applied at load time).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Effective batch size via gradient accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training order / torch RNG")
    args = parser.parse_args()

    train_dpo(args)


if __name__ == "__main__":
    main()
