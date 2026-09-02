"""DPO fine-tuning of ProteinMPNN with true mini-batch support.

This version groups pairs by PDB to enable true mini-batch training,
utilizing GPU memory more efficiently.
"""

import os
import sys
import copy
import argparse
import json
from datetime import datetime
from collections import defaultdict

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
    protein_mpnn_scores,
)

from src.utils.seed import set_global_seed


class DPOPairDataset(Dataset):
    """Dataset of (backbone, seq_winner, seq_loser) triples."""

    def __init__(self, pairs, device="cpu"):
        self.pairs = pairs
        self.device = device
        self._feat_cache = {}

        # Group pairs by pdb_path for batching
        self.pdb_to_indices = defaultdict(list)
        for i, p in enumerate(pairs):
            self.pdb_to_indices[p["pdb_path"]].append(i)
        self.unique_pdbs = list(self.pdb_to_indices.keys())

    def __len__(self):
        return len(self.pairs)

    def _get_feat(self, pdb_path):
        if pdb_path not in self._feat_cache:
            self._feat_cache[pdb_path] = featurize_pdb(pdb_path, device=self.device)
        return self._feat_cache[pdb_path]

    def get_pairs_for_pdb(self, pdb_path):
        """Get all pairs for a specific PDB (for batching)."""
        indices = self.pdb_to_indices[pdb_path]
        return [self.pairs[i] for i in indices]


def seq_to_indices(seq, L, device):
    """Convert sequence string to index tensor, padded to length L."""
    indices = torch.zeros(L, dtype=torch.long, device=device)
    for j, aa in enumerate(seq):
        if j < L:
            indices[j] = MPNN_ALPHABET.index(aa) if aa in MPNN_ALPHABET else 20
    return indices


def compute_seq_log_prob(model, feat, S_batch, device, randn=None):
    """Compute mean log-probability of sequences S under the model (batched).

    Args:
        S_batch: tensor of shape (B, L) with sequence indices
        randn: Fixed decoding-order tensor of shape (B, L)
    """
    X = feat["X"]
    mask = feat["mask"]
    chain_M = feat["chain_M"] * feat["chain_M_pos"]
    chain_encoding_all = feat["chain_encoding_all"]
    residue_idx = feat["residue_idx"]

    B = S_batch.shape[0]
    X_rep = X.expand(B, -1, -1, -1)
    mask_rep = mask.expand(B, -1)
    chain_M_rep = chain_M.expand(B, -1)
    chain_enc_rep = chain_encoding_all.expand(B, -1)
    res_idx_rep = residue_idx.expand(B, -1)

    if randn is None:
        randn = torch.zeros((B, chain_M.shape[1]), device=device)

    log_probs = model(X_rep, S_batch, mask_rep, chain_M_rep, res_idx_rep, chain_enc_rep, randn)
    neg_scores = protein_mpnn_scores(S_batch, log_probs, mask_rep * chain_M_rep)
    return -neg_scores.squeeze()


def dpo_loss_batched(
    model_theta, model_ref, feat, S_w_batch, S_l_batch, beta, device
):
    """Compute DPO loss for a batch of sequences from the same backbone.

    Args:
        S_w_batch: (B, L) winner sequences
        S_l_batch: (B, L) loser sequences
    """
    B = S_w_batch.shape[0]
    chain_M = feat["chain_M"] * feat["chain_M_pos"]
    randn = torch.zeros((B, chain_M.shape[1]), device=device)

    log_pi_theta_w = compute_seq_log_prob(model_theta, feat, S_w_batch, device, randn)
    log_pi_theta_l = compute_seq_log_prob(model_theta, feat, S_l_batch, device, randn)

    with torch.no_grad():
        log_pi_ref_w = compute_seq_log_prob(model_ref, feat, S_w_batch, device, randn)
        log_pi_ref_l = compute_seq_log_prob(model_ref, feat, S_l_batch, device, randn)

    log_ratio_w = log_pi_theta_w - log_pi_ref_w
    log_ratio_l = log_pi_theta_l - log_pi_ref_l

    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

    with torch.no_grad():
        reward_margin = (log_ratio_w - log_ratio_l).mean().item()
        accuracy = float((log_ratio_w > log_ratio_l).float().mean().item())

    return loss, {
        "reward_margin": reward_margin,
        "accuracy": accuracy,
        "log_pi_theta_w": log_pi_theta_w.mean().item(),
        "log_pi_theta_l": log_pi_theta_l.mean().item(),
    }


@torch.no_grad()
def evaluate_dpo_batched(model_theta, model_ref, dataset, beta, device):
    """Evaluate DPO loss and accuracy on a dataset (batched by PDB)."""
    model_theta.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    n = 0

    for pdb_path in dataset.unique_pdbs:
        pairs = dataset.get_pairs_for_pdb(pdb_path)
        if len(pairs) == 0:
            continue

        feat = dataset._get_feat(pdb_path)
        L = feat["X"].shape[1]

        # Stack all sequences for this PDB
        S_w_list = [seq_to_indices(p["seq_winner"], L, device) for p in pairs]
        S_l_list = [seq_to_indices(p["seq_loser"], L, device) for p in pairs]
        S_w_batch = torch.stack(S_w_list, dim=0)
        S_l_batch = torch.stack(S_l_list, dim=0)

        loss, metrics = dpo_loss_batched(
            model_theta, model_ref, feat, S_w_batch, S_l_batch,
            beta=beta, device=device,
        )

        total_loss += loss.item() * len(pairs)
        total_acc += metrics["accuracy"] * len(pairs)
        total_margin += metrics["reward_margin"] * len(pairs)
        n += len(pairs)

    model_theta.train()
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
        "reward_margin": total_margin / max(n, 1),
    }


def train_dpo_batched(args):
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

    val_pairs = None
    if args.val_pairs_path:
        print("Loading DPO validation pairs...", flush=True)
        val_data = torch.load(args.val_pairs_path, map_location="cpu")
        val_pairs = val_data["pairs"]
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
    val_dataset = DPOPairDataset(val_pairs, device=device) if val_pairs else None

    print(f"  Unique PDBs in training: {len(dataset.unique_pdbs)}")
    if val_dataset:
        print(f"  Unique PDBs in validation: {len(val_dataset.unique_pdbs)}")

    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model_theta.train()
        np.random.shuffle(dataset.unique_pdbs)

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_batches = 0
        n_pairs = 0

        optimizer.zero_grad()

        # Process all pairs for each PDB in one batch
        for pdb_idx, pdb_path in enumerate(dataset.unique_pdbs):
            pdb_pairs = dataset.get_pairs_for_pdb(pdb_path)
            if len(pdb_pairs) == 0:
                continue

            feat = dataset._get_feat(pdb_path)
            L = feat["X"].shape[1]

            # Stack all winner/loser sequences for this PDB
            S_w_list = [seq_to_indices(p["seq_winner"], L, device) for p in pdb_pairs]
            S_l_list = [seq_to_indices(p["seq_loser"], L, device) for p in pdb_pairs]
            S_w_batch = torch.stack(S_w_list, dim=0)
            S_l_batch = torch.stack(S_l_list, dim=0)

            loss, metrics = dpo_loss_batched(
                model_theta, model_ref, feat, S_w_batch, S_l_batch,
                beta=args.beta, device=device,
            )

            # Scale loss for gradient accumulation
            loss_scaled = loss / args.accum_steps
            loss_scaled.backward()

            epoch_loss += loss.item() * len(pdb_pairs)
            epoch_acc += metrics["accuracy"] * len(pdb_pairs)
            epoch_margin += metrics["reward_margin"] * len(pdb_pairs)
            n_batches += 1
            n_pairs += len(pdb_pairs)

            # Gradient update every accum_steps PDBs
            if (pdb_idx + 1) % args.accum_steps == 0 or (pdb_idx + 1) == len(dataset.unique_pdbs):
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model_theta.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if (pdb_idx + 1) % args.log_every == 0:
                print(
                    f"  Epoch {epoch+1} PDB {pdb_idx+1}/{len(dataset.unique_pdbs)} | "
                    f"pairs={n_pairs} loss={loss.item():.4f} acc={metrics['accuracy']:.2f} "
                    f"margin={metrics['reward_margin']:.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(n_pairs, 1)
        avg_acc = epoch_acc / max(n_pairs, 1)
        avg_margin = epoch_margin / max(n_pairs, 1)

        record = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
            "train_reward_margin": avg_margin,
        }

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_loss:.4f} train_acc={avg_acc:.3f} "
            f"train_margin={avg_margin:.4f} (processed {n_pairs} pairs)",
            flush=True,
        )

        # --- Validation ---
        if val_dataset is not None:
            val_metrics = evaluate_dpo_batched(
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


def _save_checkpoint(model, epoch, args, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_edges": 48,
        "noise_level": 0.0,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning of ProteinMPNN (batched)")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, default=None)
    parser.add_argument("--mpnn_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/dpo/")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--score_gap_delta", type=float, default=0.0)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dpo_batched(args)


if __name__ == "__main__":
    main()
