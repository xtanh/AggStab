"""Stage 2 & 3a: Sample sequences with ProteinMPNN, score with ProAgg, build DPO pairs.

Usage:
    python src/dpo/sample_and_score.py \
        --pdb_dir  inputs/pdbs \
        --proagg_ckpt  results/lightning_logs/version_X/checkpoints/best.ckpt \
        --output  data/dpo_pairs.pt \
        --num_samples 64 \
        --temperature 0.1 \
        --device cuda:3
"""

import os
import sys
import glob
import json
import argparse
import itertools

import torch
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

from src.mpnn.mpnn_wrapper import (
    load_mpnn_model,
    featurize_pdb,
    sample_sequences,
)
from src.ln.lightning_model import LightningProAggModel
from src.ln.lightning_data import alphabet as esm_alphabet


def tokenize_for_proagg(seq, device="cpu"):
    """Tokenize a protein sequence string for ESM-2 / ProAgg."""
    if len(seq) > 1022:
        seq = seq[:1022]
    tokens = torch.empty(len(seq) + 2, dtype=torch.int64)
    tokens[0] = esm_alphabet.cls_idx
    tokens[1:len(seq) + 1] = torch.tensor(
        esm_alphabet.encode(text=seq), dtype=torch.int64
    )
    tokens[len(seq) + 1] = esm_alphabet.eos_idx
    return tokens.unsqueeze(0).to(device)


@torch.no_grad()
def score_sequences_with_proagg(proagg_model, sequences, device="cpu", batch_size=32):
    """Score a list of sequences with ProAgg, return tensor of scores."""
    proagg_model.eval()
    all_scores = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        token_list = [tokenize_for_proagg(s, device="cpu") for s in batch_seqs]

        max_len = max(t.shape[1] for t in token_list)
        padded = torch.full(
            (len(token_list), max_len), esm_alphabet.padding_idx, dtype=torch.int64
        )
        for j, t in enumerate(token_list):
            padded[j, : t.shape[1]] = t[0]

        batch_dict = {"seq_tokens": padded.to(device)}
        out = proagg_model(batch_dict)
        all_scores.append(out["score"].cpu().flatten())

    return torch.cat(all_scores, dim=0)


def build_preference_pairs(sequences, scores, top_ratio=0.25):
    """Build (winner, loser) pairs from sequences + scores.

    Higher score = more soluble / less aggregation-prone = winner.
    """
    n = len(sequences)
    n_top = max(1, int(n * top_ratio))
    n_bot = n_top

    sorted_idx = torch.argsort(scores, descending=True)
    top_idx = sorted_idx[:n_top].tolist()
    bot_idx = sorted_idx[-n_bot:].tolist()

    pairs = []
    for w_i, l_i in itertools.product(top_idx, bot_idx):
        if scores[w_i] > scores[l_i]:
            pairs.append({
                "seq_winner": sequences[w_i],
                "seq_loser": sequences[l_i],
                "score_winner": scores[w_i].item(),
                "score_loser": scores[l_i].item(),
            })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--proagg_ckpt", type=str, required=True,
                        help="Path to trained ProAgg checkpoint")
    parser.add_argument("--proagg_config", type=str, default="configs/default.yaml",
                        help="Path to ProAgg config")
    parser.add_argument("--mpnn_ckpt", type=str, default=None,
                        help="Path to ProteinMPNN checkpoint (default: v_48_020)")
    parser.add_argument("--output", type=str, default="data/dpo_pairs.pt")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_ratio", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--proagg_batch_size", type=int, default=16)
    parser.add_argument("--max_pdbs", type=int, default=-1,
                        help="Max number of PDBs to process (-1 for all)")
    parser.add_argument("--max_pairs_per_pdb", type=int, default=256,
                        help="Cap preference pairs per backbone to avoid imbalance")
    args = parser.parse_args()

    device = args.device

    # --- Load models ---
    print("Loading ProteinMPNN...")
    mpnn_model = load_mpnn_model(
        checkpoint_path=args.mpnn_ckpt, device=device
    )

    print("Loading ProAgg...")
    from src.config.utils import load_yaml_config
    cfg = load_yaml_config(args.proagg_config)
    proagg_lightning = LightningProAggModel.load_from_checkpoint(
        args.proagg_ckpt, cfg=cfg
    )
    proagg_model = proagg_lightning.model.to(device)
    proagg_model.eval()

    # --- Process each PDB ---
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_files:
        print(f"No PDB files found in {args.pdb_dir}")
        return

    if args.max_pdbs > 0:
        np.random.seed(42)
        idx = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[i] for i in sorted(idx)]
        print(f"Selected {len(pdb_files)} PDBs (max_pdbs={args.max_pdbs})")

    all_pairs = []
    all_baseline_results = []

    for pdb_path in pdb_files:
        pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
        print(f"\n{'='*60}")
        print(f"Processing: {pdb_name}")

        feat = featurize_pdb(pdb_path, device=device)

        print(f"  Sampling {args.num_samples} sequences (T={args.temperature})...")
        sequences = sample_sequences(
            mpnn_model, feat,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=device,
        )
        unique_seqs = list(set(sequences))
        print(f"  Got {len(unique_seqs)} unique sequences")

        print(f"  Scoring with ProAgg...")
        scores = score_sequences_with_proagg(
            proagg_model, unique_seqs,
            device=device, batch_size=args.proagg_batch_size,
        )

        best_idx = torch.argmax(scores).item()
        worst_idx = torch.argmin(scores).item()
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}], "
              f"mean={scores.mean():.3f}")

        all_baseline_results.append({
            "pdb_name": pdb_name,
            "sequences": unique_seqs,
            "scores": scores.tolist(),
            "best_seq": unique_seqs[best_idx],
            "best_score": scores[best_idx].item(),
        })

        pairs = build_preference_pairs(unique_seqs, scores, top_ratio=args.top_ratio)
        if args.max_pairs_per_pdb > 0 and len(pairs) > args.max_pairs_per_pdb:
            np.random.shuffle(pairs)
            pairs = pairs[:args.max_pairs_per_pdb]
        for p in pairs:
            p["pdb_path"] = pdb_path
            p["pdb_name"] = pdb_name
        all_pairs.extend(pairs)
        print(f"  Built {len(pairs)} preference pairs")

    print(f"\n{'='*60}")
    print(f"Total preference pairs: {len(all_pairs)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch.save({
        "pairs": all_pairs,
        "baseline_results": all_baseline_results,
        "args": vars(args),
    }, args.output)
    print(f"Saved to {args.output}")

    baseline_path = args.output.replace(".pt", "_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(all_baseline_results, f, indent=2)
    print(f"Baseline results saved to {baseline_path}")


if __name__ == "__main__":
    main()
