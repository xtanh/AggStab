"""Sample sequences with ProteinMPNN, score with ProAgg, build DPO preference pairs.

Uses rank-aligned pairing (ProtAlign, ICLR 2026): the i-th best sequence is
paired with the (N/2 + i)-th best, and pairs with score gap <= delta are filtered.

Usage:
    python src/dpo/sample_and_score.py \
        --pdb_dir  data/dpo/representative_pdbs/train \
        --proagg_ckpt  results/lightning_logs/version_X/checkpoints/best.ckpt \
        --output  data/dpo/train_pairs.pt \
        --num_samples 12 \
        --temperature 1.0 \
        --device cuda:3
"""

import os
import sys
import glob
import json
import argparse
import torch
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

import pandas as pd
from transformers import EsmTokenizer

from src.mpnn.mpnn_wrapper import (
    load_mpnn_model,
    featurize_pdb,
    sample_sequences,
)
from src.ln.lightning_model import LightningProAggModel
from src.ln.lightning_data import _sa_seq_to_spaced
from src.utils.seed import set_global_seed, unique_preserve_order


def build_struct_token_lookup(data_csv):
    """Build a mapping from protein name -> structural tokens (from SA sequence)."""
    df = pd.read_csv(data_csv)
    lookup = {}
    for _, row in df.iterrows():
        sa = row["sa_sequence_foldseek"]
        struct_tokens = "".join([sa[i + 1] for i in range(0, len(sa), 2)])
        lookup[row["name"]] = struct_tokens
    return lookup


def aa_seq_to_sa_seq(aa_seq, struct_tokens):
    """Combine a new AA sequence with original structural tokens to form SA sequence."""
    L = min(len(aa_seq), len(struct_tokens))
    return "".join([aa_seq[i] + struct_tokens[i] for i in range(L)])


@torch.no_grad()
def score_sequences_with_proagg(
    proagg_model, sequences, struct_tokens, tokenizer, device="cpu", batch_size=32,
):
    """Score a list of AA sequences with SaProt-based ProAgg.

    Converts each AA sequence to SA sequence using the backbone's structural
    tokens, then tokenizes with SaProt tokenizer.
    """
    proagg_model.eval()
    all_scores = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        sa_seqs = [aa_seq_to_sa_seq(s, struct_tokens) for s in batch_seqs]
        spaced = [_sa_seq_to_spaced(sa) for sa in sa_seqs]
        encoded = tokenizer.batch_encode_plus(spaced, return_tensors="pt", padding=True)

        batch_dict = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        out = proagg_model(batch_dict)
        all_scores.append(out["score"].cpu().flatten())

    return torch.cat(all_scores, dim=0)


def build_preference_pairs(sequences, scores, score_gap_delta=0.1):
    """Build (winner, loser) pairs using rank-aligned pairing.

    Rank all N sequences by score (descending), then pair the i-th ranked
    sequence with the (N/2 + i)-th ranked sequence (i < N/2).
    Only keep pairs where score_winner - score_loser > delta.

    Reference: ProtAlign (Liu et al., ICLR 2026), Section 4.4.
    """
    n = len(sequences)
    if n < 2:
        return []

    sorted_idx = torch.argsort(scores, descending=True).tolist()
    half = n // 2

    pairs = []
    for i in range(half):
        w_i = sorted_idx[i]
        l_i = sorted_idx[half + i]
        gap = scores[w_i].item() - scores[l_i].item()
        if gap > score_gap_delta:
            pairs.append({
                "seq_winner": sequences[w_i],
                "seq_loser": sequences[l_i],
                "score_winner": scores[w_i].item(),
                "score_loser": scores[l_i].item(),
                "score_gap": gap,
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
    parser.add_argument("--num_samples", type=int, default=12,
                        help="Number of sequences to sample per backbone")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Rollout temperature (higher = more diverse)")
    parser.add_argument("--score_gap_delta", type=float, default=0.0,
                        help="Min score gap to keep a preference pair (filter noise)")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--proagg_batch_size", type=int, default=16)
    parser.add_argument("--data_csv", type=str,
                        default="data/rocklin/rawdata/data.csv",
                        help="Path to raw data CSV (for structural token lookup)")
    parser.add_argument("--pdb_list_file", type=str, default=None,
                        help="Optional newline-delimited PDB file list, relative to --pdb_dir unless absolute")
    parser.add_argument("--max_pdbs", type=int, default=-1,
                        help="Max number of PDBs to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection and sampling")
    args = parser.parse_args()

    device = args.device
    set_global_seed(args.seed)

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

    print("Loading SaProt tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)

    print("Building structural token lookup...")
    struct_lookup = build_struct_token_lookup(args.data_csv)
    print(f"  {len(struct_lookup)} proteins in lookup")

    # --- Process each PDB ---
    if args.pdb_list_file:
        with open(args.pdb_list_file) as handle:
            entries = [line.strip() for line in handle if line.strip()]
        pdb_files = []
        for entry in entries:
            pdb_path = entry if os.path.isabs(entry) else os.path.join(args.pdb_dir, entry)
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB listed in {args.pdb_list_file} not found: {pdb_path}")
            pdb_files.append(pdb_path)
    else:
        pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_files:
        print(f"No PDB files found in {args.pdb_dir}")
        return

    if args.max_pdbs > 0:
        np.random.seed(args.seed)
        idx = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[i] for i in sorted(idx)]
        print(f"Selected {len(pdb_files)} PDBs (max_pdbs={args.max_pdbs})")

    all_pairs = []
    all_baseline_results = []

    skipped = 0
    for pdb_path in pdb_files:
        pdb_file = os.path.basename(pdb_path)
        protein_name = pdb_file.replace("_ranked_0.pdb", "")
        print(f"\n{'='*60}")
        print(f"Processing: {protein_name}")

        if protein_name not in struct_lookup:
            print(f"  WARNING: no structural tokens found, skipping.")
            skipped += 1
            continue

        struct_tokens = struct_lookup[protein_name]

        feat = featurize_pdb(pdb_path, device=device)

        print(f"  Sampling {args.num_samples} sequences (T={args.temperature})...")
        sequences = sample_sequences(
            mpnn_model, feat,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=device,
        )
        unique_seqs = unique_preserve_order(sequences)
        print(f"  Got {len(unique_seqs)} unique sequences")

        print(f"  Scoring with ProAgg...")
        scores = score_sequences_with_proagg(
            proagg_model, unique_seqs, struct_tokens, tokenizer,
            device=device, batch_size=args.proagg_batch_size,
        )

        best_idx = torch.argmax(scores).item()
        worst_idx = torch.argmin(scores).item()
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}], "
              f"mean={scores.mean():.3f}")

        all_baseline_results.append({
            "pdb_name": protein_name,
            "sequences": unique_seqs,
            "scores": scores.tolist(),
            "best_seq": unique_seqs[best_idx],
            "best_score": scores[best_idx].item(),
        })

        pairs = build_preference_pairs(
            unique_seqs, scores, score_gap_delta=args.score_gap_delta,
        )
        for p in pairs:
            p["pdb_path"] = pdb_path
            p["pdb_name"] = protein_name
        all_pairs.extend(pairs)
        print(f"  Built {len(pairs)} preference pairs")

    print(f"\n{'='*60}")
    if skipped > 0:
        print(f"Skipped {skipped} PDBs (no structural tokens)")
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
