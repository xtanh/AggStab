#!/usr/bin/env python
"""Export all sampled candidate sequences for a set of backbones.

For each backbone:
  - sample `num_samples` sequences from a ProteinMPNN checkpoint
  - deduplicate while preserving order
  - score each sequence with ProAgg
  - compute ProteinMPNN log-probability
  - write one row per candidate sequence
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from transformers import EsmTokenizer

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("scripts")]
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.dpo.sample_and_score import build_struct_token_lookup, score_sequences_with_proagg
from src.ln.lightning_model import LightningProAggModel
from src.mpnn.mpnn_wrapper import compute_log_probs, featurize_pdb, load_mpnn_model, sample_sequences
from src.utils.seed import set_global_seed, unique_preserve_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--mpnn_ckpt", type=str, required=True)
    parser.add_argument("--proagg_ckpt", type=str, required=True)
    parser.add_argument("--proagg_config", type=str, default="configs/proagg_final_candidate.yaml")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proagg_batch_size", type=int, default=32)
    parser.add_argument("--data_csv", type=str, default="data/rocklin/rawdata/data.csv")
    parser.add_argument("--max_pdbs", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = args.device

    cfg = load_yaml_config(args.proagg_config)
    proagg_lightning = LightningProAggModel.load_from_checkpoint(args.proagg_ckpt, cfg=cfg)
    proagg_model = proagg_lightning.model.to(device)
    proagg_model.eval()

    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)
    struct_lookup = build_struct_token_lookup(args.data_csv)
    mpnn_model = load_mpnn_model(checkpoint_path=args.mpnn_ckpt, device=device)

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if args.max_pdbs > 0:
        np.random.seed(args.seed)
        idx = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[i] for i in sorted(idx)]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        "pdb_name",
        "candidate_rank",
        "sequence",
        "proagg_score",
        "mpnn_logprob",
        "num_unique_for_backbone",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pi, pdb_path in enumerate(pdb_files):
            pdb_file = os.path.basename(pdb_path)
            protein_name = pdb_file.replace("_ranked_0.pdb", "")
            print(f"[{pi+1}/{len(pdb_files)}] {protein_name}", flush=True)

            if protein_name not in struct_lookup:
                print("  missing structural tokens, skip", flush=True)
                continue

            struct_tokens = struct_lookup[protein_name]
            feat = featurize_pdb(pdb_path, device=device)

            set_global_seed(args.seed + pi)
            seqs = sample_sequences(
                mpnn_model,
                feat,
                num_samples=args.num_samples,
                temperature=args.temperature,
                device=device,
            )
            seqs = unique_preserve_order(seqs)

            proagg_scores = score_sequences_with_proagg(
                proagg_model,
                seqs,
                struct_tokens,
                tokenizer,
                device=device,
                batch_size=args.proagg_batch_size,
            )
            mpnn_log_probs, _ = compute_log_probs(mpnn_model, feat, seqs, device=device)

            order = torch.argsort(proagg_scores, descending=True).tolist()
            for rank, idx in enumerate(order, start=1):
                writer.writerow(
                    {
                        "pdb_name": protein_name,
                        "candidate_rank": rank,
                        "sequence": seqs[idx],
                        "proagg_score": float(proagg_scores[idx].item()),
                        "mpnn_logprob": float(mpnn_log_probs[idx].item()),
                        "num_unique_for_backbone": len(seqs),
                    }
                )

    print(f"Saved candidate table to {args.output_csv}")


if __name__ == "__main__":
    main()
    
    
