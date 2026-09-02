#!/usr/bin/env python
"""Export all sampled candidate sequences for a set of backbones.

For each backbone:
  - sample `num_samples` sequences from a ProteinMPNN checkpoint
  - deduplicate while preserving order
  - score each sequence with ProAgg
  - optionally score each sequence with deltaG
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
import pandas as pd
import torch
from transformers import EsmTokenizer

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("scripts")]
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.dpo.sample_and_score import build_struct_token_lookup, score_sequences_with_proagg
from src.dpo.sample_and_score_dual import load_stability_lookup, score_sequences_with_predictor
from src.ln.lightning_model import LightningProAggModel
from src.mpnn.mpnn_wrapper import compute_log_probs, featurize_pdb, load_mpnn_model, sample_sequences
from src.utils.seed import set_global_seed, unique_preserve_order


def compute_log_probs_in_batches(model, feat, seqs, *, device: str, batch_size: int) -> torch.Tensor:
    """Compute ProteinMPNN log-probabilities without batching all candidates at once."""
    chunks: list[torch.Tensor] = []
    for start in range(0, len(seqs), batch_size):
        chunk_seqs = seqs[start : start + batch_size]
        log_probs, _ = compute_log_probs(model, feat, chunk_seqs, device=device)
        chunks.append(log_probs.detach().cpu())
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--mpnn_ckpt", type=str, required=True)
    parser.add_argument("--proagg_ckpt", type=str, required=True)
    parser.add_argument("--proagg_config", type=str, default="configs/proagg_final_candidate.yaml")
    parser.add_argument("--stab_ckpt", type=str, default=None)
    parser.add_argument("--stab_config", type=str, default=None)
    parser.add_argument("--stability_csv", type=str, default="data/rocklin/Metagenomic_dG.csv")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proagg_batch_size", type=int, default=32)
    parser.add_argument("--mpnn_logprob_batch_size", type=int, default=16)
    parser.add_argument("--data_csv", type=str, default="data/rocklin/rawdata/data.csv")
    parser.add_argument("--max_pdbs", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Append to output_csv and skip completed pdb_name values.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = args.device

    cfg = load_yaml_config(args.proagg_config)
    proagg_lightning = LightningProAggModel.load_from_checkpoint(args.proagg_ckpt, cfg=cfg)
    proagg_model = proagg_lightning.model.to(device)
    proagg_model.eval()

    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)
    stab_model = None
    stab_tokenizer = None
    stab_output_key = "deltaG"
    wt_delta_g_lookup = None
    if args.stab_ckpt and args.stab_config:
        stab_cfg = load_yaml_config(args.stab_config)
        stab_lightning = LightningProAggModel.load_from_checkpoint(args.stab_ckpt, cfg=stab_cfg)
        stab_model = stab_lightning.model.to(device)
        stab_model.eval()
        stab_tokenizer = EsmTokenizer.from_pretrained(stab_cfg.model.saprot_path)
        wt_delta_g_lookup = load_stability_lookup(args.stability_csv)

    struct_lookup = build_struct_token_lookup(args.data_csv)
    mpnn_model = load_mpnn_model(checkpoint_path=args.mpnn_ckpt, device=device)

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if args.max_pdbs > 0:
        np.random.seed(args.seed)
        idx = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[i] for i in sorted(idx)]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    completed_pdbs: set[str] = set()
    output_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    if args.resume and output_exists:
        existing = pd.read_csv(args.output_csv, usecols=["pdb_name"])
        completed_pdbs = set(existing["pdb_name"].astype(str).unique())
        print(f"Resume enabled: skipping {len(completed_pdbs)} completed backbones", flush=True)

    fieldnames = [
        "pdb_name",
        "candidate_rank",
        "sequence",
        "proagg_score",
        "deltaG",
        "wt_deltaG",
        "deltaG_minus_wt",
        "mpnn_logprob",
        "num_unique_for_backbone",
    ]

    mode = "a" if args.resume and output_exists else "w"
    with open(args.output_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for pi, pdb_path in enumerate(pdb_files):
            pdb_file = os.path.basename(pdb_path)
            protein_name = pdb_file.replace("_ranked_0.pdb", "")
            print(f"[{pi+1}/{len(pdb_files)}] {protein_name}", flush=True)

            if protein_name in completed_pdbs:
                print("  already completed, skip", flush=True)
                continue

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
            if stab_model is not None:
                stab_scores = score_sequences_with_predictor(
                    stab_model,
                    stab_output_key,
                    seqs,
                    struct_tokens,
                    stab_tokenizer,
                    device=device,
                    batch_size=args.proagg_batch_size,
                )
                wt_delta_g = wt_delta_g_lookup.get(protein_name) if wt_delta_g_lookup is not None else None
            else:
                stab_scores = None
                wt_delta_g = None
            mpnn_log_probs = compute_log_probs_in_batches(
                mpnn_model,
                feat,
                seqs,
                device=device,
                batch_size=args.mpnn_logprob_batch_size,
            )

            order = torch.argsort(proagg_scores, descending=True).tolist()
            for rank, idx in enumerate(order, start=1):
                delta_g = float(stab_scores[idx].item()) if stab_scores is not None else None
                writer.writerow(
                    {
                        "pdb_name": protein_name,
                        "candidate_rank": rank,
                        "sequence": seqs[idx],
                        "proagg_score": float(proagg_scores[idx].item()),
                        "deltaG": delta_g,
                        "wt_deltaG": wt_delta_g,
                        "deltaG_minus_wt": (delta_g - wt_delta_g) if (delta_g is not None and wt_delta_g is not None) else None,
                        "mpnn_logprob": float(mpnn_log_probs[idx].item()),
                        "num_unique_for_backbone": len(seqs),
                    }
                )

    print(f"Saved candidate table to {args.output_csv}")


if __name__ == "__main__":
    main()
    
    
