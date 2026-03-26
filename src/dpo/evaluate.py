"""Evaluate DPO-finetuned ProteinMPNN vs. baseline.

Compares original and DPO-finetuned ProteinMPNN on a set of PDB backbones:
  1. ProAgg score (aggregation predictor)
  1b. Optional stability score (`deltaG`)
  2. ProteinMPNN sequence recovery / log-prob (proxy for designability)
  3. Sequence diversity

Usage:
    python src/dpo/evaluate.py \
        --pdb_dir inputs/pdbs \
        --original_mpnn_ckpt <path or default> \
        --dpo_mpnn_ckpt results/dpo/mpnn_dpo_epoch10.pt \
        --proagg_ckpt results/lightning_logs/version_X/checkpoints/best.ckpt \
        --output results/dpo/eval_results.json \
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

from src.mpnn.mpnn_wrapper import (
    load_mpnn_model,
    featurize_pdb,
    sample_sequences,
    compute_log_probs,
)
from transformers import EsmTokenizer

from src.dpo.sample_and_score import (
    build_struct_token_lookup,
)
from src.dpo.sample_and_score_dual import score_sequences_with_predictor
from src.ln.lightning_model import LightningProAggModel
from src.config.utils import load_yaml_config

from src.utils.seed import set_global_seed, unique_preserve_order


def sequence_diversity(sequences):
    """Average pairwise Hamming distance (normalized)."""
    if len(sequences) <= 1:
        return 0.0
    n = len(sequences)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            min_len = min(len(sequences[i]), len(sequences[j]))
            diffs = sum(a != b for a, b in zip(sequences[i][:min_len], sequences[j][:min_len]))
            total += diffs / max(min_len, 1)
            count += 1
    return total / max(count, 1)


def protein_name_from_pdb_path(pdb_path):
    pdb_file = os.path.basename(pdb_path)
    return pdb_file.replace("_ranked_0.pdb", "").replace(".pdb", "")


def load_cached_original_results(cache_path, requested_names):
    with open(cache_path) as f:
        cached = json.load(f)

    if "original" in cached:
        original_results = cached["original"]
    else:
        original_results = cached

    cached_names = [item["pdb_name"] for item in original_results]
    if cached_names != requested_names:
        raise ValueError(
            "Cached original results do not match requested PDB set. "
            f"cache_path={cache_path} requested={len(requested_names)} cached={len(cached_names)}"
        )
    return original_results


def save_cached_original_results(cache_path, original_results, args, requested_names):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "original": original_results,
        "cache_args": {
            "pdb_dir": args.pdb_dir,
            "original_mpnn_ckpt": args.original_mpnn_ckpt,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_pdbs": args.max_pdbs,
            "seed": args.seed,
            "device": args.device,
        },
        "pdb_names": requested_names,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_model(
    mpnn_model, proagg_model, pdb_files, struct_lookup, tokenizer,
    stab_model=None, stab_tokenizer=None, stab_output_key="deltaG",
    num_samples=64, temperature=0.5,
    proagg_batch_size=8, device="cuda",
    seed=None,
):
    """Evaluate a ProteinMPNN model on a set of PDB backbones."""
    results = []

    for pi, pdb_path in enumerate(pdb_files):
        protein_name = protein_name_from_pdb_path(pdb_path)
        print(f"  [{pi+1}/{len(pdb_files)}] {protein_name}", flush=True)

        if protein_name not in struct_lookup:
            print(f"    WARNING: no structural tokens, skipping.")
            continue

        struct_tokens = struct_lookup[protein_name]
        feat = featurize_pdb(pdb_path, device=device)

        if seed is not None:
            # Reset RNG per-backbone so ORIGINAL vs DPO use the same random stream,
            # improving comparability across runs.
            set_global_seed(int(seed) + int(pi))

        sequences = sample_sequences(
            mpnn_model, feat,
            num_samples=num_samples,
            temperature=temperature,
            device=device,
        )
        unique_seqs = unique_preserve_order(sequences)

        proagg_scores = score_sequences_with_predictor(
            proagg_model, "score", unique_seqs, struct_tokens, tokenizer,
            device=device, batch_size=proagg_batch_size,
        )
        if stab_model is not None:
            stab_scores = score_sequences_with_predictor(
                stab_model, stab_output_key, unique_seqs, struct_tokens, stab_tokenizer,
                device=device, batch_size=proagg_batch_size,
            )
        else:
            stab_scores = None

        mpnn_log_probs, _ = compute_log_probs(
            mpnn_model, feat, unique_seqs, device=device,
        )

        diversity = sequence_diversity(unique_seqs)

        result = {
            "pdb_name": protein_name,
            "n_unique": len(unique_seqs),
            "proagg_mean": proagg_scores.mean().item(),
            "proagg_max": proagg_scores.max().item(),
            "proagg_min": proagg_scores.min().item(),
            "proagg_std": proagg_scores.std().item(),
            "mpnn_logprob_mean": mpnn_log_probs.mean().item(),
            "diversity": diversity,
            "best_seq": unique_seqs[torch.argmax(proagg_scores).item()],
            "best_proagg_score": proagg_scores.max().item(),
        }
        if stab_scores is not None:
            result["deltaG_mean"] = stab_scores.mean().item()
            result["deltaG_max"] = stab_scores.max().item()
            result["deltaG_min"] = stab_scores.min().item()
            result["best_stab_seq"] = unique_seqs[torch.argmax(stab_scores).item()]
            result["best_deltaG"] = stab_scores.max().item()
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--original_mpnn_ckpt", type=str, default=None)
    parser.add_argument("--dpo_mpnn_ckpt", type=str, required=True)
    parser.add_argument("--proagg_ckpt", type=str, required=True)
    parser.add_argument("--proagg_config", type=str, default="configs/default.yaml")
    parser.add_argument("--stab_ckpt", type=str, default=None)
    parser.add_argument("--stab_config", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/dpo/eval_results.json")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature. Should match the temperature used "
                             "when generating training pairs (default 0.5).")
    parser.add_argument("--data_csv", type=str,
                        default="data/rocklin/rawdata/data.csv",
                        help="Path to raw data CSV (for structural token lookup)")
    parser.add_argument("--max_pdbs", type=int, default=-1,
                        help="Max PDBs to evaluate (-1 for all)")
    parser.add_argument("--proagg_batch_size", type=int, default=8,
                        help="Batch size for ProAgg scoring during evaluation")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection and sampling")
    parser.add_argument(
        "--original_results_cache",
        type=str,
        default=None,
        help="Optional JSON cache for ORIGINAL ProteinMPNN results. "
             "If the file exists, reuse it instead of recomputing baseline. "
             "If it does not exist, compute baseline once and save it there.",
    )
    args = parser.parse_args()

    device = args.device
    set_global_seed(args.seed)

    cfg = load_yaml_config(args.proagg_config)
    proagg_lightning = LightningProAggModel.load_from_checkpoint(
        args.proagg_ckpt, cfg=cfg,
    )
    proagg_model = proagg_lightning.model.to(device)
    proagg_model.eval()

    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)
    struct_lookup = build_struct_token_lookup(args.data_csv)

    stab_model = None
    stab_tokenizer = None
    if args.stab_ckpt and args.stab_config:
        stab_cfg = load_yaml_config(args.stab_config)
        stab_lightning = LightningProAggModel.load_from_checkpoint(
            args.stab_ckpt, cfg=stab_cfg,
        )
        stab_model = stab_lightning.model.to(device)
        stab_model.eval()
        stab_tokenizer = EsmTokenizer.from_pretrained(stab_cfg.model.saprot_path)

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_files:
        print(f"No PDB files found in {args.pdb_dir}")
        return

    if args.max_pdbs > 0:
        np.random.seed(args.seed)
        idx = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[i] for i in sorted(idx)]
        print(f"Selected {len(pdb_files)} PDBs for evaluation")

    requested_names = [protein_name_from_pdb_path(p) for p in pdb_files]

    # --- Evaluate original ProteinMPNN ---
    print("=" * 60)
    print("Evaluating ORIGINAL ProteinMPNN")
    print("=" * 60)
    if args.original_results_cache and os.path.exists(args.original_results_cache):
        print(f"Reusing cached ORIGINAL results from {args.original_results_cache}")
        original_results = load_cached_original_results(args.original_results_cache, requested_names)
    else:
        original_model = load_mpnn_model(
            checkpoint_path=args.original_mpnn_ckpt, device=device,
        )
        with torch.no_grad():
            original_results = evaluate_model(
                original_model, proagg_model, pdb_files, struct_lookup, tokenizer,
                stab_model=stab_model, stab_tokenizer=stab_tokenizer,
                num_samples=args.num_samples, temperature=args.temperature,
                proagg_batch_size=args.proagg_batch_size,
                device=device,
                seed=args.seed,
            )
        del original_model
        torch.cuda.empty_cache()
        if args.original_results_cache:
            save_cached_original_results(
                args.original_results_cache,
                original_results,
                args,
                requested_names,
            )
            print(f"Saved ORIGINAL results cache to {args.original_results_cache}")

    # --- Evaluate DPO-finetuned ProteinMPNN ---
    print("\n" + "=" * 60)
    print("Evaluating DPO-finetuned ProteinMPNN")
    print("=" * 60)
    dpo_model = load_mpnn_model(
        checkpoint_path=args.dpo_mpnn_ckpt, device=device,
    )
    with torch.no_grad():
        dpo_results = evaluate_model(
            dpo_model, proagg_model, pdb_files, struct_lookup, tokenizer,
            stab_model=stab_model, stab_tokenizer=stab_tokenizer,
            num_samples=args.num_samples, temperature=args.temperature,
            proagg_batch_size=args.proagg_batch_size,
            device=device,
            seed=args.seed,
        )
    del dpo_model
    torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for orig, dpo in zip(original_results, dpo_results):
        name = orig["pdb_name"]
        print(f"\n  {name}:")
        print(f"    ProAgg mean:  original={orig['proagg_mean']:.4f}  "
              f"dpo={dpo['proagg_mean']:.4f}  "
              f"delta={dpo['proagg_mean']-orig['proagg_mean']:+.4f}")
        print(f"    ProAgg max:   original={orig['proagg_max']:.4f}  "
              f"dpo={dpo['proagg_max']:.4f}  "
              f"delta={dpo['proagg_max']-orig['proagg_max']:+.4f}")
        if "deltaG_mean" in orig and "deltaG_mean" in dpo:
            print(f"    deltaG mean:  original={orig['deltaG_mean']:.4f}  "
                  f"dpo={dpo['deltaG_mean']:.4f}  "
                  f"delta={dpo['deltaG_mean']-orig['deltaG_mean']:+.4f}")
            print(f"    deltaG max:   original={orig['deltaG_max']:.4f}  "
                  f"dpo={dpo['deltaG_max']:.4f}  "
                  f"delta={dpo['deltaG_max']-orig['deltaG_max']:+.4f}")
        print(f"    MPNN logprob: original={orig['mpnn_logprob_mean']:.4f}  "
              f"dpo={dpo['mpnn_logprob_mean']:.4f}")
        print(f"    Diversity:    original={orig['diversity']:.4f}  "
              f"dpo={dpo['diversity']:.4f}")

    output_data = {
        "original": original_results,
        "dpo": dpo_results,
        "args": vars(args),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

