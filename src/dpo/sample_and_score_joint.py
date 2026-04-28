"""Sample sequences and build joint-preference pairs for DPO.

Joint preference means:
  - candidates are first filtered by a stability gate
  - winner/loser pairs are formed only when the winner is better on both:
    * aggregation score
    * predicted deltaG
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

from src.dpo.sample_and_score import build_struct_token_lookup
from src.dpo.sample_and_score_dual import (
    attach_candidate_metrics,
    filter_indices_by_stability,
    load_predictor,
    load_stability_lookup,
    score_sequences_with_predictor,
)
from src.mpnn.mpnn_wrapper import featurize_pdb, load_mpnn_model, sample_sequences
from src.utils.seed import set_global_seed, unique_preserve_order


def build_joint_preference_pairs(
    sequences,
    agg_scores,
    stab_scores,
    kept_indices,
    agg_score_gap_delta,
    stab_score_gap_delta,
):
    if len(kept_indices) < 2:
        return []

    ranked = sorted(
        kept_indices,
        key=lambda index: (float(agg_scores[index].item()), float(stab_scores[index].item())),
        reverse=True,
    )
    half = len(ranked) // 2
    if half == 0:
        return []

    pairs = []
    for offset in range(half):
        winner_index = ranked[offset]
        loser_index = ranked[half + offset]
        agg_gap = float(agg_scores[winner_index].item() - agg_scores[loser_index].item())
        stab_gap = float(stab_scores[winner_index].item() - stab_scores[loser_index].item())
        if agg_gap > agg_score_gap_delta and stab_gap > stab_score_gap_delta:
            pairs.append(
                {
                    "seq_winner": sequences[winner_index],
                    "seq_loser": sequences[loser_index],
                    "score_winner": float(agg_scores[winner_index].item()),
                    "score_loser": float(agg_scores[loser_index].item()),
                    "score_gap": agg_gap,
                    "stab_winner": float(stab_scores[winner_index].item()),
                    "stab_loser": float(stab_scores[loser_index].item()),
                    "stab_gap": stab_gap,
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument(
        "--pdb_list_file",
        type=str,
        default=None,
        help="Optional newline-delimited list of PDB filenames or full paths to use instead of scanning pdb_dir.",
    )
    parser.add_argument("--agg_ckpt", type=str, required=True)
    parser.add_argument("--agg_config", type=str, required=True)
    parser.add_argument("--stab_ckpt", type=str, required=True)
    parser.add_argument("--stab_config", type=str, required=True)
    parser.add_argument("--mpnn_ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--agg_score_gap_delta", type=float, default=0.0)
    parser.add_argument("--stab_score_gap_delta", type=float, default=0.0)
    parser.add_argument(
        "--stability_gate_mode",
        type=str,
        default="wt_absolute",
        choices=["none", "absolute", "relative", "wt_absolute"],
    )
    parser.add_argument("--stability_gate_min", type=float, default=0.0)
    parser.add_argument("--stability_gate_margin", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--predictor_batch_size", type=int, default=16)
    parser.add_argument("--data_csv", type=str, default="data/rocklin/rawdata/data.csv")
    parser.add_argument("--stability_csv", type=str, default="data/rocklin/Metagenomic_dG.csv")
    parser.add_argument("--max_pdbs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    set_global_seed(args.seed)

    print("Loading ProteinMPNN...", flush=True)
    mpnn_model = load_mpnn_model(checkpoint_path=args.mpnn_ckpt, device=device)

    print("Loading aggregation predictor...", flush=True)
    agg_model, agg_tokenizer, agg_output_key = load_predictor(args.agg_ckpt, args.agg_config, device)

    print("Loading stability predictor...", flush=True)
    stab_model, stab_tokenizer, stab_output_key = load_predictor(args.stab_ckpt, args.stab_config, device)

    print("Building structural token lookup...", flush=True)
    struct_lookup = build_struct_token_lookup(args.data_csv)
    print(f"  {len(struct_lookup)} proteins in lookup", flush=True)

    print("Building WT stability lookup...", flush=True)
    wt_delta_g_lookup = load_stability_lookup(args.stability_csv)
    print(f"  {len(wt_delta_g_lookup)} proteins with WT deltaG", flush=True)

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
    if args.max_pdbs > 0:
        np.random.seed(args.seed)
        selected = np.random.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[index] for index in sorted(selected)]
        print(f"Selected {len(pdb_files)} PDBs (max_pdbs={args.max_pdbs})", flush=True)

    joint_pairs = []
    baseline_results = []
    skipped = 0

    for pdb_path in pdb_files:
        protein_name = os.path.basename(pdb_path).replace("_ranked_0.pdb", "")
        print(f"\n{'=' * 60}\nProcessing: {protein_name}", flush=True)

        if protein_name not in struct_lookup:
            print("  WARNING: no structural tokens found, skipping.", flush=True)
            skipped += 1
            continue
        if args.stability_gate_mode == "wt_absolute" and protein_name not in wt_delta_g_lookup:
            print("  WARNING: no WT deltaG found for wt_absolute gate, skipping.", flush=True)
            skipped += 1
            continue

        struct_tokens = struct_lookup[protein_name]
        wt_delta_g = wt_delta_g_lookup.get(protein_name)
        features = featurize_pdb(pdb_path, device=device)

        sampled_sequences = sample_sequences(
            mpnn_model,
            features,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=device,
        )
        unique_sequences = unique_preserve_order(sampled_sequences)
        print(f"  Sampled {len(unique_sequences)} unique sequences", flush=True)

        agg_scores = score_sequences_with_predictor(
            agg_model,
            agg_output_key,
            unique_sequences,
            struct_tokens,
            agg_tokenizer,
            device=device,
            batch_size=args.predictor_batch_size,
        )
        stab_scores = score_sequences_with_predictor(
            stab_model,
            stab_output_key,
            unique_sequences,
            struct_tokens,
            stab_tokenizer,
            device=device,
            batch_size=args.predictor_batch_size,
        )

        kept_indices = filter_indices_by_stability(
            stab_scores,
            mode=args.stability_gate_mode,
            absolute_min=args.stability_gate_min,
            relative_margin=args.stability_gate_margin,
            wt_delta_g=wt_delta_g,
        )

        pair_list = build_joint_preference_pairs(
            unique_sequences,
            agg_scores,
            stab_scores,
            kept_indices=kept_indices,
            agg_score_gap_delta=args.agg_score_gap_delta,
            stab_score_gap_delta=args.stab_score_gap_delta,
        )

        metrics_by_sequence = {
            sequence: {
                "score": float(agg_score.item()),
                "deltaG": float(stab_score.item()),
            }
            for sequence, agg_score, stab_score in zip(unique_sequences, agg_scores, stab_scores)
        }
        joint_pairs.extend(
            attach_candidate_metrics(
                pair_list,
                metrics_by_sequence=metrics_by_sequence,
                objective_name="joint",
                protein_name=protein_name,
                pdb_path=pdb_path,
            )
        )

        baseline_results.append(
            {
                "pdb_name": protein_name,
                "sequences": unique_sequences,
                "score": agg_scores.tolist(),
                "deltaG": stab_scores.tolist(),
                "best_agg_seq": unique_sequences[int(torch.argmax(agg_scores).item())],
                "best_stab_seq": unique_sequences[int(torch.argmax(stab_scores).item())],
                "num_kept_for_joint_pairs": len(kept_indices),
                "wt_deltaG": wt_delta_g,
            }
        )

        print(
            f"  Agg score range: [{agg_scores.min():.3f}, {agg_scores.max():.3f}] | "
            f"deltaG range: [{stab_scores.min():.3f}, {stab_scores.max():.3f}]"
            + (f" | WT deltaG={wt_delta_g:.3f}" if wt_delta_g is not None else ""),
            flush=True,
        )
        print(
            f"  Built joint_pairs={len(pair_list)} (kept {len(kept_indices)} seqs)",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "pairs": joint_pairs,
            "baseline_results": baseline_results,
            "args": vars(args),
        },
        args.output,
    )
    print(f"\nSaved joint-preference pairs to {args.output}", flush=True)
    print(f"Skipped {skipped} PDBs", flush=True)

    baseline_path = args.output.replace(".pt", "_baseline.json")
    with open(baseline_path, "w") as handle:
        json.dump(baseline_results, handle, indent=2)
    print(f"Saved baseline candidate stats to {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
