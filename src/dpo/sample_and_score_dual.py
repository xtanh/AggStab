"""Sample sequences with ProteinMPNN and build dual-objective preference pairs.

This script builds two preference datasets from the same sampled candidates:
  - aggregation pairs (`agg_pairs`) ranked by ProAgg `score`
  - stability pairs (`stab_pairs`) ranked by predicted `deltaG`

Aggregation pairs can optionally be gated by a minimum stability threshold to
avoid constructing aggregation preferences from obviously unstable candidates.
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("src")]
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.dpo.sample_and_score import (
    aa_seq_to_sa_seq,
    build_preference_pairs,
    build_struct_token_lookup,
)
from src.ln.lightning_data import _sa_seq_to_spaced
from src.ln.lightning_model import LightningProAggModel
from src.mpnn.mpnn_wrapper import featurize_pdb, load_mpnn_model, sample_sequences
from src.utils.seed import set_global_seed, unique_preserve_order


def load_predictor(checkpoint_path, config_path, device):
    config = load_yaml_config(config_path)
    lightning_model = LightningProAggModel.load_from_checkpoint(checkpoint_path, cfg=config)
    model = lightning_model.model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained(config.model.saprot_path)
    output_key = config.train.get("output_key", "score")
    return model, tokenizer, output_key


def load_stability_lookup(stability_csv):
    delta_g_df = pd.read_csv(stability_csv)
    if "name" not in delta_g_df.columns or "deltaG" not in delta_g_df.columns:
        raise ValueError(f"{stability_csv} must contain 'name' and 'deltaG' columns")
    return dict(zip(delta_g_df["name"], delta_g_df["deltaG"]))


@torch.no_grad()
def score_sequences_with_predictor(
    predictor_model,
    output_key,
    sequences,
    struct_tokens,
    tokenizer,
    device="cpu",
    batch_size=32,
):
    predictor_model.eval()
    all_scores = []

    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        sa_sequences = [aa_seq_to_sa_seq(sequence, struct_tokens) for sequence in batch_sequences]
        spaced_sequences = [_sa_seq_to_spaced(sequence) for sequence in sa_sequences]
        encoded = tokenizer.batch_encode_plus(spaced_sequences, return_tensors="pt", padding=True)
        batch = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        outputs = predictor_model(batch)
        all_scores.append(outputs[output_key].cpu().flatten())

    return torch.cat(all_scores, dim=0)


def filter_indices_by_stability(
    delta_g_scores,
    mode,
    absolute_min,
    relative_margin,
    wt_delta_g=None,
):
    if mode == "none":
        return list(range(len(delta_g_scores)))

    if mode == "absolute":
        return [index for index, score in enumerate(delta_g_scores.tolist()) if score >= absolute_min]

    if mode == "relative":
        best_delta_g = float(delta_g_scores.max().item())
        threshold = best_delta_g - relative_margin
        return [index for index, score in enumerate(delta_g_scores.tolist()) if score >= threshold]

    if mode == "wt_absolute":
        if wt_delta_g is None:
            return []
        threshold = wt_delta_g - relative_margin
        return [index for index, score in enumerate(delta_g_scores.tolist()) if score >= threshold]

    raise ValueError(f"Unsupported stability gate mode: {mode}")


def build_pairs_with_indices(sequences, scores, kept_indices, score_gap_delta):
    filtered_sequences = [sequences[index] for index in kept_indices]
    filtered_scores = scores[kept_indices]
    return build_preference_pairs(
        filtered_sequences,
        filtered_scores,
        score_gap_delta=score_gap_delta,
    )


def attach_candidate_metrics(pair_list, metrics_by_sequence, objective_name, protein_name, pdb_path):
    enriched = []
    for pair in pair_list:
        winner_metrics = metrics_by_sequence[pair["seq_winner"]]
        loser_metrics = metrics_by_sequence[pair["seq_loser"]]
        enriched.append(
            {
                **pair,
                "objective": objective_name,
                "pdb_name": protein_name,
                "pdb_path": pdb_path,
                "winner_score": winner_metrics["score"],
                "winner_deltaG": winner_metrics["deltaG"],
                "loser_score": loser_metrics["score"],
                "loser_deltaG": loser_metrics["deltaG"],
            }
        )
    return enriched


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
        default="absolute",
        choices=["none", "absolute", "relative", "wt_absolute"],
    )
    parser.add_argument("--stability_gate_min", type=float, default=0.0)
    parser.add_argument("--stability_gate_margin", type=float, default=0.5)
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

    agg_pairs = []
    stab_pairs = []
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

        metrics_by_sequence = {
            sequence: {
                "score": float(agg_score.item()),
                "deltaG": float(stab_score.item()),
            }
            for sequence, agg_score, stab_score in zip(unique_sequences, agg_scores, stab_scores)
        }

        kept_indices = filter_indices_by_stability(
            stab_scores,
            mode=args.stability_gate_mode,
            absolute_min=args.stability_gate_min,
            relative_margin=args.stability_gate_margin,
            wt_delta_g=wt_delta_g,
        )

        agg_pair_list = build_pairs_with_indices(
            unique_sequences,
            agg_scores,
            kept_indices,
            score_gap_delta=args.agg_score_gap_delta,
        )
        stab_pair_list = build_preference_pairs(
            unique_sequences,
            stab_scores,
            score_gap_delta=args.stab_score_gap_delta,
        )

        agg_pairs.extend(
            attach_candidate_metrics(
                agg_pair_list,
                metrics_by_sequence=metrics_by_sequence,
                objective_name="agg",
                protein_name=protein_name,
                pdb_path=pdb_path,
            )
        )
        stab_pairs.extend(
            attach_candidate_metrics(
                stab_pair_list,
                metrics_by_sequence=metrics_by_sequence,
                objective_name="stab",
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
                "num_kept_for_agg_pairs": len(kept_indices),
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
            f"  Built agg_pairs={len(agg_pair_list)} (kept {len(kept_indices)} seqs), "
            f"stab_pairs={len(stab_pair_list)}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "agg_pairs": agg_pairs,
            "stab_pairs": stab_pairs,
            "baseline_results": baseline_results,
            "args": vars(args),
        },
        args.output,
    )
    print(f"\nSaved dual-objective pairs to {args.output}", flush=True)
    print(f"Skipped {skipped} PDBs", flush=True)

    baseline_path = args.output.replace(".pt", "_baseline.json")
    with open(baseline_path, "w") as handle:
        json.dump(baseline_results, handle, indent=2)
    print(f"Saved baseline candidate stats to {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
