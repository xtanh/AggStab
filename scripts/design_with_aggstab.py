#!/usr/bin/env python
"""Design and rank AggStab sequences for a single fixed backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_joint_topk_candidates import select_group, seq_stats  # noqa: E402
from src.inference import (  # noqa: E402
    extract_backbone_context,
    load_reward_predictor,
    score_with_reward,
    validate_sequence,
)
from src.mpnn.mpnn_wrapper import (  # noqa: E402
    compute_log_probs,
    featurize_pdb,
    load_mpnn_model,
    sample_sequences,
)
from src.utils.seed import set_global_seed, unique_preserve_order  # noqa: E402


def batched_log_probabilities(model, features, sequences, *, device: str, batch_size: int) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        log_probabilities, _ = compute_log_probs(model, features, batch, device=device)
        values.append(log_probabilities.detach().float().cpu())
    return torch.cat(values)


def write_fasta(rows: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w") as handle:
        for row in rows.itertuples(index=False):
            handle.write(
                f">AggStab_top{row.topk_rank}|agg={row.proagg_score:.4f}|"
                f"deltaG={row.deltaG:.4f}|stage={row.selection_stage}\n{row.sequence}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Single-chain target backbone in PDB format")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--policy_ckpt", type=Path, required=True)
    parser.add_argument("--aggregation_ckpt", type=Path, required=True)
    parser.add_argument("--aggregation_config", type=Path, default=Path("configs/proagg_final_candidate.yaml"))
    parser.add_argument("--stability_ckpt", type=Path, required=True)
    parser.add_argument("--stability_config", type=Path, default=Path("configs/proagg_deltaG_only.yaml"))
    parser.add_argument("--foldseek_bin", type=Path)
    parser.add_argument(
        "--mask_low_confidence",
        action="store_true",
        help="Replace 3Di tokens with # where PDB B-factors encode pLDDT below the threshold",
    )
    parser.add_argument("--plddt_threshold", type=float, default=70.0)
    parser.add_argument("--num_samples", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_top_fraction", type=float, default=0.40)
    parser.add_argument("--max_charged_fraction", type=float, default=0.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_samples < args.top_k:
        parser.error("--num_samples must be at least --top_k")
    if args.batch_size < 1 or args.top_k < 1:
        parser.error("--batch_size and --top_k must be positive")

    set_global_seed(args.seed)
    wt_sequence, structure_sequence = extract_backbone_context(
        args.pdb,
        chain=args.chain,
        foldseek_bin=args.foldseek_bin,
        mask_low_confidence=args.mask_low_confidence,
        plddt_threshold=args.plddt_threshold,
    )

    policy = load_mpnn_model(args.policy_ckpt, device=args.device)
    features = featurize_pdb(str(args.pdb), device=args.device)
    sampled = sample_sequences(
        policy,
        features,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
    )
    sequences = [validate_sequence(seq, len(wt_sequence)) for seq in unique_preserve_order(sampled)]
    if len(sequences) < args.top_k:
        raise RuntimeError(f"Only {len(sequences)} unique valid sequences were sampled")

    aggregation_model, aggregation_tokenizer, aggregation_key = load_reward_predictor(
        args.aggregation_ckpt, args.aggregation_config, args.device
    )
    stability_model, stability_tokenizer, stability_key = load_reward_predictor(
        args.stability_ckpt, args.stability_config, args.device
    )
    score_sequences = [wt_sequence, *sequences]
    aggregation_scores = score_with_reward(
        aggregation_model,
        aggregation_tokenizer,
        aggregation_key,
        score_sequences,
        structure_sequence,
        device=args.device,
        batch_size=args.batch_size,
    ).numpy()
    stability_scores = score_with_reward(
        stability_model,
        stability_tokenizer,
        stability_key,
        score_sequences,
        structure_sequence,
        device=args.device,
        batch_size=args.batch_size,
    ).numpy()
    log_probabilities = batched_log_probabilities(
        policy, features, sequences, device=args.device, batch_size=args.batch_size
    ).numpy()

    candidate_rows = pd.DataFrame(
        {
            "pdb_name": args.pdb.stem,
            "sequence": sequences,
            "candidate_rank": range(1, len(sequences) + 1),
            "proagg_score": aggregation_scores[1:],
            "wt_proagg_score": aggregation_scores[0],
            "proagg_minus_wt": aggregation_scores[1:] - aggregation_scores[0],
            "deltaG": stability_scores[1:],
            "wt_deltaG": stability_scores[0],
            "deltaG_minus_wt": stability_scores[1:] - stability_scores[0],
            "mpnn_logprob": log_probabilities,
            "num_unique_for_backbone": len(sequences),
        }
    )
    candidate_rows = pd.concat(
        [candidate_rows, pd.DataFrame([seq_stats(seq) for seq in sequences])],
        axis=1,
    )
    selected = select_group(
        candidate_rows,
        top_k=args.top_k,
        max_top_frac=args.max_top_fraction,
        max_charged_frac=args.max_charged_fraction,
    ).copy()
    selected["topk_rank"] = range(1, len(selected) + 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "all_candidates.csv"
    selected_path = args.output_dir / f"top{args.top_k}_candidates.csv"
    fasta_path = args.output_dir / f"top{args.top_k}_candidates.fasta"
    config_path = args.output_dir / "run_config.json"
    candidate_rows.to_csv(all_path, index=False)
    selected.to_csv(selected_path, index=False)
    write_fasta(selected, fasta_path)
    config_path.write_text(
        json.dumps(
            {
                "pdb": str(args.pdb.resolve()),
                "chain": args.chain,
                "num_samples": args.num_samples,
                "num_unique": len(sequences),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seed": args.seed,
                "wt_sequence": wt_sequence,
                "wt_aggregation_resistance_score": float(aggregation_scores[0]),
                "wt_predicted_deltaG": float(stability_scores[0]),
            },
            indent=2,
        )
        + "\n"
    )
    print(selected[["topk_rank", "sequence", "proagg_score", "deltaG", "selection_stage"]].to_string(index=False))
    print(f"Saved all candidates to {all_path}")
    print(f"Saved selected candidates to {selected_path} and {fasta_path}")


if __name__ == "__main__":
    main()
