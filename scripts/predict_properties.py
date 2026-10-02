#!/usr/bin/env python
"""Predict aggregation resistance and/or folding stability on a fixed backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    extract_backbone_context,
    load_reward_predictor,
    score_with_reward,
    validate_sequence,
)


def read_sequences(args, wt_sequence: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = [("WT", wt_sequence)]
    if args.sequence:
        records.append((args.sequence_id, args.sequence))
    elif args.fasta:
        records.extend((record.id, str(record.seq)) for record in SeqIO.parse(args.fasta, "fasta"))

    validated: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sequence_id, sequence in records:
        item = (str(sequence_id), validate_sequence(sequence, len(wt_sequence)))
        if item not in seen:
            validated.append(item)
            seen.add(item)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Single-chain target backbone in PDB format")
    parser.add_argument("--chain", default="A")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sequence", help="One amino-acid sequence; omit to score only the PDB sequence")
    input_group.add_argument("--fasta", type=Path, help="FASTA containing sequences to score")
    parser.add_argument("--sequence_id", default="design")
    parser.add_argument("--aggregation_ckpt", type=Path)
    parser.add_argument("--aggregation_config", type=Path, default=Path("configs/proagg_final_candidate.yaml"))
    parser.add_argument("--stability_ckpt", type=Path)
    parser.add_argument("--stability_config", type=Path, default=Path("configs/proagg_deltaG_only.yaml"))
    parser.add_argument("--foldseek_bin", type=Path)
    parser.add_argument(
        "--mask_low_confidence",
        action="store_true",
        help="Replace 3Di tokens with # where PDB B-factors encode pLDDT below the threshold",
    )
    parser.add_argument("--plddt_threshold", type=float, default=70.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    if args.aggregation_ckpt is None and args.stability_ckpt is None:
        parser.error("provide --aggregation_ckpt and/or --stability_ckpt")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")

    wt_sequence, structure_sequence = extract_backbone_context(
        args.pdb,
        chain=args.chain,
        foldseek_bin=args.foldseek_bin,
        mask_low_confidence=args.mask_low_confidence,
        plddt_threshold=args.plddt_threshold,
    )
    records = read_sequences(args, wt_sequence)
    sequences = [sequence for _, sequence in records]
    output = pd.DataFrame(
        {
            "sequence_id": [sequence_id for sequence_id, _ in records],
            "sequence": sequences,
            "is_wt": [sequence == wt_sequence for sequence in sequences],
        }
    )

    if args.aggregation_ckpt is not None:
        model, tokenizer, output_key = load_reward_predictor(
            args.aggregation_ckpt, args.aggregation_config, args.device
        )
        values = score_with_reward(
            model,
            tokenizer,
            output_key,
            sequences,
            structure_sequence,
            device=args.device,
            batch_size=args.batch_size,
        ).numpy()
        output["aggregation_resistance_score"] = values
        output["aggregation_gain_vs_wt"] = values - values[0]

    if args.stability_ckpt is not None:
        model, tokenizer, output_key = load_reward_predictor(
            args.stability_ckpt, args.stability_config, args.device
        )
        values = score_with_reward(
            model,
            tokenizer,
            output_key,
            sequences,
            structure_sequence,
            device=args.device,
            batch_size=args.batch_size,
        ).numpy()
        output["predicted_deltaG"] = values
        output["stability_gain_vs_wt"] = values - values[0]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(output.to_string(index=False))
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
