"""Sample sequences from baseline and DPO ProteinMPNN on a single backbone.

Usage:
  python scripts/sample_single_backbone_compare.py \
    --pdb data/AB42/7Q4B.pdb \
    --dpo_ckpt results/.../mpnn_dpo_best.pt \
    --output_dir results/ab42_7q4b_compare \
    --num_samples 64 \
    --temperature 0.5 \
    --device cuda:3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parent
sys.path.insert(0, str(PROJ_DIR))

from src.mpnn.mpnn_wrapper import load_mpnn_model, featurize_pdb, sample_sequences


def summarize_sequences(sequences: list[str]) -> dict:
    unique_sequences = list(dict.fromkeys(sequences))
    return {
        "n_samples": len(sequences),
        "n_unique": len(unique_sequences),
        "sequences": unique_sequences,
        "top_seq": unique_sequences[0] if unique_sequences else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample baseline vs DPO sequences on one backbone.")
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--dpo_ckpt", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Optional ProteinMPNN checkpoint. Defaults to vanilla v_48_020.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat = featurize_pdb(args.pdb, device=args.device)

    print("Loading baseline ProteinMPNN...")
    baseline_model = load_mpnn_model(checkpoint_path=args.baseline_ckpt, device=args.device)
    print("Sampling baseline sequences...")
    baseline_sequences = sample_sequences(
        baseline_model,
        feat,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
    )
    baseline_summary = summarize_sequences(baseline_sequences)

    print("Loading DPO ProteinMPNN...")
    dpo_model = load_mpnn_model(checkpoint_path=args.dpo_ckpt, device=args.device)
    print("Sampling DPO sequences...")
    dpo_sequences = sample_sequences(
        dpo_model,
        feat,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
    )
    dpo_summary = summarize_sequences(dpo_sequences)

    result = {
        "pdb": args.pdb,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "baseline_ckpt": args.baseline_ckpt,
        "dpo_ckpt": args.dpo_ckpt,
        "baseline": baseline_summary,
        "dpo": dpo_summary,
    }

    json_path = output_dir / "sample_compare.json"
    json_path.write_text(json.dumps(result, indent=2))

    baseline_txt = output_dir / "baseline_sequences.txt"
    baseline_txt.write_text("\n".join(baseline_summary["sequences"]) + ("\n" if baseline_summary["sequences"] else ""))

    dpo_txt = output_dir / "dpo_sequences.txt"
    dpo_txt.write_text("\n".join(dpo_summary["sequences"]) + ("\n" if dpo_summary["sequences"] else ""))

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {baseline_txt}")
    print(f"  {dpo_txt}")
    print("\nSummary:")
    print(f"  baseline: n_unique={baseline_summary['n_unique']} top_seq={baseline_summary['top_seq']}")
    print(f"  dpo:      n_unique={dpo_summary['n_unique']} top_seq={dpo_summary['top_seq']}")


if __name__ == "__main__":
    main()
