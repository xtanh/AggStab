#!/usr/bin/env python
"""Run Chai-1 structure prediction for a candidate CSV.

Designed for the current pilot workflow:
  - input: candidate CSV with one row per sequence
  - output: one subdirectory per candidate, plus a summary CSV

Supports sharding across multiple GPUs by row index modulo `num_shards`.
For single-chain proteins, the default model-selection metric is `plddt`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--chai_repo",
        type=Path,
        default=None,
        help="Optional local chai-lab repo to prepend to sys.path. By default use the installed chai_lab package.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_trunk_recycles", type=int, default=3)
    parser.add_argument("--num_diffn_timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument(
        "--select_metric",
        type=str,
        default="plddt",
        choices=["plddt", "ptm", "aggregate"],
        help="How to choose the best Chai sample for a monomer candidate.",
    )
    args = parser.parse_args()

    if args.chai_repo is not None:
        sys.path.insert(0, str(args.chai_repo))
    from chai_lab.chai1 import run_inference  # type: ignore
    print(f"Using chai_lab from: {Path(sys.modules['chai_lab'].__file__).resolve()}", flush=True)

    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("shard_idx must satisfy 0 <= shard_idx < num_shards")

    shard_df = df[df.index % args.num_shards == args.shard_idx].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / f"tmp_fasta_shard{args.shard_idx}"
    work_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failures: list[dict] = []
    device = torch.device(args.device)

    for local_i, row in enumerate(shard_df.itertuples(index=False), start=1):
        candidate_id = sanitize(f"{row.pdb_name}__top{row.topk_rank}")
        out_dir = args.output_dir / candidate_id
        out_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = work_dir / f"{candidate_id}.fasta"
        fasta_path.write_text(f">protein|name={candidate_id}\n{row.sequence}\n")

        print(
            f"[{local_i}/{len(shard_df)}][shard {args.shard_idx}] {candidate_id}",
            flush=True,
        )

        try:
            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=out_dir,
                num_trunk_recycles=args.num_trunk_recycles,
                num_diffn_timesteps=args.num_diffn_timesteps,
                seed=args.seed,
                device=device,
                use_esm_embeddings=True,
            )
        except Exception as exc:
            error_trace = traceback.format_exc()
            (out_dir / "chai_error.txt").write_text(error_trace)
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "pdb_name": row.pdb_name,
                    "topk_rank": int(row.topk_rank),
                    "candidate_rank": int(row.candidate_rank),
                    "selection_stage": row.selection_stage,
                    "sequence": row.sequence,
                    "proagg_score": float(row.proagg_score),
                    "mpnn_logprob": float(row.mpnn_logprob),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_path": str(out_dir / "chai_error.txt"),
                }
            )
            print(
                f"  FAILED: {candidate_id} :: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        sample_rows = []
        for model_idx, ranking in enumerate(candidates.ranking_data):
            sample_rows.append(
                {
                    "model_idx": model_idx,
                    "aggregate_score": float(ranking.aggregate_score.item()),
                    "ptm": float(ranking.ptm_scores.complex_ptm.item()),
                    "iptm": float(ranking.ptm_scores.interface_ptm.item()),
                    "plddt": float(ranking.plddt_scores.complex_plddt.item()),
                    "cif_path": str(candidates.cif_paths[model_idx]),
                }
            )

        scores_df = pd.DataFrame(sample_rows)
        if args.select_metric == "plddt":
            best_idx = int(scores_df["plddt"].idxmax())
        elif args.select_metric == "ptm":
            best_idx = int(scores_df["ptm"].idxmax())
        else:
            best_idx = int(scores_df["aggregate_score"].idxmax())

        best_row = scores_df.loc[best_idx]
        best_cif_src = Path(best_row["cif_path"])
        best_cif_dst = out_dir / "best.cif"
        if best_cif_src.resolve() != best_cif_dst.resolve():
            shutil.copy2(best_cif_src, best_cif_dst)

        scores_df.to_csv(out_dir / "chai_samples.csv", index=False)

        records.append(
            {
                "candidate_id": candidate_id,
                "pdb_name": row.pdb_name,
                "topk_rank": int(row.topk_rank),
                "candidate_rank": int(row.candidate_rank),
                "selection_stage": row.selection_stage,
                "sequence": row.sequence,
                "proagg_score": float(row.proagg_score),
                "mpnn_logprob": float(row.mpnn_logprob),
                "best_model_idx": int(best_row["model_idx"]),
                "best_metric": args.select_metric,
                "best_metric_value": float(best_row[args.select_metric if args.select_metric != "aggregate" else "aggregate_score"]),
                "best_plddt": float(best_row["plddt"]),
                "best_ptm": float(best_row["ptm"]),
                "best_iptm": float(best_row["iptm"]),
                "best_aggregate_score": float(best_row["aggregate_score"]),
                "best_cif": str(best_cif_dst),
            }
        )

    summary_path = args.output_dir / f"summary_shard{args.shard_idx}.csv"
    pd.DataFrame(records).to_csv(summary_path, index=False)
    print(f"Saved shard summary: {summary_path}")

    failures_path = args.output_dir / f"failures_shard{args.shard_idx}.csv"
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"Saved shard failures: {failures_path}")

    stats = {
        "num_input": int(len(shard_df)),
        "num_success": int(len(records)),
        "num_failed": int(len(failures)),
        "success_rate": float(len(records) / len(shard_df)) if len(shard_df) else 0.0,
    }
    stats_path = args.output_dir / f"stats_shard{args.shard_idx}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved shard stats: {stats_path}")


if __name__ == "__main__":
    main()
