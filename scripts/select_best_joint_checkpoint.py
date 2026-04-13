#!/usr/bin/env python

import argparse
import csv
import glob
import json
import os
import subprocess
import sys


def load_summary(summary_path):
    with open(summary_path) as handle:
        return json.load(handle)


def checkpoint_epoch(ckpt_path):
    name = os.path.basename(ckpt_path)
    if "epoch" not in name:
        return -1
    try:
        return int(name.split("epoch", 1)[1].split(".pt", 1)[0])
    except Exception:
        return -1


def build_eval_command(args, ckpt_path, epoch_dir):
    eval_json = os.path.join(epoch_dir, "eval_results.json")
    cmd = [
        sys.executable,
        "src/dpo/evaluate.py",
        "--pdb_dir", args.pdb_dir,
        "--dpo_mpnn_ckpt", ckpt_path,
        "--proagg_ckpt", args.agg_ckpt,
        "--proagg_config", args.agg_config,
        "--stab_ckpt", args.stab_ckpt,
        "--stab_config", args.stab_config,
        "--output", eval_json,
        "--num_samples", str(args.num_samples),
        "--temperature", str(args.temperature),
        "--proagg_batch_size", str(args.batch_size),
        "--max_pdbs", str(args.max_pdbs),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.original_results_cache:
        cmd.extend(["--original_results_cache", args.original_results_cache])
    return cmd, eval_json


def build_analyze_command(eval_json, epoch_dir):
    return [
        sys.executable,
        "scripts/analyze_dual_eval.py",
        eval_json,
        "--output_dir",
        epoch_dir,
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all epoch checkpoints on validation and select best checkpoint.")
    parser.add_argument("--ckpt_dir", required=True, help="Directory containing epoch checkpoints")
    parser.add_argument("--ckpt_glob", default="mpnn_dpo_epoch*.pt", help="Glob for epoch checkpoints inside ckpt_dir")
    parser.add_argument("--pdb_dir", required=True, help="Validation PDB directory")
    parser.add_argument("--agg_ckpt", required=True)
    parser.add_argument("--agg_config", default="configs/proagg_final_candidate.yaml")
    parser.add_argument("--stab_ckpt", required=True)
    parser.add_argument("--stab_config", default="configs/proagg_deltaG_only.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to store per-epoch validation eval")
    parser.add_argument("--original_results_cache", default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_pdbs", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_metric",
        default="joint_sum",
        choices=["joint_sum", "delta_proagg_mean", "delta_deltaG_mean", "delta_proagg_max", "delta_deltaG_max"],
        help="Primary metric used inside each tier.",
    )
    parser.add_argument(
        "--max_penalty",
        type=float,
        default=0.1,
        help="Penalty applied in Tier C for each negative max metric.",
    )
    args = parser.parse_args()

    ckpts = sorted(
        glob.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)),
        key=checkpoint_epoch,
    )
    if not ckpts:
        raise FileNotFoundError(f"No epoch checkpoints found in {args.ckpt_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for ckpt_path in ckpts:
        epoch = checkpoint_epoch(ckpt_path)
        epoch_dir = os.path.join(args.output_dir, f"epoch{epoch:02d}")
        os.makedirs(epoch_dir, exist_ok=True)

        eval_cmd, eval_json = build_eval_command(args, ckpt_path, epoch_dir)
        subprocess.run(eval_cmd, check=True)
        subprocess.run(build_analyze_command(eval_json, epoch_dir), check=True)

        summary = load_summary(os.path.join(epoch_dir, "dual_eval_summary.json"))
        row = {
            "epoch": epoch,
            "checkpoint": ckpt_path,
            "delta_proagg_mean": summary["delta_proagg_mean"],
            "delta_proagg_max": summary["delta_proagg_max"],
            "delta_deltaG_mean": summary["delta_deltaG_mean"],
            "delta_deltaG_max": summary["delta_deltaG_max"],
            "delta_logprob_mean": summary["delta_logprob_mean"],
            "improve_proagg_mean": summary["improve_proagg_mean"],
            "improve_deltaG_mean": summary["improve_deltaG_mean"],
        }
        row["joint_sum"] = row["delta_proagg_mean"] + row["delta_deltaG_mean"]
        row["tier_a"] = (
            row["delta_proagg_mean"] >= 0.0
            and row["delta_proagg_max"] >= 0.0
            and row["delta_deltaG_mean"] >= 0.0
            and row["delta_deltaG_max"] >= 0.0
        )
        row["tier_b"] = (
            row["delta_proagg_mean"] >= 0.0
            and row["delta_deltaG_mean"] >= 0.0
        )
        max_penalties = 0
        if row["delta_proagg_max"] < 0.0:
            max_penalties += 1
        if row["delta_deltaG_max"] < 0.0:
            max_penalties += 1
        row["tier_c_score"] = row["joint_sum"] - args.max_penalty * max_penalties
        rows.append(row)

    tier_a_rows = [r for r in rows if r["tier_a"]]
    tier_b_rows = [r for r in rows if r["tier_b"]]
    if tier_a_rows:
        selected_tier = "A"
        selection_pool = tier_a_rows
        key_fn = lambda r: (r[args.selection_metric], -r["epoch"])
    elif tier_b_rows:
        selected_tier = "B"
        selection_pool = tier_b_rows
        key_fn = lambda r: (r[args.selection_metric], -r["epoch"])
    else:
        selected_tier = "C"
        selection_pool = rows
        key_fn = lambda r: (r["tier_c_score"], -r["epoch"])

    best_row = max(selection_pool, key=key_fn)
    best_row = dict(best_row)
    best_row["selected_tier"] = selected_tier

    csv_path = os.path.join(args.output_dir, "epoch_selection.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_path = os.path.join(args.output_dir, "best_checkpoint.json")
    with open(best_path, "w") as handle:
        json.dump(best_row, handle, indent=2)

    print(f"Saved epoch selection table to {csv_path}")
    print(f"Saved best checkpoint summary to {best_path}")
    print(f"BEST tier={best_row['selected_tier']} epoch={best_row['epoch']} ckpt={best_row['checkpoint']}")


if __name__ == "__main__":
    main()
