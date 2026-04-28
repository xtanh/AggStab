#!/usr/bin/env python

import argparse
import csv
import json
import os
from statistics import mean


def _safe_mean(values):
    return mean(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Summarize dual-objective DPO eval results.")
    parser.add_argument("eval_json", type=str, help="Path to dual eval_results.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: eval_json parent)")
    args = parser.parse_args()

    with open(args.eval_json) as handle:
        payload = json.load(handle)

    original = payload["original"]
    dpo = payload["dpo"]
    if len(original) != len(dpo):
        raise ValueError("original and dpo result lists must have the same length")

    output_dir = args.output_dir or os.path.dirname(args.eval_json)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for orig_row, dpo_row in zip(original, dpo):
        if orig_row["pdb_name"] != dpo_row["pdb_name"]:
            raise ValueError(f"Mismatched pdb_name: {orig_row['pdb_name']} vs {dpo_row['pdb_name']}")

        rows.append(
            {
                "pdb_name": orig_row["pdb_name"],
                "orig_proagg_mean": orig_row["proagg_mean"],
                "dpo_proagg_mean": dpo_row["proagg_mean"],
                "delta_proagg_mean": dpo_row["proagg_mean"] - orig_row["proagg_mean"],
                "orig_proagg_max": orig_row["proagg_max"],
                "dpo_proagg_max": dpo_row["proagg_max"],
                "delta_proagg_max": dpo_row["proagg_max"] - orig_row["proagg_max"],
                "orig_deltaG_mean": orig_row.get("deltaG_mean"),
                "dpo_deltaG_mean": dpo_row.get("deltaG_mean"),
                "delta_deltaG_mean": dpo_row.get("deltaG_mean", 0.0) - orig_row.get("deltaG_mean", 0.0),
                "orig_deltaG_max": orig_row.get("deltaG_max"),
                "dpo_deltaG_max": dpo_row.get("deltaG_max"),
                "delta_deltaG_max": dpo_row.get("deltaG_max", 0.0) - orig_row.get("deltaG_max", 0.0),
                "orig_logprob_mean": orig_row["mpnn_logprob_mean"],
                "dpo_logprob_mean": dpo_row["mpnn_logprob_mean"],
                "delta_logprob_mean": dpo_row["mpnn_logprob_mean"] - orig_row["mpnn_logprob_mean"],
                "orig_diversity": orig_row["diversity"],
                "dpo_diversity": dpo_row["diversity"],
                "delta_diversity": dpo_row["diversity"] - orig_row["diversity"],
                "improve_proagg_mean": int(dpo_row["proagg_mean"] > orig_row["proagg_mean"]),
                "improve_proagg_max": int(dpo_row["proagg_max"] > orig_row["proagg_max"]),
                "improve_deltaG_mean": int(dpo_row.get("deltaG_mean", float("-inf")) > orig_row.get("deltaG_mean", float("inf"))),
                "improve_deltaG_max": int(dpo_row.get("deltaG_max", float("-inf")) > orig_row.get("deltaG_max", float("inf"))),
            }
        )

    summary = {
        "n_backbones": len(rows),
        "delta_proagg_mean": _safe_mean([row["delta_proagg_mean"] for row in rows]),
        "delta_proagg_max": _safe_mean([row["delta_proagg_max"] for row in rows]),
        "delta_deltaG_mean": _safe_mean([row["delta_deltaG_mean"] for row in rows]),
        "delta_deltaG_max": _safe_mean([row["delta_deltaG_max"] for row in rows]),
        "delta_logprob_mean": _safe_mean([row["delta_logprob_mean"] for row in rows]),
        "delta_diversity": _safe_mean([row["delta_diversity"] for row in rows]),
        "improve_proagg_mean": sum(row["improve_proagg_mean"] for row in rows),
        "improve_proagg_max": sum(row["improve_proagg_max"] for row in rows),
        "improve_deltaG_mean": sum(row["improve_deltaG_mean"] for row in rows),
        "improve_deltaG_max": sum(row["improve_deltaG_max"] for row in rows),
        "args": payload.get("args", {}),
    }

    summary_path = os.path.join(output_dir, "dual_eval_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = os.path.join(output_dir, "dual_eval_per_backbone.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = os.path.join(output_dir, "dual_eval_report.txt")
    with open(report_path, "w") as handle:
        handle.write(f"N (test backbones): {summary['n_backbones']}\n")
        handle.write(f"delta_proagg_mean: {summary['delta_proagg_mean']:.4f}\n")
        handle.write(f"delta_proagg_max: {summary['delta_proagg_max']:.4f}\n")
        handle.write(f"delta_deltaG_mean: {summary['delta_deltaG_mean']:.4f}\n")
        handle.write(f"delta_deltaG_max: {summary['delta_deltaG_max']:.4f}\n")
        handle.write(f"delta_logprob_mean: {summary['delta_logprob_mean']:.4f}\n")
        handle.write(f"delta_diversity: {summary['delta_diversity']:.4f}\n")
        handle.write(f"improve_proagg_mean: {summary['improve_proagg_mean']}/{summary['n_backbones']}\n")
        handle.write(f"improve_proagg_max: {summary['improve_proagg_max']}/{summary['n_backbones']}\n")
        handle.write(f"improve_deltaG_mean: {summary['improve_deltaG_mean']}/{summary['n_backbones']}\n")
        handle.write(f"improve_deltaG_max: {summary['improve_deltaG_max']}/{summary['n_backbones']}\n")

    print(f"Saved summary to {summary_path}")
    print(f"Saved per-backbone CSV to {csv_path}")
    print(f"Saved text report to {report_path}")


if __name__ == "__main__":
    main()
