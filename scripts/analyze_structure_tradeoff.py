#!/usr/bin/env python
"""Analyze baseline vs DPO pilot structure/property trade-off.

The main success definition follows the current project discussion:
  - DPO candidate must improve ProAgg relative to the best baseline candidate
    for the same backbone.
  - DPO candidate must also satisfy structure thresholds.
  - Backbone success means at least one top-k candidate for that backbone passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary_dir(path: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(path.glob("summary_shard*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No summary_shard*.csv found in {path}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--dpo_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--plddt_thr", type=float, default=0.80)
    parser.add_argument("--ptm_thr", type=float, default=0.70)
    args = parser.parse_args()

    baseline = load_summary_dir(args.baseline_dir)
    dpo = load_summary_dir(args.dpo_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_best = (
        baseline.groupby("pdb_name")
        .agg(
            baseline_best_proagg=("proagg_score", "max"),
            baseline_best_plddt=("best_plddt", "max"),
            baseline_best_ptm=("best_ptm", "max"),
        )
        .reset_index()
    )

    dpo_joined = dpo.merge(baseline_best, on="pdb_name", how="left", validate="many_to_one")

    dpo_joined["agg_improved_vs_baseline_best"] = (
        dpo_joined["proagg_score"] > dpo_joined["baseline_best_proagg"]
    )
    dpo_joined["structure_pass"] = (
        (dpo_joined["best_plddt"] >= args.plddt_thr)
        & (dpo_joined["best_ptm"] >= args.ptm_thr)
    )
    dpo_joined["joint_success"] = (
        dpo_joined["agg_improved_vs_baseline_best"] & dpo_joined["structure_pass"]
    )

    backbone_summary = (
        dpo_joined.groupby("pdb_name")
        .agg(
            num_candidates=("candidate_id", "count"),
            any_agg_improved=("agg_improved_vs_baseline_best", "any"),
            any_structure_pass=("structure_pass", "any"),
            any_joint_success=("joint_success", "any"),
            dpo_best_proagg=("proagg_score", "max"),
            dpo_best_plddt=("best_plddt", "max"),
            dpo_best_ptm=("best_ptm", "max"),
            baseline_best_proagg=("baseline_best_proagg", "max"),
            baseline_best_plddt=("baseline_best_plddt", "max"),
            baseline_best_ptm=("baseline_best_ptm", "max"),
        )
        .reset_index()
    )

    report = {
        "thresholds": {
            "plddt_thr": args.plddt_thr,
            "ptm_thr": args.ptm_thr,
        },
        "counts": {
            "num_baseline_rows": int(len(baseline)),
            "num_dpo_rows": int(len(dpo)),
            "num_backbones": int(backbone_summary["pdb_name"].nunique()),
        },
        "candidate_rates": {
            "joint_success_rate": float(dpo_joined["joint_success"].mean()),
            "agg_improved_rate": float(dpo_joined["agg_improved_vs_baseline_best"].mean()),
            "structure_pass_rate": float(dpo_joined["structure_pass"].mean()),
        },
        "backbone_rates": {
            "joint_success_rate": float(backbone_summary["any_joint_success"].mean()),
            "agg_improved_rate": float(backbone_summary["any_agg_improved"].mean()),
            "structure_pass_rate": float(backbone_summary["any_structure_pass"].mean()),
        },
        "selection_stage_breakdown": (
            dpo_joined.groupby("selection_stage")["joint_success"]
            .agg(["sum", "count", "mean"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "best_of_3_comparison": {
            "dpo_better_best_plddt_fraction": float(
                (backbone_summary["dpo_best_plddt"] > backbone_summary["baseline_best_plddt"]).mean()
            ),
            "dpo_better_best_ptm_fraction": float(
                (backbone_summary["dpo_best_ptm"] > backbone_summary["baseline_best_ptm"]).mean()
            ),
            "mean_delta_best_plddt": float(
                (backbone_summary["dpo_best_plddt"] - backbone_summary["baseline_best_plddt"]).mean()
            ),
            "mean_delta_best_ptm": float(
                (backbone_summary["dpo_best_ptm"] - backbone_summary["baseline_best_ptm"]).mean()
            ),
        },
    }

    dpo_joined.to_csv(args.output_dir / "dpo_joint_candidate_analysis.csv", index=False)
    backbone_summary.to_csv(args.output_dir / "backbone_joint_success_summary.csv", index=False)
    (args.output_dir / "tradeoff_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
