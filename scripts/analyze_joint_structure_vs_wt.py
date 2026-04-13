#!/usr/bin/env python
"""Analyze baseline vs best-run structure/property results against WT thresholds."""

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


def load_model_table(candidates_csv: Path, chai_dir: Path, wt_agg: pd.DataFrame) -> pd.DataFrame:
    cand = pd.read_csv(candidates_csv)
    chai = load_summary_dir(chai_dir)
    merge_cols = [
        "pdb_name",
        "topk_rank",
        "candidate_rank",
        "selection_stage",
        "sequence",
        "proagg_score",
        "mpnn_logprob",
    ]
    out = chai.merge(cand, on=merge_cols, how="left", validate="one_to_one")
    out = out.merge(wt_agg, on="pdb_name", how="left", validate="many_to_one")
    out["agg_gt_wt"] = out["proagg_score"] > out["wt_proagg_score"]
    out["dg_gt_wt"] = out["deltaG_minus_wt"] > 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_candidates_csv", type=Path, required=True)
    parser.add_argument("--best_chai_dir", type=Path, required=True)
    parser.add_argument("--baseline_candidates_csv", type=Path, required=True)
    parser.add_argument("--baseline_chai_dir", type=Path, required=True)
    parser.add_argument("--wt_agg_csv", type=Path, default=None, help="CSV with columns pdb_name, wt_proagg_score")
    parser.add_argument("--raw_data_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--wt_agg_column", type=str, default="log2_fold_change_75_clip")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--plddt_thr", type=float, default=0.85)
    parser.add_argument("--ptm_thr", type=float, default=0.80)
    args = parser.parse_args()

    if args.wt_agg_csv is not None:
        wt_agg = pd.read_csv(args.wt_agg_csv)
    else:
        raw = pd.read_csv(args.raw_data_csv)
        wt_agg = raw[["name", args.wt_agg_column]].rename(
            columns={"name": "pdb_name", args.wt_agg_column: "wt_proagg_score"}
        )

    best = load_model_table(args.best_candidates_csv, args.best_chai_dir, wt_agg)
    baseline = load_model_table(args.baseline_candidates_csv, args.baseline_chai_dir, wt_agg)

    def add_flags(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["structure_pass"] = (out["best_plddt"] > args.plddt_thr) & (out["best_ptm"] > args.ptm_thr)
        out["joint_success"] = out["structure_pass"] & out["agg_gt_wt"] & out["dg_gt_wt"]
        return out

    best = add_flags(best)
    baseline = add_flags(baseline)

    def backbone_mean(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grouped = df.groupby("pdb_name").agg(
            mean_plddt=("best_plddt", "mean"),
            mean_ptm=("best_ptm", "mean"),
            mean_proagg=("proagg_score", "mean"),
            mean_dgmwt=("deltaG_minus_wt", "mean"),
            mean_logprob=("mpnn_logprob", "mean"),
            mean_joint_success=("joint_success", "mean"),
            mean_structure_pass=("structure_pass", "mean"),
            mean_agg_gt_wt=("agg_gt_wt", "mean"),
            mean_dg_gt_wt=("dg_gt_wt", "mean"),
        )
        return grouped.rename(columns={c: f"{prefix}_{c}" for c in grouped.columns})

    best_bb = backbone_mean(best, "best")
    base_bb = backbone_mean(baseline, "base")
    joined = best_bb.join(base_bb, how="inner")

    report = {
        "thresholds": {"plddt_thr": args.plddt_thr, "ptm_thr": args.ptm_thr},
        "wt_agg_definition": "wt_proagg_score from csv" if args.wt_agg_csv is not None else f"{args.raw_data_csv}:{args.wt_agg_column}",
        "counts": {"num_backbones": int(len(joined))},
        "backbone_mean_of_top3": {
            "best": {
                "proagg_score": float(joined["best_mean_proagg"].mean()),
                "deltaG_minus_wt": float(joined["best_mean_dgmwt"].mean()),
                "mpnn_logprob": float(joined["best_mean_logprob"].mean()),
                "best_plddt": float(joined["best_mean_plddt"].mean()),
                "best_ptm": float(joined["best_mean_ptm"].mean()),
                "joint_success_rate": float(joined["best_mean_joint_success"].mean()),
            },
            "baseline": {
                "proagg_score": float(joined["base_mean_proagg"].mean()),
                "deltaG_minus_wt": float(joined["base_mean_dgmwt"].mean()),
                "mpnn_logprob": float(joined["base_mean_logprob"].mean()),
                "best_plddt": float(joined["base_mean_plddt"].mean()),
                "best_ptm": float(joined["base_mean_ptm"].mean()),
                "joint_success_rate": float(joined["base_mean_joint_success"].mean()),
            },
        },
        "mean_deltas": {
            "proagg_score": float((joined["best_mean_proagg"] - joined["base_mean_proagg"]).mean()),
            "deltaG_minus_wt": float((joined["best_mean_dgmwt"] - joined["base_mean_dgmwt"]).mean()),
            "mpnn_logprob": float((joined["best_mean_logprob"] - joined["base_mean_logprob"]).mean()),
            "best_plddt": float((joined["best_mean_plddt"] - joined["base_mean_plddt"]).mean()),
            "best_ptm": float((joined["best_mean_ptm"] - joined["base_mean_ptm"]).mean()),
            "joint_success_rate": float((joined["best_mean_joint_success"] - joined["base_mean_joint_success"]).mean()),
        },
        "fraction_backbones_better": {
            "proagg_score": float((joined["best_mean_proagg"] > joined["base_mean_proagg"]).mean()),
            "deltaG_minus_wt": float((joined["best_mean_dgmwt"] > joined["base_mean_dgmwt"]).mean()),
            "mpnn_logprob": float((joined["best_mean_logprob"] > joined["base_mean_logprob"]).mean()),
            "best_plddt": float((joined["best_mean_plddt"] > joined["base_mean_plddt"]).mean()),
            "best_ptm": float((joined["best_mean_ptm"] > joined["base_mean_ptm"]).mean()),
            "joint_success_rate": float((joined["best_mean_joint_success"] > joined["base_mean_joint_success"]).mean()),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.reset_index().to_csv(args.output_dir / "backbone_mean_of_top3_vs_wt.csv", index=False)
    (args.output_dir / "joint_structure_vs_wt_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
