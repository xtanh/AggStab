#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "ESM-IF", "MoMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "ESM-IF": "#B8D2CC",
    "MoMPNN": "#F0C987",
    "AggStab-DPO": "#E8B2A7",
    "AggStab-SFT": "#A7C4BC",
}
RUNS = {
    "ProteinMPNN": {
        "candidates_csv": Path("results/mpnn_baseline_fulltest_n16_joint/top3_joint_candidates.csv"),
        "chai_dir": Path("results/mpnn_baseline_fulltest_n16_joint/chai_top3_joint"),
    },
    "SolubleMPNN": {
        "candidates_csv": Path("results/mpnn_soluble_fulltest_n16_joint/top3_joint_candidates.csv"),
        "chai_dir": Path("results/mpnn_soluble_fulltest_n16_joint/chai_top3_joint"),
    },
    "ESM-IF": {
        "candidates_csv": Path("results/esmif_fulltest_n16_joint/top3_joint_candidates.csv"),
        "chai_dir": Path("results/esmif_fulltest_n16_joint/chai_top3_joint"),
    },
    "MoMPNN": {
        "candidates_csv": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/top3_joint_candidates.csv"),
        "chai_dir": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/chai_top3_joint"),
    },
    "AggStab-SFT": {
        "candidates_csv": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv"),
        "chai_dir": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint"),
    },
    "AggStab-DPO": {
        "candidates_csv": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/top3_joint_candidates.csv"),
        "chai_dir": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/chai_top3_joint"),
    },
}
THRESHOLDS = [
    ("No structure filter", None, None),
    ("pLDDT>0.75\npTM>0.60", 0.75, 0.60),
    ("pLDDT>0.80\npTM>0.70", 0.80, 0.70),
    ("pLDDT>0.85\npTM>0.80", 0.85, 0.80),
]


def load_summary_dir(path: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(path.glob("summary_shard*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No summary_shard*.csv found in {path}")
    return pd.concat(frames, ignore_index=True)


def load_method_table(candidates_csv: Path, chai_dir: Path, wt_agg: pd.DataFrame) -> pd.DataFrame:
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


def compute_success_curve(df: pd.DataFrame) -> list[float]:
    values = []
    for _, plddt_thr, ptm_thr in THRESHOLDS:
        if plddt_thr is None:
            structure_pass = pd.Series(True, index=df.index)
        else:
            structure_pass = (df["best_plddt"] > plddt_thr) & (df["best_ptm"] > ptm_thr)
        joint_success = structure_pass & df["agg_gt_wt"] & df["dg_gt_wt"]
        per_backbone = joint_success.groupby(df["pdb_name"]).mean()
        values.append(float(per_backbone.mean()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot threshold-sensitivity line chart for joint success.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=list(RUNS.keys()),
        help="Methods to include.",
    )
    parser.add_argument(
        "--raw_data_csv",
        type=Path,
        default=Path("data/rocklin/rawdata/data.csv"),
    )
    parser.add_argument(
        "--wt_agg_column",
        type=str,
        default="log2_fold_change_75_clip",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=Path("results/figures/threshold_sensitivity_lines"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_data_csv, usecols=["name", args.wt_agg_column])
    wt_agg = raw.rename(columns={"name": "pdb_name", args.wt_agg_column: "wt_proagg_score"})

    summary_rows = []
    xlabels = [name for name, _, _ in THRESHOLDS]
    method_order = args.methods
    for method in method_order:
        cfg = RUNS[method]
        df = load_method_table(cfg["candidates_csv"], cfg["chai_dir"], wt_agg)
        vals = compute_success_curve(df)
        for label, value in zip(xlabels, vals):
            summary_rows.append({"Method": method, "Threshold": label, "Joint success": value})

    summary = pd.DataFrame(summary_rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=300)
    x = np.arange(len(xlabels))
    for method in method_order:
        sub = summary[summary["Method"] == method].set_index("Threshold").loc[xlabels]
        y = sub["Joint success"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Joint success rate")
    ax.set_xlabel("Structure filtering criterion")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.12))
    fig.tight_layout()

    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {svg_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved summary CSV: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
