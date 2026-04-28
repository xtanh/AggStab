#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import add_stats_box, configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt


METHOD_ORDER = ["ProteinMPNN", "SolubleMPNN", "SFT", "DPO+SFT"]
METHOD_COLORS = {
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "SFT": "#B8D2CC",
    "DPO+SFT": "#E8B2A7",
}
METHOD_FILES = {
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "SFT": Path("results/semi_joint_sft_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
    "DPO+SFT": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
}


def load_cluster_long_df(raw_csv: Path, min_cluster_size: int) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, usecols=["name", "cluster", "split"])
    test_map = raw[raw["split"] == "test"][["name", "cluster"]].rename(columns={"name": "pdb_name"})

    frames = []
    for method, path in METHOD_FILES.items():
        df = pd.read_csv(path, usecols=["pdb_name", "proagg_score", "deltaG"])
        bb = (
            df.groupby("pdb_name", as_index=False)
            .agg(
                agg_mean_over16=("proagg_score", "mean"),
                dgunfolding_mean_over16=("deltaG", "mean"),
            )
        )
        bb = bb.merge(test_map, on="pdb_name", how="inner", validate="one_to_one")
        cluster_df = (
            bb.groupby("cluster", as_index=False)
            .agg(
                agg_cluster_mean=("agg_mean_over16", "mean"),
                dgunfolding_cluster_mean=("dgunfolding_mean_over16", "mean"),
                n_backbones=("pdb_name", "count"),
            )
        )
        cluster_df = cluster_df[cluster_df["n_backbones"] >= min_cluster_size].copy()
        cluster_df["Method"] = method
        frames.append(cluster_df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cluster-level aggregation vs stability scatter.")
    parser.add_argument("--raw_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--output_prefix", type=Path, default=Path("results/figures/cluster_level_property_scatter"))
    parser.add_argument("--min_cluster_size", type=int, default=2, help="Minimum number of test backbones per cluster.")
    args = parser.parse_args()

    df = load_cluster_long_df(args.raw_csv, args.min_cluster_size)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_long.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=300)
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method]
        ax.scatter(
            sub["agg_cluster_mean"],
            sub["dgunfolding_cluster_mean"],
            s=34,
            alpha=0.72,
            color=METHOD_COLORS[method],
            edgecolors="white",
            linewidths=0.35,
            label=f"{method} (n={len(sub)})",
        )

    ax.set_xlabel("Cluster-level mean anti-aggregation score")
    ax.set_ylabel("Cluster-level mean ΔG_unfolding")
    ax.set_title("Cluster-level property landscape on the test set")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")

    summary = (
        df.groupby("Method")[["agg_cluster_mean", "dgunfolding_cluster_mean"]]
        .mean()
        .reindex(METHOD_ORDER)
    )
    text = "\n".join(
        f"{method}: ({summary.loc[method, 'agg_cluster_mean']:.3f}, {summary.loc[method, 'dgunfolding_cluster_mean']:.3f})"
        for method in METHOD_ORDER
    )
    add_stats_box(ax, text, x=0.03, y=0.97)

    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # dominance summary relative to all other methods on a per-cluster basis
    wide = df.pivot(index="cluster", columns="Method", values=["agg_cluster_mean", "dgunfolding_cluster_mean"])
    dominance_rows = []
    for cluster in wide.index:
        row = wide.loc[cluster]
        if any(pd.isna(row[metric, method]) for metric in ["agg_cluster_mean", "dgunfolding_cluster_mean"] for method in METHOD_ORDER):
            continue
        dpo_agg = row["agg_cluster_mean", "DPO+SFT"]
        dpo_dg = row["dgunfolding_cluster_mean", "DPO+SFT"]
        dominance_rows.append(
            {
                "cluster": cluster,
                "dpo_beats_all_agg": all(dpo_agg > row["agg_cluster_mean", m] for m in METHOD_ORDER if m != "DPO+SFT"),
                "dpo_beats_all_dg": all(dpo_dg > row["dgunfolding_cluster_mean", m] for m in METHOD_ORDER if m != "DPO+SFT"),
            }
        )
    dom = pd.DataFrame(dominance_rows)
    if not dom.empty:
        dom["dpo_beats_all_both"] = dom["dpo_beats_all_agg"] & dom["dpo_beats_all_dg"]
    else:
        dom = pd.DataFrame(columns=["cluster", "dpo_beats_all_agg", "dpo_beats_all_dg", "dpo_beats_all_both"])
    dom.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_dominance.csv"), index=False)
    print(f"Saved SVG: {args.output_prefix.with_suffix('.svg')}")
    print(f"Saved PDF: {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved long CSV: {args.output_prefix.with_name(args.output_prefix.name + '_long.csv')}")
    print(f"Saved dominance CSV: {args.output_prefix.with_name(args.output_prefix.name + '_dominance.csv')}")
    if not dom.empty:
        print(dom.mean(numeric_only=True).to_string())
    else:
        print("No complete cluster overlap available for dominance summary under current filter.")


if __name__ == "__main__":
    main()
