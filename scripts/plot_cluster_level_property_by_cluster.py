#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from plot_pub_utils import configure_matplotlib, style_axis

configure_matplotlib()

import matplotlib.pyplot as plt


DEFAULT_METHOD_ORDER = ["ESM-IF","ProteinMPNN", "SolubleMPNN", "MoMPNN", "AggStab-DPO"]
METHOD_COLORS = {
    "ESM-IF": "#B8D2CC",
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "MoMPNN": "#F0C987",
    "AggStab-DPO": "#E8B2A7",
}
METHOD_FILES = {
    "ESM-IF": Path("results/esmif_fulltest_n16_joint/full_candidates.csv"),
    "ProteinMPNN": Path("results/mpnn_baseline_fulltest_n16_joint/full_candidates.csv"),
    "SolubleMPNN": Path("results/mpnn_soluble_fulltest_n16_joint/full_candidates.csv"),
    "MoMPNN": Path("results/mompnn_protsol_ig_esm_fulltest_n16_t05_joint/full_candidates.csv"),
    "AggStab-DPO": Path("results/semi_joint_dpo_sft10_thbalfull_halves_r2_e1_valselect_m05/fulltest_candidates.csv"),
}


def load_cluster_df(raw_csv: Path, min_cluster_size: int, method_order: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, usecols=["name", "cluster", "split"])
    test_map = raw[raw["split"] == "test"][["name", "cluster"]].rename(columns={"name": "pdb_name"})
    true_cluster_sizes = (
        raw[raw["split"] == "test"]
        .groupby("cluster", as_index=False)
        .agg(cluster_size=("name", "count"))
    )

    frames = []
    for method in method_order:
        path = METHOD_FILES[method]
        df = pd.read_csv(path, usecols=["pdb_name", "proagg_score", "deltaG"])
        bb = (
            df.groupby("pdb_name", as_index=False)
            .agg(
                agg_mean_over16=("proagg_score", "mean"),
                dgunfolding_mean_over16=("deltaG", "mean"),
            )
        )
        bb = bb.merge(test_map, on="pdb_name", how="inner", validate="one_to_one")
        cl = (
            bb.groupby("cluster", as_index=False)
            .agg(
                agg_cluster_mean=("agg_mean_over16", "mean"),
                dgunfolding_cluster_mean=("dgunfolding_mean_over16", "mean"),
                n_backbones=("pdb_name", "count"),
            )
        )
        cl["Method"] = method
        frames.append(cl)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.merge(true_cluster_sizes, on="cluster", how="left", validate="many_to_one")
    merged = merged[merged["cluster_size"] >= min_cluster_size].copy()
    return merged


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_prefix: Path,
    method_order: list[str],
    point_size: float,
) -> None:
    dpo_sort = (
        df[df["Method"] == "AggStab-DPO"][["cluster", metric]]
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )
    cluster_order = dpo_sort["cluster"].tolist()
    pos_map = {cluster: idx for idx, cluster in enumerate(cluster_order)}

    fig, ax = plt.subplots(figsize=(max(12, len(cluster_order) * 0.42), 5.2), dpi=300)

    for method in method_order:
        sub = df[df["Method"] == method].copy()
        sub = sub[sub["cluster"].isin(pos_map)].copy()
        sub["x"] = sub["cluster"].map(pos_map).astype(float)
        ax.scatter(
            sub["x"],
            sub[metric],
            s=point_size,
            color=METHOD_COLORS[method],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=method,
            zorder=3,
        )

    ax.set_xlabel("Test clusters (ordered by AggStab-DPO)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    xvals = list(range(len(cluster_order)))
    ax.set_xticks(xvals)
    ax.set_xticklabels(cluster_order, rotation=60, ha="right")
    style_axis(ax)
    ax.legend(frameon=False, ncol=min(4, len(method_order)), loc="upper right")

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVG: {output_prefix.with_suffix('.svg')}")
    print(f"Saved PDF: {output_prefix.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cluster-level metrics across clusters.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHOD_ORDER,
        choices=list(METHOD_FILES.keys()),
        help="Methods to plot.",
    )
    parser.add_argument("--raw_csv", type=Path, default=Path("data/rocklin/rawdata/data.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--output_stem",
        type=str,
        default="cluster_level_property_by_cluster",
        help="Stem used for long CSV output; figure filenames derive from this suffix.",
    )
    parser.add_argument(
        "--figure_suffix",
        type=str,
        default="default",
        help="Suffix appended to figure filenames, e.g. all or min10.",
    )
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument(
        "--point_size",
        type=float,
        default=42,
        help="Scatter point size for per-cluster method markers.",
    )
    args = parser.parse_args()

    method_order = args.methods
    df = load_cluster_df(args.raw_csv, args.min_cluster_size, method_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = args.output_dir / f"{args.output_stem}_long.csv"
    df.to_csv(long_csv, index=False)

    plot_metric(
        df,
        metric="agg_cluster_mean",
        ylabel="Cluster-level mean anti-aggregation score",
        title="Cluster-level anti-aggregation scores across test clusters",
        output_prefix=args.output_dir / f"cluster_level_aggregation_by_cluster_{args.figure_suffix}",
        method_order=method_order,
        point_size=args.point_size,
    )
    plot_metric(
        df,
        metric="dgunfolding_cluster_mean",
        ylabel="Cluster-level mean ΔG_unfolding",
        title="Cluster-level stability scores across test clusters",
        output_prefix=args.output_dir / f"cluster_level_dgunfolding_by_cluster_{args.figure_suffix}",
        method_order=method_order,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
