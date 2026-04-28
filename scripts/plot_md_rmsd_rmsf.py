#!/usr/bin/env python
"""Plot MD RMSD and RMSF curves for WT and designed sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


PALETTE = {
    "WT": "#E8B2A7",
    "ProteinMPNN": "#B5AED5",
    "SolubleMPNN": "#B2E6FD",
    "SFT": "#B8D2CC",
    "DPO+SFT": "#D99C8E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md_dir", type=Path, required=True, help="Directory containing *_trajectory.dcd and *_equilibrated.pdb.")
    parser.add_argument("--output_prefix", type=Path, required=True, help="Prefix for output figure files.")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride for loading trajectories.")
    parser.add_argument("--atom_selection", type=str, default="name CA and protein", help="Atom selection for RMSD/RMSF.")
    parser.add_argument("--time_per_frame_ps", type=float, default=10.0, help="Physical time between stored frames in ps.")
    return parser.parse_args()


def build_runs(md_dir: Path) -> list[tuple[str, str]]:
    return [
        ("WT", "wt"),
        ("ProteinMPNN", "proteinmpnn"),
        ("SolubleMPNN", "solublempnn"),
        ("SFT", "sft"),
        ("DPO+SFT", "dpo_sft"),
    ]


def load_metrics(dcd_path: Path, top_path: Path, atom_selection: str, time_per_frame_ps: float, stride: int) -> dict:
    traj = md.load_dcd(str(dcd_path), top=str(top_path), stride=stride)
    atom_indices = traj.topology.select(atom_selection)
    if len(atom_indices) == 0:
        raise ValueError(f"No atoms matched selection '{atom_selection}' for {dcd_path}")

    traj.superpose(traj, 0, atom_indices=atom_indices)
    rmsd = md.rmsd(traj, traj, 0, atom_indices=atom_indices) * 10.0
    rmsf = md.rmsf(traj, traj, 0, atom_indices=atom_indices) * 10.0

    time_ns = (np.arange(traj.n_frames) * time_per_frame_ps * stride) / 1000.0
    residue_ids = [traj.topology.atom(atom_index).residue.resSeq for atom_index in atom_indices]

    return {
        "time_ns": time_ns,
        "rmsd": rmsd,
        "residue_ids": residue_ids,
        "rmsf": rmsf,
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.4)
    axis.spines["bottom"].set_linewidth(1.4)
    axis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    axis.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    runs = build_runs(args.md_dir)

    metrics: dict[str, dict] = {}
    for label, prefix in runs:
        dcd_path = args.md_dir / f"{prefix}_trajectory.dcd"
        top_path = args.md_dir / f"{prefix}_equilibrated.pdb"
        if not dcd_path.exists():
            raise FileNotFoundError(f"Missing trajectory: {dcd_path}")
        if not top_path.exists():
            raise FileNotFoundError(f"Missing topology: {top_path}")
        metrics[label] = load_metrics(
            dcd_path=dcd_path,
            top_path=top_path,
            atom_selection=args.atom_selection,
            time_per_frame_ps=args.time_per_frame_ps,
            stride=args.stride,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    rmsd_ax, rmsf_ax = axes
    for label, _ in runs:
        rmsd_ax.plot(
            metrics[label]["time_ns"],
            metrics[label]["rmsd"],
            color=PALETTE[label],
            linewidth=1.6,
            label=label,
            alpha=0.95,
        )
        rmsf_ax.plot(
            metrics[label]["residue_ids"],
            metrics[label]["rmsf"],
            color=PALETTE[label],
            linewidth=1.6,
            label=label,
            alpha=0.95,
        )

    rmsd_ax.set_xlabel("Time (ns)", fontsize=12)
    rmsd_ax.set_ylabel("Cα RMSD (Å)", fontsize=12)
    rmsd_ax.set_title("MD RMSD", fontsize=13)
    style_axis(rmsd_ax)

    rmsf_ax.set_xlabel("Residue index", fontsize=12)
    rmsf_ax.set_ylabel("Cα RMSF (Å)", fontsize=12)
    rmsf_ax.set_title("MD RMSF", fontsize=13)
    style_axis(rmsf_ax)

    handles, labels = rmsd_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(f"{args.output_prefix}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output_prefix}.pdf")
    print(f"Saved: {args.output_prefix}.svg")
    print(f"Saved: {args.output_prefix}.png")


if __name__ == "__main__":
    main()
