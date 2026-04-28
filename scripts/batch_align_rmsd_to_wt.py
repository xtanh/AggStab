#!/usr/bin/env python
"""Batch-align designed structures to WT backbones and export PyMOL RMSD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_dir",
        type=Path,
        action="append",
        default=[],
        help="One or more Chai result directories; all summary_shard*.csv files under each directory will be loaded.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        action="append",
        default=[],
        help="One or more Chai summary CSV files containing best_cif and pdb_name columns.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        action="append",
        default=None,
        help="Optional method label for each summary CSV; must match the number of --summary_csv entries.",
    )
    parser.add_argument(
        "--wt_dir",
        type=Path,
        default=Path("data/rocklin/all_AF_rank0_pdbs"),
        help="Directory containing WT/reference PDB files named <pdb_name>_ranked_0.pdb.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV for per-candidate RMSD results.",
    )
    parser.add_argument(
        "--atom_selection",
        type=str,
        default="name CA",
        help='PyMOL atom selection used for alignment and RMSD, e.g. "name CA" or "backbone".',
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    args = parser.parse_args()
    if not args.summary_dir and not args.summary_csv:
        parser.error("at least one of --summary_dir or --summary_csv is required")
    return args


def expand_summary_inputs(summary_dirs: Iterable[Path], summary_csvs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for summary_dir in summary_dirs:
        shard_paths = sorted(summary_dir.glob("summary_shard*.csv"))
        if not shard_paths:
            raise FileNotFoundError(f"No summary_shard*.csv found in {summary_dir}")
        paths.extend(shard_paths)
    paths.extend(Path(path) for path in summary_csvs)
    if not paths:
        raise ValueError("No summary CSVs resolved from inputs")
    return paths


def load_tables(summary_csvs: Iterable[Path], method_names: list[str] | None) -> list[dict]:
    summary_csvs = list(summary_csvs)
    if method_names is not None and len(method_names) not in {0, len(summary_csvs)}:
        raise ValueError("--method_name count must match the resolved summary CSV count")

    rows: list[dict] = []
    for index, summary_csv in enumerate(summary_csvs):
        with summary_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = {"pdb_name", "best_cif"} - set(fieldnames)
            if missing:
                raise ValueError(f"{summary_csv} missing required columns: {sorted(missing)}")
            method = method_names[index] if method_names else summary_csv.parent.parent.name
            for row in reader:
                row["method"] = method
                row["summary_csv"] = str(summary_csv)
                rows.append(row)
    return rows


def resolve_wt_path(wt_dir: Path, pdb_name: str) -> Path:
    wt_path = wt_dir / f"{pdb_name}_ranked_0.pdb"
    if not wt_path.exists():
        raise FileNotFoundError(f"WT structure not found for {pdb_name}: {wt_path}")
    return wt_path


def compute_rmsd(rows: list[dict], wt_dir: Path, atom_selection: str) -> list[dict]:
    import pymol2

    records: list[dict] = []
    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        for row in rows:
            wt_path = resolve_wt_path(wt_dir, row["pdb_name"])
            cmd.reinitialize()
            cmd.load(str(wt_path), "wt")
            cmd.load(str(Path(row["best_cif"])), "design")

            mobile_sel = f"design and ({atom_selection})"
            target_sel = f"wt and ({atom_selection})"

            align_stats = cmd.align(mobile_sel, target_sel)
            rmsd_after = float(align_stats[0])
            aligned_atoms = int(align_stats[1])
            cycles = int(align_stats[2])
            rmsd_before = float(align_stats[3])
            atoms_before = int(align_stats[4])
            score = float(align_stats[5])
            residues_aligned = int(align_stats[6])

            records.append(
                {
                    "method": row["method"],
                    "summary_csv": row["summary_csv"],
                    "candidate_id": row.get("candidate_id"),
                    "pdb_name": row["pdb_name"],
                    "topk_rank": row.get("topk_rank"),
                    "candidate_rank": row.get("candidate_rank"),
                    "best_cif": row["best_cif"],
                    "wt_pdb": str(wt_path),
                    "atom_selection": atom_selection,
                    "rmsd_aligned": rmsd_after,
                    "rmsd_before_refinement": rmsd_before,
                    "aligned_atoms": aligned_atoms,
                    "atoms_before_refinement": atoms_before,
                    "alignment_score": score,
                    "alignment_cycles": cycles,
                    "aligned_residues": residues_aligned,
                    "best_plddt": row.get("best_plddt"),
                    "best_ptm": row.get("best_ptm"),
                    "proagg_score": row.get("proagg_score"),
                    "deltaG": row.get("deltaG"),
                    "deltaG_unfolding": row.get("deltaG_unfolding"),
                    "deltaG_minus_wt": row.get("deltaG_minus_wt"),
                }
            )
    return records


def summarize(records: list[dict]) -> str:
    grouped: dict[str, list[float]] = {}
    for row in records:
        grouped.setdefault(row["method"], []).append(float(row["rmsd_aligned"]))

    lines = ["method,count,mean,median,std,min,max"]
    for method, values in grouped.items():
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(
            ",".join(
                [
                    method,
                    str(len(values)),
                    f"{statistics.mean(values):.6f}",
                    f"{statistics.median(values):.6f}",
                    f"{std:.6f}",
                    f"{min(values):.6f}",
                    f"{max(values):.6f}",
                ]
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_csvs = expand_summary_inputs(args.summary_dir, args.summary_csv)
    method_names: list[str] | None = None
    if args.method_name:
        if args.summary_dir and not args.summary_csv and len(args.method_name) == len(args.summary_dir):
            expanded: list[str] = []
            for method_name, summary_dir in zip(args.method_name, args.summary_dir):
                shard_count = len(sorted(summary_dir.glob("summary_shard*.csv")))
                expanded.extend([method_name] * shard_count)
            method_names = expanded
        else:
            method_names = args.method_name
    rows = load_tables(summary_csvs, method_names)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    result = compute_rmsd(rows, wt_dir=args.wt_dir, atom_selection=args.atom_selection)
    write_csv(args.output_csv, result)
    print(summarize(result))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
