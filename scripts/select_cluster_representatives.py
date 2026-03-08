"""从每个 cluster 中随机挑选 1 个蛋白质作为代表，并将对应的 PDB 文件
软链接（symlink）到输出目录，按 train/valid/test 三个子目录分别存放。

Usage:
    python scripts/select_cluster_representatives.py \
        --data_csv data/rocklin/rawdata/data.csv \
        --pdb_dir  data/rocklin/all_AF_rank0_pdbs \
        --output_dir data/dpo/representative_pdbs \
        --seed 42
"""

import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Select one representative PDB per cluster for DPO pipeline."
    )
    parser.add_argument(
        "--data_csv", type=str,
        default="data/rocklin/rawdata/data.csv",
        help="Path to the raw data CSV with columns: name, cluster, split",
    )
    parser.add_argument(
        "--pdb_dir", type=str,
        default="data/rocklin/all_AF_rank0_pdbs",
        help="Directory containing all PDB files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/dpo/representative_pdbs",
        help="Output root directory (will contain train/valid/test subdirs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    required_cols = {"name", "cluster", "split"}
    assert required_cols.issubset(df.columns), (
        f"CSV must contain columns {required_cols}, got {set(df.columns)}"
    )

    pdb_set = set(os.listdir(args.pdb_dir))

    df["pdb_file"] = df["name"] + "_ranked_0.pdb"
    df["has_pdb"] = df["pdb_file"].isin(pdb_set)

    n_missing = (~df["has_pdb"]).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} proteins have no matching PDB, will be skipped:")
        print(df.loc[~df["has_pdb"], "name"].tolist())

    df = df[df["has_pdb"]].copy()

    representatives = (
        df.groupby(["split", "cluster"])
        .sample(n=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    summary = {}
    for split in ["train", "valid", "test"]:
        split_df = representatives[representatives["split"] == split]
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        count = 0
        for _, row in split_df.iterrows():
            src = os.path.abspath(os.path.join(args.pdb_dir, row["pdb_file"]))
            dst = os.path.join(split_dir, row["pdb_file"])
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            count += 1

        summary[split] = count
        print(f"[{split}] {count} representative PDBs -> {split_dir}")

    out_csv = os.path.join(args.output_dir, "representatives.csv")
    representatives.to_csv(out_csv, index=False)
    print(f"\nRepresentative list saved to {out_csv}")
    print(f"Total: {len(representatives)} representatives from {df['cluster'].nunique()} clusters")


if __name__ == "__main__":
    main()
