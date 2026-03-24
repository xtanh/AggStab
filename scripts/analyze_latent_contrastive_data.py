import argparse
import os

import numpy as np
import pandas as pd


TARGET_COL = "log2_fold_change_75_clip"


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(path)
    df["seq_len"] = df["protein_sequence"].str.len()
    return df


def summarize_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    s = df[TARGET_COL]
    thresholds = [-3.5, -3.0, -2.5, -2.0, -1.0, 0.0]
    row = {
        "split": split,
        "n": len(df),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "q05": s.quantile(0.05),
        "q10": s.quantile(0.10),
        "q25": s.quantile(0.25),
        "median": s.quantile(0.50),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
        "q95": s.quantile(0.95),
        "max": s.max(),
    }
    for t in thresholds:
        row[f"pct<= {t}"] = (s <= t).mean()
    return pd.DataFrame([row])


def pair_threshold_table(values: np.ndarray, split: str, pos_thresholds, neg_thresholds) -> pd.DataFrame:
    diff = np.abs(values[:, None] - values[None, :])
    triu = np.triu_indices(len(values), k=1)
    pair_diff = diff[triu]

    rows = []
    for t in pos_thresholds:
        rows.append({
            "split": split,
            "type": "positive",
            "threshold": t,
            "pair_fraction": float((pair_diff < t).mean()),
            "pair_count": int((pair_diff < t).sum()),
        })
    for t in neg_thresholds:
        rows.append({
            "split": split,
            "type": "negative",
            "threshold": t,
            "pair_fraction": float((pair_diff > t).mean()),
            "pair_count": int((pair_diff > t).sum()),
        })
    return pd.DataFrame(rows)


def error_bucket_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"target", "prediction", "abs_error", "signed_error"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    edges = [-np.inf, -3.5, -3.0, -2.0, -1.0, 0.0, np.inf]
    labels = ["<=-3.5", "(-3.5,-3.0]", "(-3.0,-2.0]", "(-2.0,-1.0]", "(-1.0,0.0)", ">=0.0"]
    tmp = df.copy()
    tmp["target_bin"] = pd.cut(tmp["target"], bins=edges, labels=labels, right=True)

    out = (
        tmp.groupby("target_bin", observed=False)
        .agg(
            n=("target", "size"),
            target_mean=("target", "mean"),
            pred_mean=("prediction", "mean"),
            mae=("abs_error", "mean"),
            bias=("signed_error", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "split", split)
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze split distributions and error cases for latent-contrastive experiments.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_pred_csv", default=None)
    parser.add_argument("--valid_pred_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    split_frames = {split: load_split(args.data_dir, split) for split in ["train", "valid", "test"]}

    dist_df = pd.concat([summarize_split(df, split) for split, df in split_frames.items()], ignore_index=True)
    dist_path = os.path.join(output_dir, "split_distribution_summary.csv")
    dist_df.to_csv(dist_path, index=False)

    pair_df = pd.concat(
        [
            pair_threshold_table(
                df[TARGET_COL].to_numpy(dtype=np.float32),
                split,
                pos_thresholds=[0.25, 0.5, 0.75, 1.0],
                neg_thresholds=[1.5, 2.0, 2.5, 3.0],
            )
            for split, df in split_frames.items()
            if split in {"train", "valid"}
        ],
        ignore_index=True,
    )
    pair_path = os.path.join(output_dir, "pair_threshold_summary.csv")
    pair_df.to_csv(pair_path, index=False)

    error_frames = []
    for split, pred_path in [("train", args.train_pred_csv), ("valid", args.valid_pred_csv)]:
        if pred_path:
            pred_df = pd.read_csv(pred_path)
            error_frames.append(error_bucket_table(pred_df, split))
    if error_frames:
        error_df = pd.concat(error_frames, ignore_index=True)
        error_path = os.path.join(output_dir, "error_bucket_summary.csv")
        error_df.to_csv(error_path, index=False)
    else:
        error_path = None

    print("Saved:")
    print(dist_path)
    print(pair_path)
    if error_path:
        print(error_path)

    print("\nSplit distributions:")
    print(dist_df.to_string(index=False))
    print("\nPair threshold summary:")
    print(pair_df.to_string(index=False))
    if error_frames:
        print("\nError bucket summary:")
        print(error_df.to_string(index=False))


if __name__ == "__main__":
    main()
