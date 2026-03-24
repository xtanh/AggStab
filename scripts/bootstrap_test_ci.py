import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def bootstrap_metric(y_true, y_pred, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if metric == "spearman":
            stats[i] = spearmanr(yp, yt).statistic
        elif metric == "pearson":
            stats[i] = pearsonr(yp, yt).statistic
        else:
            raise ValueError(metric)
    mean = float(np.mean(stats))
    lower = float(np.quantile(stats, 0.025))
    upper = float(np.quantile(stats, 0.975))
    half_ci = (upper - lower) / 2.0
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "half_ci": half_ci,
        "samples": stats,
    }


def bootstrap_delta(y_true, y_pred_a, y_pred_b, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ya = y_pred_a[idx]
        yb = y_pred_b[idx]
        if metric == "spearman":
            a = spearmanr(ya, yt).statistic
            b = spearmanr(yb, yt).statistic
        elif metric == "pearson":
            a = pearsonr(ya, yt).statistic
            b = pearsonr(yb, yt).statistic
        else:
            raise ValueError(metric)
        deltas[i] = b - a
    mean = float(np.mean(deltas))
    lower = float(np.quantile(deltas, 0.025))
    upper = float(np.quantile(deltas, 0.975))
    half_ci = (upper - lower) / 2.0
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "half_ci": half_ci,
    }


def load_preds(path):
    df = pd.read_csv(path)
    return df["target"].to_numpy(), df["prediction"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Bootstrap test-set CI for prediction CSVs.")
    parser.add_argument("--csv", nargs="+", required=True, help="Prediction CSV paths")
    parser.add_argument("--labels", nargs="+", help="Optional labels")
    parser.add_argument("--metric", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csvs = [Path(p) for p in args.csv]
    labels = args.labels or [p.parent.parent.name for p in csvs]
    if len(labels) != len(csvs):
        raise ValueError("labels length must match csv length")

    loaded = []
    base_target = None
    for label, path in zip(labels, csvs):
        y_true, y_pred = load_preds(path)
        if base_target is None:
            base_target = y_true
        elif not np.allclose(base_target, y_true):
            raise ValueError(f"Target mismatch for {label}: {path}")
        result = bootstrap_metric(base_target, y_pred, args.metric, args.n_boot, args.seed)
        loaded.append((label, path, y_pred, result))

    print(f"Bootstrap metric: {args.metric}, n_boot={args.n_boot}")
    for label, path, _, result in loaded:
        print(
            f"{label}: {result['mean']:.6f} ± {result['half_ci']:.6f} "
            f"(95% CI [{result['lower']:.6f}, {result['upper']:.6f}]) "
            f"[{path}]"
        )

    if len(loaded) >= 2:
        ref_label, _, ref_pred, _ = loaded[0]
        for label, _, pred, _ in loaded[1:]:
            delta = bootstrap_delta(base_target, ref_pred, pred, args.metric, args.n_boot, args.seed)
            print(
                f"delta {label} - {ref_label}: {delta['mean']:.6f} ± {delta['half_ci']:.6f} "
                f"(95% CI [{delta['lower']:.6f}, {delta['upper']:.6f}])"
            )


if __name__ == "__main__":
    main()
