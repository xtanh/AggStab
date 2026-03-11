"""Summarize a DPO beta sweep into a single CSV.

Assumes runs were produced by `scripts/run_dpo_beta_sweep.sh` with output dirs:
  results/dpo_<run_tag_base>_beta<beta>/
and each contains:
  - dpo_eval_summary.csv
  - dpo_v2_pathology_enriched.csv
  - training_history.json (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_beta(dir_name: str) -> float | None:
    m = re.search(r"_beta([0-9.]+)$", dir_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _read_json(path: Path):
    return json.loads(path.read_text())


def _safe_quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_tag_base", type=str, required=True)
    args = p.parse_args()

    base = Path("results")
    prefix = f"dpo_{args.run_tag_base}_beta"

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if not run_dirs:
        raise SystemExit(f"No run dirs found under results/ with prefix: {prefix}")

    rows = []
    for d in run_dirs:
        beta = _parse_beta(d.name)
        if beta is None:
            continue

        eval_csv = d / "dpo_eval_summary.csv"
        path_csv = d / "dpo_v2_pathology_enriched.csv"
        hist_json = d / "training_history.json"

        if not eval_csv.exists():
            continue

        df = pd.read_csv(eval_csv)
        n = len(df)

        delta_mean = float(df["delta_mean"].mean())
        delta_max = float(df["delta_max"].mean())
        improve_mean = float((df["delta_mean"] > 0).mean())
        improve_max = float((df["delta_max"] > 0).mean())
        delta_logp = float((df["logprob_dpo"] - df["logprob_orig"]).mean())
        delta_div = float((df["diversity_dpo"] - df["diversity_orig"]).mean())

        has_run = float("nan")
        top_p90 = float("nan")
        charged_p90 = float("nan")
        hydro_p10 = float("nan")
        if path_csv.exists():
            pdf = pd.read_csv(path_csv)
            has_run = float(pdf["has_run_dpo"].mean())
            top_p90 = _safe_quantile(pdf["top_frac_dpo"].to_numpy(), 0.9)
            charged_p90 = _safe_quantile(pdf["charged_frac_dpo"].to_numpy(), 0.9)
            hydro_p10 = _safe_quantile(pdf["hydrophobic_frac_dpo"].to_numpy(), 0.1)

        last_val_margin = float("nan")
        last_val_loss = float("nan")
        if hist_json.exists():
            hist = _read_json(hist_json)
            if isinstance(hist, list) and hist:
                last = hist[-1]
                last_val_margin = float(last.get("val_reward_margin", float("nan")))
                last_val_loss = float(last.get("val_loss", float("nan")))

        rows.append(
            {
                "run_dir": str(d),
                "beta": beta,
                "n_test": n,
                "delta_mean": delta_mean,
                "delta_max": delta_max,
                "improve_mean": improve_mean,
                "improve_max": improve_max,
                "delta_logprob": delta_logp,
                "delta_diversity": delta_div,
                "has_run_rate": has_run,
                "top_frac_p90": top_p90,
                "charged_frac_p90": charged_p90,
                "hydrophobic_frac_p10": hydro_p10,
                "last_val_reward_margin": last_val_margin,
                "last_val_loss": last_val_loss,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("beta")
    out_path = base / f"dpo_{args.run_tag_base}_beta_sweep_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved sweep summary: {out_path}")


if __name__ == "__main__":
    main()

