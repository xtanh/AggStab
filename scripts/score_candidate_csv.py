#!/usr/bin/env python
"""Score candidate sequences from a CSV with aggregation/stability predictors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from transformers import EsmTokenizer

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index("scripts")]
sys.path.insert(0, PROJ_DIR)

from src.config.utils import load_yaml_config
from src.dpo.sample_and_score import build_struct_token_lookup, score_sequences_with_proagg
from src.dpo.sample_and_score_dual import load_stability_lookup, score_sequences_with_predictor
from src.ln.lightning_model import LightningProAggModel


def load_predictor(checkpoint_path: str, config_path: str, device: str):
    cfg = load_yaml_config(config_path)
    lightning = LightningProAggModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model = lightning.model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)
    output_key = cfg.train.get("output_key", "score")
    return model, tokenizer, output_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--proagg_ckpt", type=str, required=True)
    parser.add_argument("--proagg_config", type=str, default="configs/proagg_final_candidate.yaml")
    parser.add_argument("--stab_ckpt", type=str, default=None)
    parser.add_argument("--stab_config", type=str, default=None)
    parser.add_argument("--stability_csv", type=str, default="data/rocklin/Metagenomic_dG.csv")
    parser.add_argument("--data_csv", type=str, default="data/rocklin/rawdata/data.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    struct_lookup = build_struct_token_lookup(args.data_csv)

    agg_cfg = load_yaml_config(args.proagg_config)
    agg_lightning = LightningProAggModel.load_from_checkpoint(args.proagg_ckpt, cfg=agg_cfg)
    agg_model = agg_lightning.model.to(args.device)
    agg_model.eval()
    agg_tokenizer = EsmTokenizer.from_pretrained(agg_cfg.model.saprot_path)

    stab_model = None
    stab_tokenizer = None
    stab_output_key = "deltaG"
    wt_delta_g_lookup = None
    if args.stab_ckpt and args.stab_config:
        stab_model, stab_tokenizer, stab_output_key = load_predictor(
            args.stab_ckpt,
            args.stab_config,
            args.device,
        )
        wt_delta_g_lookup = load_stability_lookup(args.stability_csv)

    rows: list[dict] = []
    for pdb_name, group in df.groupby("pdb_name", sort=True):
        print(f"Scoring {pdb_name} ({len(group)} candidates)", flush=True)
        if pdb_name not in struct_lookup:
            print(f"  missing structural tokens for {pdb_name}, skip", flush=True)
            continue

        struct_tokens = struct_lookup[pdb_name]
        sequences = group["sequence"].astype(str).tolist()
        agg_scores = score_sequences_with_proagg(
            agg_model,
            sequences,
            struct_tokens,
            agg_tokenizer,
            device=args.device,
            batch_size=args.batch_size,
        )

        if stab_model is not None:
            stab_scores = score_sequences_with_predictor(
                stab_model,
                stab_output_key,
                sequences,
                struct_tokens,
                stab_tokenizer,
                device=args.device,
                batch_size=args.batch_size,
            )
            wt_delta_g = wt_delta_g_lookup.get(pdb_name) if wt_delta_g_lookup is not None else None
        else:
            stab_scores = None
            wt_delta_g = None

        for local_idx, row in enumerate(group.itertuples(index=False)):
            delta_g = float(stab_scores[local_idx].item()) if stab_scores is not None else None
            out_row = row._asdict()
            out_row["proagg_score"] = float(agg_scores[local_idx].item())
            out_row["deltaG"] = delta_g
            out_row["wt_deltaG"] = wt_delta_g
            out_row["deltaG_minus_wt"] = (
                delta_g - wt_delta_g if (delta_g is not None and wt_delta_g is not None) else None
            )
            rows.append(out_row)

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved scored candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
