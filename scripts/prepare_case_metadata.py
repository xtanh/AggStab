#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from transformers import EsmTokenizer

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parent
sys.path.insert(0, str(PROJ_DIR))
sys.path.insert(0, "/home/xy_th/SaProt")

from utils.foldseek_util import get_struc_seq

from src.config.utils import load_yaml_config
from src.dpo.sample_and_score import score_sequences_with_proagg
from src.dpo.sample_and_score_dual import score_sequences_with_predictor
from src.ln.lightning_model import LightningProAggModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare case-study metadata and WT predictor scores for a single backbone.")
    parser.add_argument("--pdb_path", type=Path, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--foldseek_path", type=Path, default=Path("/home/xy_th/SaProt/bin/foldseek"))
    parser.add_argument("--proagg_ckpt", type=Path, required=True)
    parser.add_argument("--proagg_config", type=Path, default=Path("configs/proagg_final_candidate.yaml"))
    parser.add_argument("--stab_ckpt", type=Path, required=True)
    parser.add_argument("--stab_config", type=Path, default=Path("configs/proagg_deltaG_only.yaml"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--copy_ranked_pdb", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed = get_struc_seq(str(args.foldseek_path), str(args.pdb_path), [args.chain], plddt_mask=False)
    if args.chain not in parsed:
        raise KeyError(f"Chain {args.chain} not found in {args.pdb_path}")
    seq, foldseek_seq, combined_seq = parsed[args.chain]

    agg_cfg = load_yaml_config(str(args.proagg_config))
    agg_lightning = LightningProAggModel.load_from_checkpoint(str(args.proagg_ckpt), cfg=agg_cfg)
    agg_model = agg_lightning.model.to(args.device)
    agg_model.eval()
    agg_tokenizer = EsmTokenizer.from_pretrained(agg_cfg.model.saprot_path)

    stab_cfg = load_yaml_config(str(args.stab_config))
    stab_lightning = LightningProAggModel.load_from_checkpoint(str(args.stab_ckpt), cfg=stab_cfg)
    stab_model = stab_lightning.model.to(args.device)
    stab_model.eval()
    stab_tokenizer = EsmTokenizer.from_pretrained(stab_cfg.model.saprot_path)
    stab_output_key = stab_cfg.train.get("output_key", "deltaG")

    wt_proagg = float(
        score_sequences_with_proagg(
            agg_model,
            [seq],
            foldseek_seq,
            agg_tokenizer,
            device=args.device,
            batch_size=1,
        )[0].item()
    )
    wt_delta_g = float(
        score_sequences_with_predictor(
            stab_model,
            stab_output_key,
            [seq],
            foldseek_seq,
            stab_tokenizer,
            device=args.device,
            batch_size=1,
        )[0].item()
    )

    metadata = pd.DataFrame(
        [
            {
                "name": args.case_name,
                "chain": args.chain,
                "pdb_path": str(args.pdb_path.resolve()),
                "protein_sequence": seq,
                "foldseek_sequence": foldseek_seq,
                "combined_sequence": combined_seq,
                "sa_sequence_foldseek": combined_seq,
                "length": len(seq),
                "log2_fold_change_75_clip": wt_proagg,
            }
        ]
    )
    wt_delta = pd.DataFrame([{"name": args.case_name, "deltaG": wt_delta_g}])
    wt_scores = pd.DataFrame(
        [
            {
                "name": args.case_name,
                "wt_proagg_score": wt_proagg,
                "wt_deltaG": wt_delta_g,
            }
        ]
    )

    metadata_path = args.output_dir / f"{args.case_name}_metadata.csv"
    wt_delta_path = args.output_dir / f"{args.case_name}_wt_deltaG.csv"
    wt_scores_path = args.output_dir / f"{args.case_name}_wt_scores.csv"
    metadata.to_csv(metadata_path, index=False)
    wt_delta.to_csv(wt_delta_path, index=False)
    wt_scores.to_csv(wt_scores_path, index=False)

    if args.copy_ranked_pdb:
        ranked_pdb_path = args.output_dir / f"{args.case_name}_ranked_0.pdb"
        shutil.copyfile(args.pdb_path, ranked_pdb_path)
        print(f"Saved ranked-style PDB: {ranked_pdb_path}")

    print(f"Saved metadata CSV: {metadata_path}")
    print(f"Saved WT deltaG CSV: {wt_delta_path}")
    print(f"Saved WT scores CSV: {wt_scores_path}")
    print(f"WT proagg_score: {wt_proagg:.6f}")
    print(f"WT deltaG: {wt_delta_g:.6f}")
    print(f"Sequence length: {len(seq)}")


if __name__ == "__main__":
    main()
