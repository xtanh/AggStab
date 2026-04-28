#!/usr/bin/env python
"""Sample candidate sequences from ESM-IF for a set of backbone PDBs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from esm.inverse_folding.util import load_coords
from esm.inverse_folding.util import CoordBatchConverter
import esm
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@torch.no_grad()
def sample_sequence_on_device(
    model,
    coords,
    *,
    temperature: float,
    device: torch.device,
    partial_seq: str | None = None,
    confidence=None,
) -> str:
    length = len(coords)
    batch_converter = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = batch_converter([(coords, confidence, None)])
    batch_coords = batch_coords.to(device)
    padding_mask = padding_mask.to(device)
    confidence = confidence.to(device)

    mask_idx = model.decoder.dictionary.get_idx("<mask>")
    sampled_tokens = torch.full((1, 1 + length), mask_idx, dtype=torch.long, device=device)
    sampled_tokens[0, 0] = model.decoder.dictionary.get_idx("<cath>")
    if partial_seq is not None:
        for index, token in enumerate(partial_seq):
            sampled_tokens[0, index + 1] = model.decoder.dictionary.get_idx(token)

    incremental_state = {}
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)

    for index in range(1, length + 1):
        if sampled_tokens[0, index] != mask_idx:
            continue
        logits, _ = model.decoder(
            sampled_tokens[:, :index],
            encoder_out,
            incremental_state=incremental_state,
        )
        logits = logits[0].transpose(0, 1)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        sampled_tokens[:, index] = torch.multinomial(probs, 1).squeeze(-1)

    sampled_seq = sampled_tokens[0, 1:].detach().cpu().tolist()
    return "".join(model.decoder.dictionary.get_tok(token) for token in sampled_seq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chain", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_pdbs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    device = torch.device(args.device)
    model = model.eval().to(device)

    pdb_files = sorted(glob.glob(str(args.pdb_dir / "*.pdb")))
    if args.max_pdbs > 0:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(pdb_files), min(args.max_pdbs, len(pdb_files)), replace=False)
        pdb_files = [pdb_files[index] for index in sorted(selected)]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pdb_name",
        "candidate_rank",
        "sequence",
        "num_unique_for_backbone",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for pdb_idx, pdb_path in enumerate(pdb_files):
            protein_name = os.path.basename(pdb_path).replace("_ranked_0.pdb", "")
            print(f"[{pdb_idx + 1}/{len(pdb_files)}] {protein_name}", flush=True)
            coords, _ = load_coords(pdb_path, args.chain)

            samples: list[str] = []
            for sample_idx in range(args.num_samples):
                set_seed(args.seed + pdb_idx * 1000 + sample_idx)
                sequence = sample_sequence_on_device(
                    model,
                    coords,
                    temperature=args.temperature,
                    device=device,
                )
                samples.append(sequence)

            unique_sequences = unique_preserve_order(samples)
            print(
                f"  sampled={len(samples)} unique={len(unique_sequences)} temp={args.temperature}",
                flush=True,
            )
            for rank, sequence in enumerate(unique_sequences, start=1):
                writer.writerow(
                    {
                        "pdb_name": protein_name,
                        "candidate_rank": rank,
                        "sequence": sequence,
                        "num_unique_for_backbone": len(unique_sequences),
                    }
                )

    print(f"Saved candidate table to {args.output_csv}")


if __name__ == "__main__":
    main()
