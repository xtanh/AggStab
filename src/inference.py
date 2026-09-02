"""Utilities for backbone-conditioned AggStab reward inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from transformers import EsmTokenizer

from src.config.utils import load_yaml_config
from src.ln.lightning_model import LightningProAggModel


STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def extract_backbone_context(
    pdb_path: str | Path,
    *,
    chain: str = "A",
    foldseek_bin: str | Path | None = None,
    mask_low_confidence: bool = False,
    plddt_threshold: float = 70.0,
) -> tuple[str, str]:
    """Return the amino-acid and lowercase Foldseek 3Di sequences for one chain."""
    pdb_path = Path(pdb_path).expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Backbone file not found: {pdb_path}")

    configured_bin = str(foldseek_bin or os.environ.get("FOLDSEEK_BIN", "foldseek"))
    executable = shutil.which(configured_bin)
    if executable is None:
        candidate = Path(configured_bin).expanduser()
        if candidate.is_file():
            executable = str(candidate.resolve())
        else:
            raise FileNotFoundError(
                "Foldseek was not found. Install Foldseek or set FOLDSEEK_BIN to "
                "the structureto3didescriptor-capable executable."
            )

    with tempfile.TemporaryDirectory(prefix="aggstab_foldseek_") as temp_dir:
        output_path = Path(temp_dir) / "descriptor.tsv"
        command = [
            executable,
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(pdb_path),
            str(output_path),
        ]
        subprocess.run(command, check=True)

        records: dict[str, tuple[str, str]] = {}
        with output_path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                descriptor, sequence, structure_sequence = fields[:3]
                record_chain = descriptor.split()[0].rsplit("_", 1)[-1]
                records.setdefault(record_chain, (sequence.upper(), structure_sequence.lower()))

    if chain not in records:
        available = ", ".join(sorted(records)) or "none"
        raise KeyError(f"Chain {chain!r} was not found in {pdb_path}; available chains: {available}")

    sequence, structure_sequence = records[chain]
    if len(sequence) != len(structure_sequence):
        raise ValueError(
            f"Foldseek returned unequal sequence lengths for chain {chain}: "
            f"{len(sequence)} amino acids and {len(structure_sequence)} 3Di tokens"
        )
    if mask_low_confidence:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("target", str(pdb_path))
        model = next(structure.get_models())
        if chain not in model:
            raise KeyError(f"Chain {chain!r} was not found by the PDB parser")
        residue_confidences = []
        for residue in model[chain]:
            atom_values = [atom.get_bfactor() for atom in residue.get_atoms()]
            if atom_values:
                residue_confidences.append(float(np.mean(atom_values)))
        if len(residue_confidences) != len(structure_sequence):
            raise ValueError(
                "Cannot apply pLDDT masking because parsed residue count "
                f"{len(residue_confidences)} differs from Foldseek length {len(structure_sequence)}"
            )
        tokens = list(structure_sequence)
        for index, confidence in enumerate(residue_confidences):
            if confidence < plddt_threshold:
                tokens[index] = "#"
        structure_sequence = "".join(tokens)
    return sequence, structure_sequence


def validate_sequence(sequence: str, expected_length: int) -> str:
    sequence = "".join(sequence.split()).upper()
    invalid = sorted(set(sequence) - STANDARD_AAS)
    if invalid:
        raise ValueError(f"Sequence contains unsupported residues: {''.join(invalid)}")
    if len(sequence) != expected_length:
        raise ValueError(
            f"Sequence length {len(sequence)} does not match backbone length {expected_length}"
        )
    return sequence


def load_reward_predictor(checkpoint_path: str | Path, config_path: str | Path, device: str):
    cfg = load_yaml_config(str(config_path))
    lightning_model = LightningProAggModel.load_from_checkpoint(
        str(checkpoint_path),
        cfg=cfg,
        map_location=device,
    )
    model = lightning_model.model.to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.saprot_path)
    output_key = cfg.train.get("output_key", "score")
    return model, tokenizer, output_key


def _combine_sequence_and_structure(sequence: str, structure_sequence: str) -> str:
    return "".join(aa + token for aa, token in zip(sequence, structure_sequence))


@torch.inference_mode()
def score_with_reward(
    model,
    tokenizer,
    output_key: str,
    sequences: list[str],
    structure_sequence: str,
    *,
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """Score amino-acid sequences against one fixed-backbone 3Di context."""
    scores: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        combined = [_combine_sequence_and_structure(seq, structure_sequence) for seq in batch]
        spaced = [" ".join(sa[i : i + 2] for i in range(0, len(sa), 2)) for sa in combined]
        encoded = tokenizer.batch_encode_plus(spaced, return_tensors="pt", padding=True)
        model_batch = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        output = model(model_batch)
        scores.append(output[output_key].detach().float().flatten().cpu())
    return torch.cat(scores)
