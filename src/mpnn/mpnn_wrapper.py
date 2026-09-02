"""Thin wrapper around ProteinMPNN for loading, sampling, and log-prob computation."""

import os
import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_DEFAULT_MPNN_DIR = Path(__file__).resolve().parents[3] / "ProteinMPNN"
MPNN_DIR = Path(os.environ.get("PROTEINMPNN_DIR", _DEFAULT_MPNN_DIR)).expanduser().resolve()
if not (MPNN_DIR / "protein_mpnn_utils.py").is_file():
    raise ImportError(
        "ProteinMPNN was not found. Clone the ProteinMPNN repository and set "
        "PROTEINMPNN_DIR to its root directory."
    )
sys.path.insert(0, str(MPNN_DIR))

from protein_mpnn_utils import (
    ProteinMPNN,
    parse_PDB,
    tied_featurize,
    _S_to_seq,
    _scores,
)

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
protein_mpnn_scores = _scores


def load_mpnn_model(
    checkpoint_path=None,
    device="cuda",
    ca_only=False,
    backbone_noise=0.0,
):
    """Load a pretrained ProteinMPNN model."""
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "PROTEINMPNN_CKPT",
            str(MPNN_DIR / "vanilla_model_weights" / "v_48_020.pt"),
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_edges = checkpoint["num_edges"]

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=backbone_noise,
        k_neighbors=num_edges,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def featurize_pdb(pdb_path, device="cuda", ca_only=False):
    """Parse a PDB file and return featurized tensors for ProteinMPNN."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)

    chain_id_dict = None
    fixed_positions_dict = None
    omit_AA_dict = None
    tied_positions_dict = None
    pssm_dict = None
    bias_by_res_dict = None

    batch = [pdb_dict_list[0]]
    (
        X, S, mask, lengths, chain_M, chain_encoding_all,
        letter_list_list, visible_list_list, masked_list_list,
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
        residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
        pssm_coef_all, pssm_bias_all, pssm_log_odds_all,
        bias_by_res_all, tied_beta,
    ) = tied_featurize(
        batch, device, chain_id_dict, fixed_positions_dict,
        omit_AA_dict, tied_positions_dict, pssm_dict,
        bias_by_res_dict, ca_only=ca_only,
    )

    return {
        "X": X,
        "S": S,
        "mask": mask,
        "lengths": lengths,
        "chain_M": chain_M,
        "chain_encoding_all": chain_encoding_all,
        "chain_M_pos": chain_M_pos,
        "omit_AA_mask": omit_AA_mask,
        "residue_idx": residue_idx,
        "bias_by_res": bias_by_res_all,
        "pdb_name": pdb_dict_list[0]["name"],
    }


@torch.no_grad()
def sample_sequences(model, feat, num_samples=64, temperature=0.1, device="cuda"):
    """Sample multiple sequences from ProteinMPNN for a given backbone.

    Returns list of (sequence_str, probs) tuples.
    """
    X = feat["X"]
    S = feat["S"]
    mask = feat["mask"]
    chain_M = feat["chain_M"]
    chain_M_pos = feat["chain_M_pos"]
    chain_encoding_all = feat["chain_encoding_all"]
    residue_idx = feat["residue_idx"]
    bias_by_res = feat["bias_by_res"]

    omit_AAs_np = np.zeros(21)
    bias_AAs_np = np.zeros(21)

    results = []
    for _ in range(num_samples):
        randn = torch.randn(chain_M.shape, device=device)
        sample_dict = model.sample(
            X, randn, S, chain_M, chain_encoding_all, residue_idx,
            mask=mask, temperature=temperature,
            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
            chain_M_pos=chain_M_pos, omit_AA_mask=feat["omit_AA_mask"],
            pssm_coef=None, pssm_bias=None, pssm_multi=None,
            pssm_log_odds_flag=None, pssm_log_odds_mask=None,
            pssm_bias_flag=None, bias_by_res=bias_by_res,
        )

        S_sample = sample_dict["S"]
        seq_str = _S_to_seq(S_sample[0], chain_M[0])
        results.append(seq_str)

    return results


def compute_log_probs(model, feat, S_seqs, device="cuda"):
    """Compute per-sequence log-probabilities under ProteinMPNN.

    Args:
        model: ProteinMPNN model
        feat: featurized backbone dict from featurize_pdb
        S_seqs: list of sequence strings

    Returns:
        log_prob_per_seq: tensor of shape (N,) with mean log-prob per sequence
        log_probs_all: tensor of shape (N, L, 21) full log-probs
    """
    X = feat["X"]
    mask = feat["mask"]
    chain_M = feat["chain_M"]
    chain_M_pos = feat["chain_M_pos"]
    chain_encoding_all = feat["chain_encoding_all"]
    residue_idx = feat["residue_idx"]

    L = X.shape[1]

    S_indices = []
    for seq in S_seqs:
        full_seq_idx = torch.zeros(L, dtype=torch.long, device=device)
        for j, aa in enumerate(seq):
            if j < L:
                full_seq_idx[j] = MPNN_ALPHABET.index(aa) if aa in MPNN_ALPHABET else 20
        S_indices.append(full_seq_idx)

    S_batch = torch.stack(S_indices, dim=0)

    X_rep = X.expand(len(S_seqs), -1, -1, -1)
    mask_rep = mask.expand(len(S_seqs), -1)
    chain_M_rep = (chain_M * chain_M_pos).expand(len(S_seqs), -1)
    chain_enc_rep = chain_encoding_all.expand(len(S_seqs), -1)
    res_idx_rep = residue_idx.expand(len(S_seqs), -1)

    randn = torch.randn(S_batch.shape, device=device)
    log_probs = model(X_rep, S_batch, mask_rep, chain_M_rep, res_idx_rep, chain_enc_rep, randn)

    scores = _scores(S_batch, log_probs, mask_rep)
    return -scores, log_probs


def seq_to_S_tensor(seq, device="cuda"):
    """Convert a sequence string to ProteinMPNN index tensor."""
    return torch.tensor(
        [MPNN_ALPHABET.index(aa) if aa in MPNN_ALPHABET else 20 for aa in seq],
        dtype=torch.long, device=device,
    )
