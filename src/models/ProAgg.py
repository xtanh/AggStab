import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.models.esm2 import ESM2


class ProAgg(torch.nn.Module):
    def __init__(self, cfg):
        super(ProAgg, self).__init__()
        self.cfg = cfg
        self.decoder = ESM2.from_pretrained(args=None)

        self.proj = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch_dict):
        seq_tokens = batch_dict['seq_tokens']

        decoder_out = self.decoder(tokens=seq_tokens, encoder_out=None)
        protein_repr = decoder_out["representations"][-1]

        # Remove CLS and EOS tokens
        protein_repr = protein_repr[:, 1:-1, :]

        protein_repr = self.proj(protein_repr)  # (B, L, 1)
        out = protein_repr.mean(dim=1)          # (B, 1)

        return {'score': out}


if __name__ == "__main__":
    from src.config.utils import load_yaml_config
    cfg = load_yaml_config('/home/xy_th/Project_protein_aggregation/configs/default.yaml')
    model = ProAgg(cfg)
    print(model)
