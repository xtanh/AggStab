import os
import sys
import torch
import pytorch_lightning as pl
from esm.data import Alphabet


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.insert(0, PROJ_DIR)

from datasets.dataset import ProteinDataset
from torch.utils.data import DataLoader

alphabet = Alphabet.from_architecture('ESM-1b')


def simple_collate(batch):
    protein_seqs = [b['protein_sequence'] for b in batch]
    scores = [b['score'] for b in batch]
    
    seq_tokens_list = []
    for seq in protein_seqs:
        # Truncate if necessary (usually 1024 or 1022 for ESM-1b/ESM-2)
        if len(seq) > 1022:
            seq = seq[:1022]
            
        tokens = torch.empty(len(seq)+2, dtype=torch.int64)
        tokens[0] = alphabet.cls_idx
        tokens[1:len(seq)+1] = torch.tensor(alphabet.encode(text=seq), dtype=torch.int64)
        tokens[len(seq)+1] = alphabet.eos_idx
        seq_tokens_list.append(tokens)
    
    max_len = max(len(t) for t in seq_tokens_list)
    # Default padding idx for ESM is 1
    seq_tokens = torch.full((len(seq_tokens_list), max_len), alphabet.padding_idx, dtype=torch.int64)

    for i, tokens in enumerate(seq_tokens_list):
        seq_tokens[i, :len(tokens)] = tokens
        
    return {
        'seq_tokens': seq_tokens,
        'score': torch.tensor(scores, dtype=torch.float32)
    }
    

class ProAggDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.train.num_workers
        self.batch_size = cfg.train.batch_size
        self.train_dataset = ProteinDataset(cfg, 'train')
        self.valid_dataset = ProteinDataset(cfg, 'valid')
        self.test_dataset = ProteinDataset(cfg, 'test')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=simple_collate, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=simple_collate, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=simple_collate, num_workers=self.num_workers)