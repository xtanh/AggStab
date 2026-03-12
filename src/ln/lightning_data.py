import os
import sys
import torch
import pytorch_lightning as pl
from transformers import EsmTokenizer


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.insert(0, PROJ_DIR)

from datasets.dataset import ProteinDataset
from torch.utils.data import DataLoader

_tokenizer = None


def _get_tokenizer(saprot_path):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = EsmTokenizer.from_pretrained(saprot_path)
    return _tokenizer


def _sa_seq_to_spaced(sa_seq, max_tokens=1022):
    """Convert contiguous SA sequence 'H#MwKa...' to space-separated 'H# Mw Ka ...'"""
    tokens = [sa_seq[i:i+2] for i in range(0, len(sa_seq), 2)]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)


def make_collate_fn(saprot_path):
    tokenizer = _get_tokenizer(saprot_path)

    def simple_collate(batch):
        sa_seqs = [b['sa_sequence'] for b in batch]
        scores = [b['score'] for b in batch]

        spaced_seqs = [_sa_seq_to_spaced(seq) for seq in sa_seqs]
        encoded = tokenizer.batch_encode_plus(spaced_seqs, return_tensors='pt', padding=True)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'score': torch.tensor(scores, dtype=torch.float32),
        }

    return simple_collate


class ProAggDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.train.num_workers
        self.batch_size = cfg.train.batch_size
        self.collate_fn = make_collate_fn(cfg.model.saprot_path)
        self.train_dataset = ProteinDataset(cfg, 'train')
        self.valid_dataset = ProteinDataset(cfg, 'valid')
        self.test_dataset = ProteinDataset(cfg, 'test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
