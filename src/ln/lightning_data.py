import os
import sys
import torch
import pytorch_lightning as pl
from transformers import EsmTokenizer
from torch.utils.data import DataLoader


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.insert(0, PROJ_DIR)

from datasets.dataset import ProteinDataset

_tokenizer = None
_protein_tokenizers = {}


def _get_tokenizer(saprot_path):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = EsmTokenizer.from_pretrained(saprot_path)
    return _tokenizer


def _get_protein_tokenizer(esm2_path):
    if esm2_path not in _protein_tokenizers:
        _protein_tokenizers[esm2_path] = EsmTokenizer.from_pretrained(esm2_path)
    return _protein_tokenizers[esm2_path]


def _sa_seq_to_spaced(sa_seq, max_tokens=1022):
    """Convert contiguous SA sequence 'H#MwKa...' to space-separated 'H# Mw Ka ...'"""
    tokens = [sa_seq[i:i+2] for i in range(0, len(sa_seq), 2)]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)


def make_collate_fn(saprot_path, esm2_path=None):
    tokenizer = _get_tokenizer(saprot_path)
    protein_tokenizer = _get_protein_tokenizer(esm2_path) if esm2_path else None

    def simple_collate(batch):
        sa_seqs = [b['sa_sequence'] for b in batch]
        protein_seqs = [b['protein_sequence'] for b in batch]
        scores = [b['score'] for b in batch]
        delta_gs = [b.get('deltaG', float('nan')) for b in batch]
        delta_g_cis = [b.get('deltaG_95CI', float('nan')) for b in batch]

        spaced_seqs = [_sa_seq_to_spaced(seq) for seq in sa_seqs]
        encoded = tokenizer.batch_encode_plus(spaced_seqs, return_tensors='pt', padding=True)
        batch_out = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'score': torch.tensor(scores, dtype=torch.float32),
            'deltaG': torch.tensor(delta_gs, dtype=torch.float32),
            'deltaG_95CI': torch.tensor(delta_g_cis, dtype=torch.float32),
        }
        if protein_tokenizer is not None:
            protein_encoded = protein_tokenizer.batch_encode_plus(
                protein_seqs,
                return_tensors='pt',
                padding=True,
            )
            batch_out['protein_input_ids'] = protein_encoded['input_ids']
            batch_out['protein_attention_mask'] = protein_encoded['attention_mask']
        return batch_out

    return simple_collate


class ProAggDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.train.num_workers
        self.batch_size = cfg.train.batch_size
        esm2_path = cfg.model.get("esm2_path", None)
        self.collate_fn = make_collate_fn(cfg.model.saprot_path, esm2_path)
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
