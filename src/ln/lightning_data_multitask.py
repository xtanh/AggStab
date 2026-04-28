"""Multi-task Lightning Data Module for ProAgg."""
import os
import sys
import torch
import pytorch_lightning as pl
from transformers import EsmTokenizer

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.insert(0, PROJ_DIR)

from datasets.dataset_multitask import ProteinDatasetMultiTask
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


def make_collate_fn_multitask(saprot_path):
    tokenizer = _get_tokenizer(saprot_path)

    def collate_fn(batch):
        sa_seqs = [b['sa_sequence'] for b in batch]
        targets_list = [b['targets'] for b in batch]

        spaced_seqs = [_sa_seq_to_spaced(seq) for seq in sa_seqs]
        encoded = tokenizer.batch_encode_plus(spaced_seqs, return_tensors='pt', padding=True)

        # Stack targets for each task
        all_task_names = set()
        for t in targets_list:
            all_task_names.update(t.keys())

        targets_dict = {}
        for task_name in all_task_names:
            task_values = [t.get(task_name, 0.0) for t in targets_list]
            targets_dict[task_name] = torch.tensor(task_values, dtype=torch.float32)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'targets': targets_dict,
            'score': targets_dict.get('75', torch.zeros(len(batch))),  # For compatibility
        }

    return collate_fn


class ProAggDataModuleMultiTask(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.train.num_workers
        self.batch_size = cfg.train.batch_size
        self.collate_fn = make_collate_fn_multitask(cfg.model.saprot_path)

    def setup(self, stage=None):
        self.train_dataset = ProteinDatasetMultiTask(self.cfg, 'train')
        self.valid_dataset = ProteinDatasetMultiTask(self.cfg, 'valid')
        self.test_dataset = ProteinDatasetMultiTask(self.cfg, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)
