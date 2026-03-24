"""Multi-task dataset for ProAgg - predicts multiple aggregation scores."""
import os
import torch
import pandas as pd


class ProteinDatasetMultiTask(torch.utils.data.Dataset):
    """Dataset that returns multiple aggregation targets (50, 75, 4 clip)."""

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        data_path = os.path.join(self.cfg.data_dir, f'{split}.csv')
        self.df = pd.read_csv(data_path)
        print(f"Including {len(self.df)} proteins in the multi-task dataset")

        # Check which targets are available
        self.available_targets = []
        for col in ['log2_fold_change_50_clip', 'log2_fold_change_75_clip', 'log2_fold_change_4_clip']:
            if col in self.df.columns:
                self.available_targets.append(col)
        print(f"Available targets: {self.available_targets}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Return all available targets
        targets = {}
        for col in self.available_targets:
            targets[col.replace('log2_fold_change_', '').replace('_clip', '')] = row[col]

        return {
            'sa_sequence': row['sa_sequence_foldseek'],
            'targets': targets,
            'score': row['log2_fold_change_75_clip'],  # Keep for compatibility
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.create()
    cfg.data_dir = '/home/xy_th/Project_protein_aggregation/data/rocklin'
    dataset = ProteinDatasetMultiTask(cfg, 'train')
    sample = dataset[0]
    print(sample)
