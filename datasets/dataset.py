import os
import torch
import pickle
import pandas as pd
from ipdb import set_trace


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        data_path = os.path.join(self.cfg.data_dir, f'{split}.csv')
        self.df = pd.read_csv(data_path)
        print("Including %s proteins in the dataset" % len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'protein_sequence': row['protein_sequence'],
            # 'log2_fold_change_50_clip': row['log2_fold_change_50_clip'],
            # 'log2_fold_change_75_clip': row['log2_fold_change_75_clip'],
            # 'log2_fold_change_4_clip': row['log2_fold_change_4_clip'],
            'score': row['log2_fold_change_75_clip'],
        }
        

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.create()
    cfg.data_dir = '/home/xy_th/Project_protein_aggregation/data/rocklin'
    cfg.split = 'train' 
    dataset = ProteinDataset(cfg, 'train')
    sample = dataset[0]
    set_trace()
    print("sample")