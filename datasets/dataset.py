import os
import torch
import pandas as pd


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        data_path = os.path.join(self.cfg.data_dir, f'{split}.csv')
        self.df = pd.read_csv(data_path)

        dg_path = os.path.join(self.cfg.data_dir, 'Metagenomic_dG.csv')
        if os.path.exists(dg_path):
            dg_df = pd.read_csv(
                dg_path,
                usecols=['name', 'deltaG', 'deltaG_95CI'],
            )
            self.df = self.df.merge(dg_df, on='name', how='left')
        else:
            self.df['deltaG'] = float('nan')
            self.df['deltaG_95CI'] = float('nan')
        print("Including %s proteins in the dataset" % len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'sa_sequence': row['sa_sequence_foldseek'],
            'protein_sequence': row['protein_sequence'],
            'score': row['log2_fold_change_75_clip'],
            'deltaG': row.get('deltaG', float('nan')),
            'deltaG_95CI': row.get('deltaG_95CI', float('nan')),
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.create()
    cfg.data_dir = '/home/xy_th/Project_protein_aggregation/data/rocklin'
    dataset = ProteinDataset(cfg, 'train')
    sample = dataset[0]
    print(sample)
