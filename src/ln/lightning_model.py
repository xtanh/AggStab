import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.regression import  PearsonCorrCoef, SpearmanCorrCoef

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.models.ProAgg import ProAgg

class LightningProAggModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ProAgg(self.cfg)
        self.criterion = nn.MSELoss()

        self.train_metrics = nn.ModuleDict({
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        })
        self.val_metrics = nn.ModuleDict({
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        })
        self.test_metrics = nn.ModuleDict({
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        })

    def forward(self, batch):
        return self.model(batch)

    def _pred_tgt(self, out, batch):
        pred = out["score"].float().flatten()
        tgt  = batch["score"].float().flatten()
        return pred, tgt

    def training_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self.criterion(pred, tgt)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.train_metrics["pearson"].update(pred, tgt)
        self.train_metrics["spearman"].update(pred, tgt)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self.criterion(pred, tgt)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.val_metrics["pearson"].update(pred, tgt)
        self.val_metrics["spearman"].update(pred, tgt)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self.criterion(pred, tgt)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.test_metrics["pearson"].update(pred, tgt)
        self.test_metrics["spearman"].update(pred, tgt)
        return loss

    def on_train_epoch_end(self):
        self.log("train_pearson", self.train_metrics["pearson"].compute(), sync_dist=True)
        self.log("train_spearman", self.train_metrics["spearman"].compute(), sync_dist=True)
        self.train_metrics["pearson"].reset()
        self.train_metrics["spearman"].reset()

    def on_validation_epoch_end(self):
        self.log("val_pearson", self.val_metrics["pearson"].compute(), sync_dist=True, prog_bar=True)
        self.log("val_spearman", self.val_metrics["spearman"].compute(), sync_dist=True, prog_bar=True)
        self.val_metrics["pearson"].reset()
        self.val_metrics["spearman"].reset()

    def on_test_epoch_end(self):
        self.log("test_pearson", self.test_metrics["pearson"].compute(), sync_dist=True)
        self.log("test_spearman", self.test_metrics["spearman"].compute(), sync_dist=True)
        self.test_metrics["pearson"].reset()
        self.test_metrics["spearman"].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)

        if self.cfg.train.get("use_scheduler", False):
            sch_cfg = self.cfg.train.scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sch_cfg.get("mode", "max"),
                factor=sch_cfg.get("factor", 0.5),
                patience=sch_cfg.get("patience", 5),
                min_lr=sch_cfg.get("min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.train.get("monitor_metric", "val_spearman"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

