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

from src.models.factory import build_proagg_model

class LightningProAggModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": dict(cfg)})
        self.cfg = cfg
        self.model = build_proagg_model(self.cfg)
        self.target_key = cfg.train.get("target_key", "score")
        self.output_key = cfg.train.get("output_key", "score")
        self.ci_weighting = cfg.train.get("ci_weighting", False)
        self.ci_key = cfg.train.get("ci_key", None)
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
        pred = out[self.output_key].float().flatten()
        tgt = batch[self.target_key].float().flatten()
        return pred, tgt

    def _loss(self, pred, tgt, batch):
        valid_mask = torch.isfinite(tgt)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[valid_mask]
        tgt = tgt[valid_mask]
        sq_err = (pred - tgt) ** 2

        if self.ci_weighting and self.ci_key and self.ci_key in batch:
            ci = batch[self.ci_key].float().flatten()[valid_mask]
            ci = torch.nan_to_num(ci, nan=0.0, posinf=0.0, neginf=0.0)
            weights = 1.0 / (1.0 + ci)
            sq_err = sq_err * weights

        return sq_err.mean()

    def training_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self._loss(pred, tgt, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.train_metrics["pearson"].update(pred, tgt)
        self.train_metrics["spearman"].update(pred, tgt)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self._loss(pred, tgt, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.val_metrics["pearson"].update(pred, tgt)
        self.val_metrics["spearman"].update(pred, tgt)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred, tgt = self._pred_tgt(out, batch)
        loss = self._loss(pred, tgt, batch)
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
        use_lora = getattr(self.model, "use_lora", False)

        if use_lora:
            lora_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_" in name:
                    lora_params.append(param)
                else:
                    head_params.append(param)

            lora_lr = self.cfg.train.get("lora_lr", 5e-5)
            head_lr = self.cfg.train.get("lr", 1e-4)
            weight_decay = self.cfg.train.get("weight_decay", 0.01)

            optimizer = torch.optim.AdamW([
                {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
            ])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)

        if self.cfg.train.get("use_scheduler", False):
            sch_cfg = self.cfg.train.scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sch_cfg.get("mode", "max"),
                factor=sch_cfg.get("factor", 0.5),
                patience=sch_cfg.get("lr_patience", 5),
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
