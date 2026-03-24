"""Lightning module for ProAgg with Multi-Task Learning."""
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.models.factory import build_proagg_model


class LightningProAggModelMultiTask(pl.LightningModule):
    """ProAgg with Multi-Task Learning (50, 75, 4 clip)."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": dict(cfg)})
        self.cfg = cfg
        self.model = build_proagg_model(self.cfg)

        # Task weights (can be tuned)
        self.task_weights = cfg.train.get("task_weights", {
            "50": 1.0,
            "75": 1.0,
            "4": 1.0
        })

        # Loss weights
        self.mse_weight = cfg.train.get("mse_weight", 1.0)
        self.ranking_weight = cfg.train.get("ranking_weight", 0.5)
        self.ranking_margin = cfg.train.get("ranking_margin", 0.3)

        # Metrics for each task
        self.tasks = ["50", "75", "4"]
        self.train_metrics = nn.ModuleDict({
            task: nn.ModuleDict({
                "pearson": PearsonCorrCoef(),
                "spearman": SpearmanCorrCoef(),
            }) for task in self.tasks
        })
        self.val_metrics = nn.ModuleDict({
            task: nn.ModuleDict({
                "pearson": PearsonCorrCoef(),
                "spearman": SpearmanCorrCoef(),
            }) for task in self.tasks
        })
        self.test_metrics = nn.ModuleDict({
            task: nn.ModuleDict({
                "pearson": PearsonCorrCoef(),
                "spearman": SpearmanCorrCoef(),
            }) for task in self.tasks
        })

    def forward(self, batch):
        return self.model(batch)

    def ranking_loss(self, pred, tgt):
        """Pairwise ranking loss."""
        batch_size = pred.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        tgt_diff = tgt.unsqueeze(0) - tgt.unsqueeze(1)

        mask = (torch.abs(tgt_diff) > self.ranking_margin).float()
        eye_mask = 1 - torch.eye(batch_size, device=pred.device)
        mask = mask * eye_mask

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        sign = torch.sign(tgt_diff)
        loss_matrix = torch.clamp(1.0 - pred_diff * sign, min=0.0)
        loss = (loss_matrix * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def training_step(self, batch, batch_idx):
        out = self(batch)
        targets = batch["targets"]

        total_loss = 0.0
        task_losses = {}

        # Multi-task MSE + Ranking loss
        for task in self.tasks:
            key = f"score_{task}"
            if key in out and task in targets:
                pred = out[key].float().flatten()
                tgt = targets[task].float().flatten()

                # MSE loss
                mse_loss = nn.functional.mse_loss(pred, tgt)

                # Ranking loss (only for main task 75 if specified)
                rank_loss = 0.0
                if self.ranking_weight > 0:
                    rank_loss = self.ranking_loss(pred, tgt)

                # Combined loss for this task
                task_loss = self.mse_weight * mse_loss + self.ranking_weight * rank_loss
                weighted_loss = self.task_weights[task] * task_loss

                total_loss += weighted_loss
                task_losses[f"train_mse_{task}"] = mse_loss
                task_losses[f"train_rank_{task}"] = rank_loss

                # Update metrics
                self.train_metrics[task]["pearson"].update(pred, tgt)
                self.train_metrics[task]["spearman"].update(pred, tgt)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in task_losses.items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        targets = batch["targets"]

        total_loss = 0.0

        for task in self.tasks:
            key = f"score_{task}"
            if key in out and task in targets:
                pred = out[key].float().flatten()
                tgt = targets[task].float().flatten()

                mse_loss = nn.functional.mse_loss(pred, tgt)
                total_loss += self.task_weights[task] * mse_loss

                self.val_metrics[task]["pearson"].update(pred, tgt)
                self.val_metrics[task]["spearman"].update(pred, tgt)

        # Log main task (75) val_spearman for early stopping
        if "75" in self.tasks and "score_75" in out:
            main_pred = out["score_75"].float().flatten()
            main_tgt = targets["75"].float().flatten()
            self.val_metrics["75"]["spearman"].update(main_pred, main_tgt)

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        targets = batch["targets"]

        total_loss = 0.0

        for task in self.tasks:
            key = f"score_{task}"
            if key in out and task in targets:
                pred = out[key].float().flatten()
                tgt = targets[task].float().flatten()

                mse_loss = nn.functional.mse_loss(pred, tgt)
                total_loss += self.task_weights[task] * mse_loss

                self.test_metrics[task]["pearson"].update(pred, tgt)
                self.test_metrics[task]["spearman"].update(pred, tgt)

        self.log("test_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def on_train_epoch_end(self):
        for task in self.tasks:
            self.log(f"train_pearson_{task}", self.train_metrics[task]["pearson"].compute(), sync_dist=True)
            self.log(f"train_spearman_{task}", self.train_metrics[task]["spearman"].compute(), sync_dist=True)
            self.train_metrics[task]["pearson"].reset()
            self.train_metrics[task]["spearman"].reset()

    def on_validation_epoch_end(self):
        for task in self.tasks:
            self.log(f"val_pearson_{task}", self.val_metrics[task]["pearson"].compute(), sync_dist=True)
            self.log(f"val_spearman_{task}", self.val_metrics[task]["spearman"].compute(), sync_dist=True, prog_bar=(task=="75"))
            self.val_metrics[task]["pearson"].reset()
            self.val_metrics[task]["spearman"].reset()

    def on_test_epoch_end(self):
        for task in self.tasks:
            self.log(f"test_pearson_{task}", self.test_metrics[task]["pearson"].compute(), sync_dist=True)
            self.log(f"test_spearman_{task}", self.test_metrics[task]["spearman"].compute(), sync_dist=True)
            self.test_metrics[task]["pearson"].reset()
            self.test_metrics[task]["spearman"].reset()

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
                    "monitor": self.cfg.train.get("monitor_metric", "val_spearman_75"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
