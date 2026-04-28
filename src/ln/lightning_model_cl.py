"""Lightning module for ProAgg with Contrastive Learning."""
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


class LightningProAggModelCL(pl.LightningModule):
    """ProAgg with Contrastive Learning."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": dict(cfg)})
        self.cfg = cfg
        self.model = build_proagg_model(self.cfg)
        self.criterion = nn.MSELoss()

        # 对比学习超参数
        self.cl_weight = cfg.train.get("cl_weight", 0.1)  # 对比学习loss权重
        self.cl_temp = cfg.train.get("cl_temperature", 0.1)  # 温度参数
        self.pos_threshold = cfg.train.get("cl_pos_threshold", 0.5)  # 正样本阈值
        self.neg_threshold = cfg.train.get("cl_neg_threshold", 2.0)  # 负样本阈值

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

    def forward(self, batch, return_embedding=False):
        return self.model(batch, return_embedding=return_embedding)

    def contrastive_loss(self, embeddings, scores):
        """计算对比学习loss (InfoNCE).

        Args:
            embeddings: (B, D) 归一化后的embedding
            scores: (B,) 聚集分数
        """
        batch_size = embeddings.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=embeddings.device)

        # 计算分数差异矩阵
        score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (B, B)
        score_diff = score_diff.abs()

        # 构建正负样本mask
        pos_mask = (score_diff < self.pos_threshold).float()  # 相似分数为正样本
        neg_mask = (score_diff > self.neg_threshold).float()  # 差异大为负样本

        # 排除对角线（自身）
        eye_mask = 1 - torch.eye(batch_size, device=embeddings.device)
        pos_mask = pos_mask * eye_mask
        neg_mask = neg_mask * eye_mask

        # 计算embedding相似度
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.cl_temp  # (B, B)

        # InfoNCE loss
        loss = 0.0
        num_pos_pairs = 0

        for i in range(batch_size):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # 正样本相似度
            pos_sim = sim_matrix[i, pos_indices]  # (num_pos,)

            # 负样本相似度
            neg_sim = sim_matrix[i, neg_indices]  # (num_neg,)

            # InfoNCE: -log(exp(pos_sim) / sum(exp(neg_sim)))
            numerator = torch.exp(pos_sim).sum()
            denominator = numerator + torch.exp(neg_sim).sum()

            if denominator > 0:
                loss -= torch.log(numerator / denominator)
                num_pos_pairs += len(pos_indices)

        if num_pos_pairs > 0:
            loss = loss / num_pos_pairs
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss

    def training_step(self, batch, batch_idx):
        out = self(batch, return_embedding=True)
        pred = out["score"].float().flatten()
        tgt = batch["score"].float().flatten()
        emb = out["embedding"]  # (B, D)

        # MSE loss
        mse_loss = self.criterion(pred, tgt)

        # 对比学习loss
        cl_loss = self.contrastive_loss(emb, tgt)

        # 总loss
        total_loss = mse_loss + self.cl_weight * cl_loss

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_cl_loss", cl_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.train_metrics["pearson"].update(pred, tgt)
        self.train_metrics["spearman"].update(pred, tgt)

        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        pred = out["score"].float().flatten()
        tgt = batch["score"].float().flatten()
        loss = self.criterion(pred, tgt)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.val_metrics["pearson"].update(pred, tgt)
        self.val_metrics["spearman"].update(pred, tgt)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred = out["score"].float().flatten()
        tgt = batch["score"].float().flatten()
        loss = self.criterion(pred, tgt)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

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
