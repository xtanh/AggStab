"""Lightning module for ProAgg with Ranking Loss."""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.models.factory import build_proagg_model


class LightningProAggModelRanking(pl.LightningModule):
    """ProAgg with Ranking Loss (Direct Spearman optimization)."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": dict(cfg)})
        self.cfg = cfg
        self.model = build_proagg_model(self.cfg)

        # Loss weights
        self.mse_weight = cfg.train.get("mse_weight", 1.0)
        self.ranking_weight = cfg.train.get("ranking_weight", 0.5)
        self.ranking_margin = cfg.train.get("ranking_margin", 0.5)
        self.ranking_pair_weighting = cfg.train.get("ranking_pair_weighting", "uniform")
        self.ranking_gap_scale = cfg.train.get("ranking_gap_scale", 1.0)
        self.ranking_pair_clip = cfg.train.get("ranking_pair_clip", None)
        self.ranking_tail_threshold = cfg.train.get("ranking_tail_threshold", None)
        self.ranking_tail_pair_weight = cfg.train.get("ranking_tail_pair_weight", 1.0)
        self.ranking_tail_pair_weight_both = cfg.train.get("ranking_tail_pair_weight_both", None)
        self.regression_sample_weighting = cfg.train.get("regression_sample_weighting", "uniform")
        self.tail_bin_edges = torch.tensor(cfg.train.get("tail_bin_edges", [-3.0, 0.0]), dtype=torch.float32)
        self.register_buffer("tail_bin_edges_buffer", self.tail_bin_edges)
        self.tail_bin_weights = torch.tensor(cfg.train.get("tail_bin_weights", [2.0, 1.0, 1.3]), dtype=torch.float32)
        self.register_buffer("tail_bin_weights_buffer", self.tail_bin_weights)
        self.ordinal_weight = cfg.train.get("ordinal_weight", 0.0)
        self.ordinal_thresholds = torch.tensor(cfg.train.get("ordinal_thresholds", [-3.0, -2.0, -1.0, 0.0]), dtype=torch.float32)
        self.register_buffer("ordinal_thresholds_buffer", self.ordinal_thresholds)
        self.bin_classification_weight = cfg.train.get("bin_classification_weight", 0.0)
        self.bin_edges = torch.tensor(cfg.train.get("bin_edges", [-3.0, -2.0, -1.0, 0.0]), dtype=torch.float32)
        self.register_buffer("bin_edges_buffer", self.bin_edges)
        self.class_weights = self._build_bin_class_weights() if self.bin_classification_weight > 0 else None

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
        self.train_bin_correct = 0
        self.train_bin_total = 0
        self.val_bin_correct = 0
        self.val_bin_total = 0
        self.test_bin_correct = 0
        self.test_bin_total = 0

    def forward(self, batch):
        return self.model(batch)

    def _build_bin_class_weights(self):
        if not self.cfg.train.get("use_bin_class_weights", False):
            return None

        train_path = os.path.join(self.cfg.data_dir, "train.csv")
        df = pd.read_csv(train_path, usecols=["log2_fold_change_75_clip"])
        scores = torch.tensor(df["log2_fold_change_75_clip"].values, dtype=torch.float32)
        labels = self._targets_to_bins(scores)
        counts = torch.bincount(labels, minlength=len(self.bin_edges_buffer) + 1).float()
        counts = counts.clamp(min=1.0)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        return weights

    def _targets_to_bins(self, tgt):
        return torch.bucketize(tgt, self.bin_edges_buffer.to(tgt.device))

    def _classification_loss(self, logits, tgt_bins):
        if self.class_weights is not None:
            return nn.functional.cross_entropy(logits, tgt_bins, weight=self.class_weights.to(logits.device))
        return nn.functional.cross_entropy(logits, tgt_bins)

    def _regression_loss(self, pred, tgt):
        if self.regression_sample_weighting != "tail":
            return nn.functional.mse_loss(pred, tgt)

        bucket_idx = torch.bucketize(tgt.detach(), self.tail_bin_edges_buffer.to(tgt.device))
        sample_weights = self.tail_bin_weights_buffer.to(tgt.device)[bucket_idx]
        sq_error = (pred - tgt).pow(2)
        return (sq_error * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)

    def _ordinal_targets(self, tgt):
        thresholds = self.ordinal_thresholds_buffer.to(tgt.device)
        return (tgt.unsqueeze(-1) > thresholds.unsqueeze(0)).float()

    def _ordinal_loss(self, logits, tgt):
        ordinal_tgt = self._ordinal_targets(tgt)
        return nn.functional.binary_cross_entropy_with_logits(logits, ordinal_tgt)

    def _ordinal_accuracy(self, logits, tgt):
        ordinal_tgt = self._ordinal_targets(tgt)
        pred = (torch.sigmoid(logits) > 0.5).float()
        return (pred == ordinal_tgt).float().mean()

    def _update_bin_stats(self, stage, logits, tgt_bins):
        pred_bins = torch.argmax(logits, dim=-1)
        correct = (pred_bins == tgt_bins).sum().item()
        total = tgt_bins.numel()

        if stage == "train":
            self.train_bin_correct += correct
            self.train_bin_total += total
        elif stage == "val":
            self.val_bin_correct += correct
            self.val_bin_total += total
        else:
            self.test_bin_correct += correct
            self.test_bin_total += total

    def ranking_loss(self, pred, tgt):
        """Pairwise ranking loss - encourages correct ordering.

        Loss = max(0, margin - (pred_i - pred_j) * sign(tgt_i - tgt_j))
        """
        batch_size = pred.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=pred.device)

        # 构建所有pairs的差异
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # (B, B)
        tgt_diff = tgt.unsqueeze(0) - tgt.unsqueeze(1)     # (B, B)

        # 只考虑差异明显的pairs (|tgt_i - tgt_j| > margin)
        mask = (torch.abs(tgt_diff) > self.ranking_margin).float()

        # 排除对角线
        eye_mask = 1 - torch.eye(batch_size, device=pred.device)
        mask = mask * eye_mask

        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Ranking loss: hinge loss
        # 如果 tgt_i > tgt_j, 我们希望 pred_i > pred_j
        # sign = +1 when tgt_i > tgt_j, -1 when tgt_i < tgt_j
        sign = torch.sign(tgt_diff)

        # 理想情况下: pred_diff * sign > 0 (顺序正确)
        # Loss = max(0, 1 - pred_diff * sign)
        loss_matrix = torch.clamp(1.0 - pred_diff * sign, min=0.0)

        if self.ranking_pair_weighting == "gap":
            pair_weights = (torch.abs(tgt_diff) - self.ranking_margin).clamp(min=0.0)
            pair_weights = pair_weights / max(self.ranking_gap_scale, 1e-8)
            pair_weights = 1.0 + pair_weights
            if self.ranking_pair_clip is not None:
                pair_weights = pair_weights.clamp(max=float(self.ranking_pair_clip))
        else:
            pair_weights = torch.ones_like(loss_matrix)

        if self.ranking_tail_threshold is not None:
            tail_mask = (tgt <= float(self.ranking_tail_threshold))
            tail_pair_mask = tail_mask.unsqueeze(0) | tail_mask.unsqueeze(1)
            pair_weights = torch.where(
                tail_pair_mask,
                pair_weights * float(self.ranking_tail_pair_weight),
                pair_weights,
            )

            both_tail_weight = self.ranking_tail_pair_weight_both
            if both_tail_weight is not None:
                both_tail_mask = tail_mask.unsqueeze(0) & tail_mask.unsqueeze(1)
                pair_weights = torch.where(
                    both_tail_mask,
                    pair_weights * (float(both_tail_weight) / max(float(self.ranking_tail_pair_weight), 1e-8)),
                    pair_weights,
                )

        weighted_mask = mask * pair_weights

        # 只计算满足mask的pairs
        loss = (loss_matrix * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)

        return loss

    def training_step(self, batch, batch_idx):
        out = self(batch)
        tgt = batch["score"].float().flatten()
        pred = out["score"].float().flatten()
        mse_loss = self._regression_loss(pred, tgt)

        # Ranking loss
        rank_loss = self.ranking_loss(pred, tgt)

        # Total loss
        total_loss = self.mse_weight * mse_loss + self.ranking_weight * rank_loss
        if self.ordinal_weight > 0 and "ordinal_logits" in out:
            ordinal_loss = self._ordinal_loss(out["ordinal_logits"].float(), tgt)
            total_loss = total_loss + self.ordinal_weight * ordinal_loss
            self.log("train_ordinal_loss", ordinal_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_ordinal_acc", self._ordinal_accuracy(out["ordinal_logits"].float(), tgt), on_step=False, on_epoch=True, sync_dist=True)
        if self.bin_classification_weight > 0 and "score_bin_logits" in out:
            tgt_bins = self._targets_to_bins(tgt)
            bin_loss = self._classification_loss(out["score_bin_logits"].float(), tgt_bins)
            total_loss = total_loss + self.bin_classification_weight * bin_loss
            self.log("train_bin_loss", bin_loss, on_step=False, on_epoch=True, sync_dist=True)
            self._update_bin_stats("train", out["score_bin_logits"].float(), tgt_bins)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rank_loss", rank_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.train_metrics["pearson"].update(pred, tgt)
        self.train_metrics["spearman"].update(pred, tgt)

        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        tgt = batch["score"].float().flatten()
        pred = out["score"].float().flatten()
        loss = self._regression_loss(pred, tgt)
        if self.ordinal_weight > 0 and "ordinal_logits" in out:
            ordinal_loss = self._ordinal_loss(out["ordinal_logits"].float(), tgt)
            self.log("val_ordinal_loss", ordinal_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("val_ordinal_acc", self._ordinal_accuracy(out["ordinal_logits"].float(), tgt), on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.bin_classification_weight > 0 and "score_bin_logits" in out:
            tgt_bins = self._targets_to_bins(tgt)
            bin_loss = self._classification_loss(out["score_bin_logits"].float(), tgt_bins)
            self.log("val_bin_loss", bin_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            self._update_bin_stats("val", out["score_bin_logits"].float(), tgt_bins)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.val_metrics["pearson"].update(pred, tgt)
        self.val_metrics["spearman"].update(pred, tgt)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        tgt = batch["score"].float().flatten()
        pred = out["score"].float().flatten()
        loss = self._regression_loss(pred, tgt)
        if self.ordinal_weight > 0 and "ordinal_logits" in out:
            ordinal_loss = self._ordinal_loss(out["ordinal_logits"].float(), tgt)
            self.log("test_ordinal_loss", ordinal_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_ordinal_acc", self._ordinal_accuracy(out["ordinal_logits"].float(), tgt), on_step=False, on_epoch=True, sync_dist=True)
        if self.bin_classification_weight > 0 and "score_bin_logits" in out:
            tgt_bins = self._targets_to_bins(tgt)
            bin_loss = self._classification_loss(out["score_bin_logits"].float(), tgt_bins)
            self.log("test_bin_loss", bin_loss, on_step=False, on_epoch=True, sync_dist=True)
            self._update_bin_stats("test", out["score_bin_logits"].float(), tgt_bins)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.test_metrics["pearson"].update(pred, tgt)
        self.test_metrics["spearman"].update(pred, tgt)
        return loss

    def on_train_epoch_end(self):
        self.log("train_pearson", self.train_metrics["pearson"].compute(), sync_dist=True)
        self.log("train_spearman", self.train_metrics["spearman"].compute(), sync_dist=True)
        if self.train_bin_total > 0:
            self.log("train_bin_acc", self.train_bin_correct / self.train_bin_total, sync_dist=True)
            self.train_bin_correct = 0
            self.train_bin_total = 0
        self.train_metrics["pearson"].reset()
        self.train_metrics["spearman"].reset()

    def on_validation_epoch_end(self):
        self.log("val_pearson", self.val_metrics["pearson"].compute(), sync_dist=True, prog_bar=True)
        self.log("val_spearman", self.val_metrics["spearman"].compute(), sync_dist=True, prog_bar=True)
        if self.val_bin_total > 0:
            self.log("val_bin_acc", self.val_bin_correct / self.val_bin_total, sync_dist=True, prog_bar=False)
            self.val_bin_correct = 0
            self.val_bin_total = 0
        self.val_metrics["pearson"].reset()
        self.val_metrics["spearman"].reset()

    def on_test_epoch_end(self):
        self.log("test_pearson", self.test_metrics["pearson"].compute(), sync_dist=True)
        self.log("test_spearman", self.test_metrics["spearman"].compute(), sync_dist=True)
        if self.test_bin_total > 0:
            self.log("test_bin_acc", self.test_bin_correct / self.test_bin_total, sync_dist=True)
            self.test_bin_correct = 0
            self.test_bin_total = 0
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
