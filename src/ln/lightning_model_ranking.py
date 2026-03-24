"""Lightning module for ProAgg with Ranking Loss."""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

# Import ProAgg models to ensure registration
import src.models.ProAgg  # noqa: F401
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
        self.cl_weight = cfg.train.get("cl_weight", 0.0)
        self.cl_temp = cfg.train.get("cl_temperature", 0.1)
        self.cl_pos_sigma = cfg.train.get("cl_pos_sigma", 0.5)
        self.cl_neg_sigma = cfg.train.get("cl_neg_sigma", 2.0)
        self.delta_g_weight = cfg.train.get("deltaG_weight", 0.0)
        self.delta_g_ci_weighting = cfg.train.get("deltaG_ci_weighting", False)

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

    def _forward_for_training(self, batch):
        if self.cl_weight <= 0:
            return self.model(batch)

        try:
            return self.model(batch, return_embedding=True)
        except TypeError as exc:
            raise TypeError(
                "Contrastive auxiliary loss requires a model that supports "
                "forward(batch, return_embedding=True)."
            ) from exc

    def _regression_loss(self, pred, tgt):
        return nn.functional.mse_loss(pred, tgt)

    def _delta_g_loss(self, pred_delta_g, batch):
        target = batch["deltaG"].float().flatten()
        pred = pred_delta_g.float().flatten()
        valid_mask = torch.isfinite(target)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        loss = (pred[valid_mask] - target[valid_mask]) ** 2
        if self.delta_g_ci_weighting and "deltaG_95CI" in batch:
            ci = batch["deltaG_95CI"].float().flatten()[valid_mask]
            ci = torch.nan_to_num(ci, nan=0.0, posinf=0.0, neginf=0.0)
            weights = 1.0 / (1.0 + ci)
            loss = loss * weights
        return loss.mean()

    def _contrastive_loss(self, embeddings, scores):
        """Continuous-weighted contrastive loss.

        Instead of hard thresholds (|diff| < pos_threshold for positive,
        |diff| > neg_threshold for negative), uses continuous weights based on
        Gaussian kernels. All pairs contribute to the loss with varying weights.
        """
        batch_size = embeddings.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute score differences matrix
        score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (B, B)
        abs_score_diff = score_diff.abs()

        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.cl_temp

        # Exclude diagonal (self-similarity)
        eye_mask = 1 - torch.eye(batch_size, device=embeddings.device)

        # Continuous positive weights: higher when |diff| is small
        # Gaussian kernel: w_pos = exp(-|diff|^2 / (2 * sigma_pos^2))
        pos_weights = torch.exp(-abs_score_diff ** 2 / (2 * self.cl_pos_sigma ** 2))
        pos_weights = pos_weights * eye_mask  # Remove diagonal

        # Continuous negative weights: higher when |diff| is large
        # Using (1 - Gaussian) or directly based on distance
        # Gaussian with larger sigma, then invert: w_neg = 1 - exp(-|diff|^2 / (2 * sigma_neg^2))
        neg_weights = 1.0 - torch.exp(-abs_score_diff ** 2 / (2 * self.cl_neg_sigma ** 2))
        neg_weights = neg_weights * eye_mask  # Remove diagonal

        # InfoNCE with continuous weights
        loss = torch.tensor(0.0, device=embeddings.device)
        total_weight = torch.tensor(0.0, device=embeddings.device)

        for i in range(batch_size):
            # All other samples are potential positives/negatives with weights
            pos_w = pos_weights[i]  # (B,)
            neg_w = neg_weights[i]  # (B,)

            # Skip if this sample has no valid pairs
            if pos_w.sum() < 1e-8 or neg_w.sum() < 1e-8:
                continue

            # Weighted positive similarity
            # Instead of hard selection, use weighted sum
            pos_sim = sim_matrix[i]  # (B,)

            # Weighted negative similarity
            neg_sim = sim_matrix[i]  # (B,)

            # Compute weighted InfoNCE
            # numerator: weighted sum of positive similarities
            weighted_pos_sim = (pos_w * torch.exp(pos_sim)).sum()

            # denominator: weighted sum of all similarities (pos + neg)
            # Use pos_weights for positive contribution, neg_weights for negative
            weighted_all_sim = (pos_w * torch.exp(pos_sim)).sum() + (neg_w * torch.exp(neg_sim)).sum()

            if weighted_all_sim > 0:
                # Use average of pos and neg weights as anchor weight
                anchor_weight = (pos_w.sum() + neg_w.sum()) / 2
                loss = loss - anchor_weight * torch.log(weighted_pos_sim / weighted_all_sim)
                total_weight = total_weight + anchor_weight

        if total_weight.item() > 0:
            loss = loss / total_weight
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss

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

        weighted_mask = mask * pair_weights

        # 只计算满足mask的pairs
        loss = (loss_matrix * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)

        return loss

    def training_step(self, batch, batch_idx):
        out = self._forward_for_training(batch)
        tgt = batch["score"].float().flatten()
        pred = out["score"].float().flatten()
        mse_loss = self._regression_loss(pred, tgt)

        # Ranking loss
        rank_loss = self.ranking_loss(pred, tgt)

        # Total loss
        total_loss = self.mse_weight * mse_loss + self.ranking_weight * rank_loss
        if self.cl_weight > 0:
            if "embedding" not in out:
                raise ValueError("Contrastive auxiliary loss enabled but model output has no 'embedding'.")
            cl_loss = self._contrastive_loss(out["embedding"], tgt)
            total_loss = total_loss + self.cl_weight * cl_loss
            self.log("train_cl_loss", cl_loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.delta_g_weight > 0 and "deltaG" in out:
            delta_g_loss = self._delta_g_loss(out["deltaG"], batch)
            total_loss = total_loss + self.delta_g_weight * delta_g_loss
            self.log("train_deltaG_loss", delta_g_loss, on_step=False, on_epoch=True, sync_dist=True)

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

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.val_metrics["pearson"].update(pred, tgt)
        self.val_metrics["spearman"].update(pred, tgt)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        tgt = batch["score"].float().flatten()
        pred = out["score"].float().flatten()
        loss = self._regression_loss(pred, tgt)

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
            for name, param in self.named_parameters():
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
