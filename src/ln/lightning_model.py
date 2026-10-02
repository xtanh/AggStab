"""Minimal Lightning wrapper retained for released checkpoint compatibility."""

import pytorch_lightning as pl

from src.models.factory import build_proagg_model


class LightningProAggModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters({"cfg": dict(cfg)})
        self.cfg = cfg
        self.model = build_proagg_model(cfg)

    def forward(self, batch):
        return self.model(batch)
