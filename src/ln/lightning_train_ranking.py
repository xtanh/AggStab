"""Train ProAgg with Ranking Loss."""
import os
import sys
import argparse
from collections import OrderedDict
import pytorch_lightning as pl
from datetime import datetime
import torch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_ranking import LightningProAggModelRanking
from src.config.utils import load_yaml_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def maybe_load_init_checkpoint(model, cfg):
    init_checkpoint = cfg.train.get("init_checkpoint", None)
    if not init_checkpoint:
        return

    checkpoint = torch.load(init_checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value

    missing_keys, unexpected_keys = model.model.load_state_dict(model_state, strict=False)
    print(f"Initialized model weights from: {init_checkpoint}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")


def main():
    parser = argparse.ArgumentParser(description='Train ProAgg with Ranking Loss')
    parser.add_argument('--config', type=str, default='configs/proagg_v10.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    pl.seed_everything(42)

    data_module = ProAggDataModule(cfg)
    model = LightningProAggModelRanking(cfg)
    maybe_load_init_checkpoint(model, cfg)

    logger = TensorBoardLogger(save_dir=cfg.train.save_dir)
    monitor_metric = cfg.train.get('monitor_metric', 'val_spearman')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=monitor_metric,
        mode='max',
        save_last=False,
        filename='best_{epoch:02d}_{' + monitor_metric + ':.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=cfg.train.patience,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=cfg.train.get('gradient_clip_val', 0),
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, data_module)

    if trainer.is_global_zero:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    best_model = LightningProAggModelRanking.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=cfg
    )

    test_results = trainer.test(best_model, datamodule=data_module)

    if trainer.is_global_zero and len(test_results) > 0:
        run_dir = os.path.dirname(checkpoint_callback.best_model_path)
        results = dict(test_results[0])
        results["_best_checkpoint"] = checkpoint_callback.best_model_path
        results["_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        txt_path = os.path.join(run_dir, "test_results.txt")
        with open(txt_path, "w") as f:
            f.write("Test Results (best checkpoint - Ranking Loss)\n")
            f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
            f.write(f"Time: {results['_time']}\n")
            f.write("=" * 60 + "\n")
            for k, v in results.items():
                if k.startswith("_"):
                    continue
                try:
                    f.write(f"{k}: {float(v):.6f}\n")
                except Exception:
                    f.write(f"{k}: {v}\n")
            f.write("=" * 60 + "\n")
        print(f"Saved test results to:\n  {txt_path}")


if __name__ == "__main__":
    main()
