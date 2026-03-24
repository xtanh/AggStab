"""Train ProAgg with Contrastive Learning."""
import os
import sys
import argparse
import pytorch_lightning as pl
from datetime import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model_cl import LightningProAggModelCL
from src.config.utils import load_yaml_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    parser = argparse.ArgumentParser(description='Train ProAgg with Contrastive Learning')
    parser.add_argument('--config', type=str, default='configs/proagg_v9.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    pl.seed_everything(42)

    data_module = ProAggDataModule(cfg)
    model = LightningProAggModelCL(cfg)

    logger = TensorBoardLogger(save_dir=cfg.train.save_dir)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_spearman',
        mode='max',
        save_last=False,
        filename='best_{epoch:02d}_{val_spearman:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_spearman',
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

    best_model = LightningProAggModelCL.load_from_checkpoint(
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
            f.write("Test Results (best checkpoint - Contrastive Learning)\n")
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
