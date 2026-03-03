import os
import sys
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from datetime import datetime

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from src.ln.lightning_data import ProAggDataModule
from src.ln.lightning_model import LightningProAggModel
from src.config.utils import load_yaml_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    parser = argparse.ArgumentParser(description='Train ProAgg for protein aggregation prediction')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # load config
    cfg = load_yaml_config(args.config)
    pl.seed_everything(42)
    
    data_module = ProAggDataModule(cfg)
    
    model = LightningProAggModel(cfg)
    
    logger = TensorBoardLogger(save_dir=cfg.train.save_dir)
    
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_spearman', mode='max', save_last=False, filename='best_{epoch:02d}_{val_spearman:.4f}')
    
    early_stop_callback = EarlyStopping(monitor='val_spearman', patience=cfg.train.patience, mode='max')
    
    trainer = pl.Trainer(max_epochs=cfg.train.epochs,
                         accelerator=cfg.hardware.accelerator,
                         devices=cfg.hardware.devices,
                         precision=cfg.hardware.precision,
                         logger=logger, 
                         log_every_n_steps=10,
                         callbacks=[checkpoint_callback, early_stop_callback])
    
    trainer.fit(model, data_module)
    
    
    # ====== after trainer.fit(...) ======
    if trainer.is_global_zero:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    best_model = LightningProAggModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=cfg
    )

    test_results = trainer.test(best_model, datamodule=data_module)  # list[dict]

    # 保存到权重文件夹
    if trainer.is_global_zero and len(test_results) > 0:
        save_dir = cfg.train.save_dir
        results = dict(test_results[0])  # {'test_loss':..., 'test_spearman':...}
        results["_best_checkpoint"] = checkpoint_callback.best_model_path
        results["_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        txt_path = os.path.join(save_dir, "test_results.txt")
        with open(txt_path, "w") as f:
            f.write("Test Results (best checkpoint)\n")
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