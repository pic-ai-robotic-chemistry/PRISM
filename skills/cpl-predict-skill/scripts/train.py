"""Training entry point for CPL spectrum prediction.

Usage:
    python scripts/train.py
    python scripts/train.py trainer.max_epochs=200 model.hidden_dim=512
"""

import os
import time
from datetime import datetime
from pathlib import Path

# Disable NCCL P2P to avoid multi-GPU deadlock without NVCC
os.environ["NCCL_P2P_DISABLE"] = "1"

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from cpl_model import CPLLightningModule, CPLDataModule
from viz_callback import NatureVizCallback


ConfigStore.instance().store(
    name="train",
    node={
        "tag": "cpl_cnn",
        "data_path": "data/cpl_cd.pkl",
        "split_ratio": 0.9,
        "train_keys_path": None,
        "val_keys_path": None,
        "seed": 42,
        "num_workers": 4,
        "spectrum_transform": {
            "type": "asinh",
            "scale_quantile": 0.75,
        },
        "model": {
            "embed_dim": 32,
            "hidden_dim": 256,
            "use_rbf": True,
            "rbf_centers": 16,
            "rbf_gamma": 10.0,
            "dropout": 0.1,
            "n_layers": 5,
            "base_ch": 128,
            "init_len": 8,
            "smooth_kernel": 5,
            "smooth_sigma": 1.0,
        },
        "trainer": {
            "max_epochs": 5000,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "raw_loss_weight": 0.000001,
            "warmup_epochs": 50,
            "patience": 100,
            "gradient_clip_val": 1.0,
            "precision": 32,
        },
        "hydra": {
            "run": {"dir": "."},
            "output_subdir": None,
            "job": {"chdir": False},
        },
        "viz": {
            "plot_dir": "plots",
            "n_samples": 6,
            "loss_curve_name": "loss_curve.png",
        },
    },
)


def _get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


@hydra.main(version_base=None, config_path=None, config_name="train")
def main(cfg: DictConfig):
    # Print config (rank 0 only to avoid duplicate logs)
    rank = _get_rank()
    if rank == 0:
        print(OmegaConf.to_yaml(cfg))

    if cfg.spectrum_transform.type != "asinh":
        raise ValueError(f"Unsupported spectrum_transform.type: {cfg.spectrum_transform.type}")

    # Unified output directory: logs/{tag}_{timestamp}/
    # All processes use the same minute-level timestamp to avoid mismatched dirs.
    tag = cfg.get("tag", "cpl_cnn")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs") / f"{tag}_{timestamp}"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write a sentinel so other ranks know the exact dir name
        sentinel = Path("logs") / ".latest_run"
        sentinel.write_text(str(output_dir.name))
    else:
        # Wait for rank 0 to create the sentinel, then read the dir name
        sentinel = Path("logs") / ".latest_run"
        for _ in range(100):
            if sentinel.exists():
                break
            time.sleep(0.05)
        dir_name = sentinel.read_text().strip() if sentinel.exists() else f"{tag}_{timestamp}"
        output_dir = Path("logs") / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"Output directory: {output_dir.absolute()}")

    # Reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # DataModule
    datamodule = CPLDataModule(
        data_path=cfg.data_path,
        batch_size=cfg.trainer.batch_size,
        split_ratio=cfg.split_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        spectrum_scale_quantile=cfg.spectrum_transform.scale_quantile,
        train_keys_path=cfg.train_keys_path,
        val_keys_path=cfg.val_keys_path,
    )
    datamodule.setup()

    # Model
    model = CPLLightningModule(
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        use_rbf=cfg.model.use_rbf,
        rbf_centers=cfg.model.rbf_centers,
        rbf_gamma=cfg.model.rbf_gamma,
        dropout=cfg.model.dropout,
        n_layers=cfg.model.n_layers,
        base_ch=cfg.model.base_ch,
        init_len=cfg.model.init_len,
        smooth_kernel=cfg.model.smooth_kernel,
        smooth_sigma=cfg.model.smooth_sigma,
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        warmup_epochs=cfg.trainer.warmup_epochs,
        max_epochs=cfg.trainer.max_epochs,
        spectrum_scale=datamodule.spectrum_scale,
        raw_loss_weight=cfg.trainer.raw_loss_weight,
    )

    # Logger -> logs/{tag}_{time}/csv_logs/
    logger = CSVLogger(
        save_dir=str(output_dir),
        name="csv_logs",
        version="",
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.trainer.patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        NatureVizCallback(
            plot_dir=str(output_dir / "plots"),
            n_samples=cfg.viz.n_samples,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    if rank == 0:
        print("\n=== Training finished ===")
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score:.6f}")


if __name__ == "__main__":
    main()
