"""Nature-style visualization callback for CPL training.

Produces two plots each validation epoch:
1. Loss curve (train + val) vs step, updated every epoch (overwritten)
2. Pred vs Exp spectrum comparison for 6 validation samples
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Set Nature-style matplotlib defaults at import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.5,
    "xtick.minor.width": 0.5,
    "xtick.top": True,
    "ytick.right": True,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


WAVELENGTHS = np.linspace(800, 400, 801)


class NatureVizCallback(Callback):
    """Callback for Nature-style training visualizations."""

    def __init__(self, plot_dir: str = "plots", n_samples: int = 6):
        super().__init__()
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples

        # Loss history: list of (step, loss) tuples
        self.train_losses: List[tuple] = []
        self.val_losses: List[tuple] = []

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step
        loss = trainer.callback_metrics.get("train_loss_step")
        if loss is not None:
            self.train_losses.append((step, loss.item()))

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Only record once per epoch using the epoch-end metric
        pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch

        # Record val loss at this epoch boundary
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            step = trainer.global_step
            self.val_losses.append((step, val_loss.item()))

        # 1. Plot loss curve (overwritten each epoch)
        self._plot_loss_curve()

        # 2. Plot Pred vs Exp spectra
        self._plot_spectra(trainer, pl_module, epoch)

    def _plot_loss_curve(self) -> None:
        if len(self.train_losses) == 0:
            return

        fig, ax = plt.subplots(figsize=(5, 3.5))

        train_steps, train_vals = zip(*self.train_losses)
        ax.plot(train_steps, train_vals, color="#c0392b", linewidth=1.2, label="Train", alpha=0.85)

        if len(self.val_losses) > 0:
            val_steps, val_vals = zip(*self.val_losses)
            ax.plot(val_steps, val_vals, color="#2980b9", linewidth=1.5, label="Validation", marker="o", markersize=3)

        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("MSE Loss", fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        ax.set_title("Training Progress", fontsize=11, fontweight="bold")

        # Log scale if loss spans >2 orders of magnitude
        all_vals = [v for _, v in self.train_losses] + [v for _, v in self.val_losses]
        if len(all_vals) > 0:
            vmin, vmax = min(all_vals), max(all_vals)
            if vmax > 0 and vmin > 0 and vmax / vmin > 100:
                ax.set_yscale("log")

        fig.savefig(self.plot_dir / "loss_curve.png")
        plt.close(fig)

    @torch.no_grad()
    def _plot_spectra(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int) -> None:
        # Get val dataset and pick 6 samples
        val_dataset = trainer.datamodule.val_dataset
        if len(val_dataset) == 0:
            return

        indices = np.linspace(0, len(val_dataset) - 1, min(self.n_samples, len(val_dataset)), dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
        axes = axes.flatten()

        for ax_idx, idx in enumerate(indices):
            ax = axes[ax_idx]
            sample = val_dataset[idx]
            key = val_dataset.keys[idx]
            color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx, spectrum_transformed = sample

            # Inference
            pl_module.eval()
            batch_inputs = [
                color_idx.unsqueeze(0).to(pl_module.device),
                gray_norm.unsqueeze(0).to(pl_module.device),
                thickness_norm.unsqueeze(0).to(pl_module.device),
                stretch_norm.unsqueeze(0).to(pl_module.device),
                is_red_idx.unsqueeze(0).to(pl_module.device),
            ]
            pred_transformed = pl_module(*batch_inputs).squeeze(0).cpu().numpy()

            # Inverse asinh transform back to CD[mdeg]
            scale = pl_module.spectrum_scale
            pred = np.sinh(pred_transformed) * scale
            exp = np.sinh(spectrum_transformed.cpu().numpy()) * scale

            ax.plot(WAVELENGTHS, exp, color="#c0392b", linewidth=1.2, label="Exp")
            ax.plot(WAVELENGTHS, pred, color="#2980b9", linewidth=1.2, label="Pred")

            ax.set_xlabel("Wavelength (nm)", fontsize=8)
            ax.set_ylabel("CD (mdeg)", fontsize=8)
            ax.set_title(key.replace("_", ", "), fontsize=8, fontweight="bold")
            if ax_idx == 0:
                ax.legend(frameon=False, fontsize=7, loc="upper right")

        plt.tight_layout(pad=0.8)
        fig.savefig(self.plot_dir / "spectrum_comparison.png")
        plt.close(fig)
