"""Inference script for CPL spectrum prediction model.

Usage:
    python scripts/infer.py -c logs/training-run -d data/cpl_cd.pkl -k data/val_keys.txt
"""

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cpl_model import CPLLightningModule

# Scientific visualization style (publication-quality)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cycler

# Publication style: sans-serif, colorblind-safe palette, clean spines
plt.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.grid": False,
    "axes.prop_cycle": cycler(color=["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                                     "#0072B2", "#D55E00", "#CC79A7", "#000000"]),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

WAVELENGTHS = np.linspace(800, 400, 801)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPL model inference")
    parser.add_argument(
        "-c", "--checkpoint_dir", required=True,
        help="Checkpoint directory containing checkpoints/best.ckpt",
    )
    parser.add_argument(
        "-d", "--data_path", required=True,
        help="Path to pickled CPL data (e.g. cpl_cd.pkl)",
    )
    parser.add_argument(
        "-k", "--keys_txt", required=True,
        help="Path to txt file listing data keys to infer (one per line)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for inference: cuda/cpu/auto (default: auto)",
    )
    parser.add_argument(
        "--output_tag", default=None,
        help="Optional tag inserted into output dir name, e.g. infer_val_YYYYmmdd_HHMMSS.",
    )
    return parser.parse_args()


def load_model(checkpoint_dir: str, device: str):
    ckpt_path = Path(checkpoint_dir) / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "spectrum_scale" not in ckpt.get("hyper_parameters", {}):
        raise ValueError(
            "Checkpoint does not contain spectrum_scale. "
            "It was likely trained with the old z-score target transform; retrain before using this infer.py."
        )

    model = CPLLightningModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(device)
    return model, device


def load_data_and_keys(data_path: str, keys_txt: str) -> Tuple[Dict, List[str]]:
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)

    with open(keys_txt, "r") as f:
        keys = [line.strip() for line in f if line.strip()]

    # Validate keys exist in data
    missing = [k for k in keys if k not in raw_data]
    if missing:
        raise ValueError(f"Keys not found in data: {missing}")

    return raw_data, keys


def compute_normalization_params(raw_data: Dict) -> Tuple[Dict, float, float, float, float, float, float]:
    """Compute color_map and min/max ranges from all data."""
    color_set = set()
    grays, thicknesses, stretches = [], [], []
    for key in raw_data.keys():
        color, gray, thickness, stretch, _is_red = key.split("_")
        color_set.add(color)
        grays.append(float(gray))
        thicknesses.append(float(thickness))
        stretches.append(float(stretch))

    color_map = {c: i for i, c in enumerate(sorted(color_set))}
    return (
        color_map,
        min(grays), max(grays),
        min(thicknesses), max(thicknesses),
        min(stretches), max(stretches),
    )


def build_tensors(
    raw_data: Dict,
    keys: List[str],
    color_map: Dict,
    gray_min: float, gray_max: float,
    thickness_min: float, thickness_max: float,
    stretch_min: float, stretch_max: float,
    spectrum_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build input tensors and target spectra from keys."""
    color_idxs, gray_norms, thickness_norms, stretch_norms, is_red_idxs = [], [], [], [], []
    spectra = []

    for key in keys:
        color, gray, thickness, stretch, is_red = key.split("_")
        color_idx = color_map[color]
        is_red_idx = 1 if is_red == "True" else 0

        gray_norm = (float(gray) - gray_min) / (gray_max - gray_min)
        thickness_norm = (float(thickness) - thickness_min) / (thickness_max - thickness_min)
        stretch_norm = (float(stretch) - stretch_min) / (stretch_max - stretch_min)

        spectrum = np.array(raw_data[key]["cd"], dtype=np.float32)
        spectrum_transformed = np.arcsinh(spectrum / spectrum_scale).astype(np.float32)

        color_idxs.append(color_idx)
        gray_norms.append(gray_norm)
        thickness_norms.append(thickness_norm)
        stretch_norms.append(stretch_norm)
        is_red_idxs.append(is_red_idx)
        spectra.append(spectrum_transformed)

    return (
        torch.tensor(color_idxs, dtype=torch.long),
        torch.tensor(gray_norms, dtype=torch.float32).unsqueeze(1),
        torch.tensor(thickness_norms, dtype=torch.float32).unsqueeze(1),
        torch.tensor(stretch_norms, dtype=torch.float32).unsqueeze(1),
        torch.tensor(is_red_idxs, dtype=torch.long),
        torch.tensor(np.stack(spectra), dtype=torch.float32),
    )


@torch.no_grad()
def run_inference(
    model: CPLLightningModule,
    color_idx: torch.Tensor,
    gray_norm: torch.Tensor,
    thickness_norm: torch.Tensor,
    stretch_norm: torch.Tensor,
    is_red_idx: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    dataset = TensorDataset(color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for batch in loader:
        batch = [b.to(device) for b in batch]
        pred = model(*batch)
        preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


def compute_rmse(exp: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-sample RMSE over 801 wavelengths."""
    return np.sqrt(np.mean((pred - exp) ** 2, axis=-1))


def find_peaks(spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find peak wavelength and signed intensity for each spectrum.

    Peak is defined as argmax of absolute value.
    """
    peak_idxs = np.argmax(np.abs(spectra), axis=-1)
    peak_wl = WAVELENGTHS[peak_idxs]
    peak_int = spectra[np.arange(len(spectra)), peak_idxs]
    return peak_wl, peak_int


def _compute_r2_rmse(exp: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    ss_res = np.sum((exp - pred) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((pred - exp) ** 2))
    return r2, rmse


def plot_error_distribution(
    errors: np.ndarray, save_path: Path, xlabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.hist(errors, bins=20, color="#56B4E9", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(errors), color="#D55E00", linestyle="--", linewidth=1.2, label=f"Mean={np.mean(errors):.2f}")
    ax.axvline(np.median(errors), color="#009E73", linestyle="--", linewidth=1.2, label=f"Median={np.median(errors):.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

    fig.savefig(save_path)
    plt.close(fig)


def plot_peak_parity(
    exp_peak: np.ndarray,
    pred_peak: np.ndarray,
    save_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3))

    r2, rmse = _compute_r2_rmse(exp_peak, pred_peak)

    # Scatter
    ax.scatter(exp_peak, pred_peak, color="#0072B2", s=20, alpha=0.7, edgecolors="white", linewidth=0.5)

    # y=x reference line
    lim_min = min(exp_peak.min(), pred_peak.min())
    lim_max = max(exp_peak.max(), pred_peak.max())
    margin = (lim_max - lim_min) * 0.05
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
            color="#000000", linestyle="--", linewidth=1.0, label="y=x")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Annotation
    text = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    ax.legend(loc="lower right")

    fig.savefig(save_path)
    plt.close(fig)


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"infer_{args.output_tag}_{timestamp}" if args.output_tag else f"infer_{timestamp}"
    output_dir = checkpoint_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print("Loading model...")
    model, device = load_model(str(checkpoint_dir), args.device)
    spectrum_scale = model.spectrum_scale
    print(f"Model loaded on {device}")
    print(f"Asinh spectrum scale={spectrum_scale:.4f}")

    # Load data
    print("Loading data...")
    raw_data, keys = load_data_and_keys(args.data_path, args.keys_txt)
    print(f"Loaded {len(keys)} samples from {args.keys_txt}")

    # Compute normalization params from all data
    color_map, gray_min, gray_max, thickness_min, thickness_max, stretch_min, stretch_max = \
        compute_normalization_params(raw_data)

    # Build tensors
    color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx, spectra_transformed = build_tensors(
        raw_data, keys, color_map,
        gray_min, gray_max, thickness_min, thickness_max, stretch_min, stretch_max,
        spectrum_scale,
    )

    # Inference
    print("Running inference...")
    t0 = time.time()
    pred_transformed = run_inference(
        model, color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx,
        args.batch_size, device,
    )
    print(f"Inference done in {time.time() - t0:.2f}s")

    # Inverse asinh transform
    pred = np.sinh(pred_transformed) * spectrum_scale
    exp = np.sinh(spectra_transformed.numpy()) * spectrum_scale

    # Metrics
    rmse_per_sample = compute_rmse(exp, pred)
    overall_rmse = float(np.mean(rmse_per_sample))
    print(f"Overall RMSE: {overall_rmse:.6f}")

    # Peak finding
    exp_peak_wl, exp_peak_int = find_peaks(exp)
    pred_peak_wl, pred_peak_int = find_peaks(pred)

    # Save inference results
    np.savez(
        output_dir / "inference.npz",
        keys=np.array(keys, dtype=object),
        exp_spectra=exp,
        pred_spectra=pred,
        exp_peak_wl=exp_peak_wl,
        exp_peak_int=exp_peak_int,
        pred_peak_wl=pred_peak_wl,
        pred_peak_int=pred_peak_int,
        rmse_per_sample=rmse_per_sample,
    )
    print(f"Saved inference.npz")

    # Plot peak intensity error distribution
    int_error = np.abs(pred_peak_int - exp_peak_int)
    plot_error_distribution(
        int_error, output_dir / "peak_intensity_error_distribution.png",
        xlabel="|Pred - Exp| Peak Intensity (mdeg)",
    )
    print(f"Saved peak_intensity_error_distribution.png")

    # Plot peak wavelength error distribution
    wl_error = np.abs(pred_peak_wl - exp_peak_wl)
    plot_error_distribution(
        wl_error, output_dir / "peak_wavelength_error_distribution.png",
        xlabel="|Pred - Exp| Peak Wavelength (nm)",
    )
    print(f"Saved peak_wavelength_error_distribution.png")

    # Plot peak intensity parity
    plot_peak_parity(
        exp_peak_int, pred_peak_int,
        output_dir / "peak_intensity_parity.png",
        xlabel="Exp Peak Intensity (mdeg)",
        ylabel="Pred Peak Intensity (mdeg)",
    )
    print(f"Saved peak_intensity_parity.png")

    # Plot peak wavelength parity
    plot_peak_parity(
        exp_peak_wl, pred_peak_wl,
        output_dir / "peak_wavelength_parity.png",
        xlabel="Exp Peak Wavelength (nm)",
        ylabel="Pred Peak Wavelength (nm)",
    )
    print(f"Saved peak_wavelength_parity.png")

    print(f"\n=== Inference finished ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
