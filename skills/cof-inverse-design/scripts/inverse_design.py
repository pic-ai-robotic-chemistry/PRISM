"""Red-phase Bayesian inverse design over thickness and stretch."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Categorical, Real

from cpl_model import CPLInferenceModel
from cpl_profile import RED_PHASE, normalize

# Limit CPU threads per worker to avoid oversubscription during batch searches.
torch.set_num_threads(max(1, int(os.environ.get("CPL_NUM_THREADS", "1"))))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # already initialised; nothing to do
    pass

# Scientific visualization style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

WAVELENGTHS = np.linspace(800, 400, 801)

# Used to bound the in-window loss so a wavelength window can be enforced by
# ordering. Any fixed value far above what the model actually emits works; the
# network's peaks top out near 12 000 mdeg. The margin keeps the ordering
# strict rather than merely non-strict at the bound.
INTENSITY_CEILING_MDEG = 1.0e5
WINDOW_CEILING_MARGIN = 2.0

COLOR_MAP = {"B": 0, "G": 1, "R": 2}


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Red-phase CPL inverse design with the bundled assets/best.ckpt."
    )
    parser.add_argument(
        "--color", required=True, choices=["B", "G", "R"],
        help="Fixed color parameter",
    )
    parser.add_argument(
        "-t", "--target_wavelength", type=float, default=None,
        help="Target peak wavelength in nm (e.g. 550). Optional if --target_intensity is set.",
    )
    parser.add_argument(
        "--target_intensity", type=float, default=None,
        help="Target peak intensity in mdeg (e.g. -1000). Optional if -t is set.",
    )
    parser.add_argument(
        "--intensity_weight", type=float, default=1e-2,
        help="Weight for intensity term when both targets are set (default: 1e-2)",
    )
    parser.add_argument(
        "--wl-window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
        help=(
            "Constrain the predicted peak wavelength to [LO, HI] nm. Candidates outside "
            "the window are penalized so the search prioritizes candidates inside it. Use when maximising |intensity| "
            "alone would otherwise let the peak drift out of the colour of interest."
        ),
    )
    parser.add_argument(
        "--n_calls", type=int, default=100,
        help="Number of Bayesian optimization iterations (default: 100)",
    )
    parser.add_argument(
        "--n_random_starts", type=int, default=20,
        help="Number of random initialization points (default: 20)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for inference: cuda/cpu/auto (default: auto)",
    )
    parser.add_argument(
        "--acq_func", default="EI",
        choices=["LCB", "EI", "PI"],
        help="Acquisition function: LCB, EI, or PI (default: EI)",
    )
    parser.add_argument(
        "--skopt_n_jobs", type=int, default=-1,
        help="n_jobs passed to skopt.gp_minimize (default: -1)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to inverse_discrete_gray_red_{color}_{wl}_{int}_{timestamp} under CWD.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print compact JSON summary to stdout at the end",
    )
    args = parser.parse_args(argv)
    if args.target_wavelength is None and args.target_intensity is None:
        parser.error("At least one of --target_wavelength (-t) or --target_intensity must be specified.")
    if args.wl_window is not None:
        lo, hi = args.wl_window
        if not lo < hi:
            parser.error(f"--wl-window LO HI requires LO < HI, got {lo} {hi}")
        if lo < WAVELENGTHS.min() or hi > WAVELENGTHS.max():
            parser.error(
                f"--wl-window {lo} {hi} lies outside the modelled range "
                f"{WAVELENGTHS.min():.0f}-{WAVELENGTHS.max():.0f} nm"
            )
    return args


def load_model(device: str) -> CPLInferenceModel:
    ckpt_path = RED_PHASE.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading model from {ckpt_path} ...")
    model = CPLInferenceModel.from_checkpoint(ckpt_path, device=device)
    print(f"Model loaded on {next(model.parameters()).device}")
    print(f"Spectrum transform: {model.spectrum_transform}, scale={model.spectrum_scale:.4f}")
    return model


def make_progress_callback(n_calls: int, started_at: float, label: str = ""):
    """One compact, flushed progress line per optimisation round.

    skopt's own ``verbose=True`` emits three lines per iteration, which buries
    the one number that matters: how many rounds are left. The ETA assumes a
    constant per-round cost, so treat it as a floor -- the Gaussian-process fit
    gets more expensive as observations accumulate.
    """

    def callback(result) -> None:
        done = len(result.func_vals)
        elapsed = time.time() - started_at
        per_round = elapsed / done if done else 0.0
        eta = per_round * max(0, n_calls - done)
        print(
            f"[progress] {label}{done}/{n_calls} "
            f"best={result.fun:.6g} elapsed={elapsed:.0f}s eta>={eta:.0f}s",
            flush=True,
        )

    return callback


def build_discrete_search_space():
    """Search surveyed thickness values and continuous stretch in the red phase."""
    space = [
        Categorical(list(RED_PHASE.thickness_choices), name="thickness"),
        Real(RED_PHASE.stretch_norm.min, RED_PHASE.stretch_norm.max, name="stretch"),
    ]
    return space, ["thickness", "stretch"]


def expand_params(param_names: list, values: list) -> dict:
    """Turn a skopt point into a full {gray, thickness, stretch} dict."""
    params = {name: float(value) for name, value in zip(param_names, values)}
    params["gray"] = RED_PHASE.gray_fixed
    return params


@torch.no_grad()
def evaluate_process_grid(
    model: CPLInferenceModel, color: str, *, stretch_steps: int
) -> list[dict[str, float]]:
    """Evaluate red-phase signed CD peaks at sampled thickness and stretch values."""
    if stretch_steps < 2:
        raise ValueError("stretch_steps must be at least 2.")
    device = next(model.parameters()).device
    names = ("gray", "thickness", "stretch")
    stretches = np.linspace(RED_PHASE.stretch_norm.min, RED_PHASE.stretch_norm.max, stretch_steps)
    combinations = list(product((RED_PHASE.gray_fixed,), RED_PHASE.thickness_choices, stretches))
    peaks = []
    for start in range(0, len(combinations), 128):
        chunk = combinations[start : start + 128]
        normalized = torch.tensor(
            [[normalize(name, float(value)) for name, value in zip(names, combo)] for combo in chunk],
            device=device,
            dtype=torch.float32,
        )
        spectra = model.predict_spectrum(
            torch.full((len(chunk),), COLOR_MAP[color], device=device, dtype=torch.long),
            *normalized.split(1, dim=1),
            torch.full((len(chunk),), RED_PHASE.is_red, device=device, dtype=torch.long),
        )
        indices = spectra.abs().argmax(dim=1, keepdim=True)
        intensities = spectra.gather(1, indices).flatten().tolist()
        for combo, index, intensity in zip(chunk, indices.flatten().tolist(), intensities):
            peaks.append({
                **dict(zip(names, map(float, combo))),
                "peak_wavelength_nm": float(WAVELENGTHS[index]),
                "peak_intensity_mdeg": intensity,
            })
    return peaks


def window_ceiling(target_wl, target_int, intensity_weight) -> float:
    """A score strictly worse than any candidate inside the wavelength window.

    Bounding the in-window loss lets the window be enforced by ordering rather
    than by a hand-tuned penalty weight. A penalty strong enough to dominate a
    large intensity term would make the surrogate landscape a cliff; instead
    every out-of-window point scores above every in-window point, and the
    squared excursion added on top still gives the surrogate a gradient that
    points back toward the window.
    """
    ceiling = 0.0
    if target_wl is not None:
        ceiling += float((WAVELENGTHS.max() - WAVELENGTHS.min()) ** 2)
    if target_int is not None:
        reach = abs(float(target_int)) + INTENSITY_CEILING_MDEG
        ceiling += float(intensity_weight) * reach ** 2
    return WINDOW_CEILING_MARGIN * ceiling


@torch.no_grad()
def make_objective(
    model: CPLInferenceModel,
    color: str,
    param_names: list,
    target_wl: float | None,
    target_int: float | None,
    intensity_weight: float,
    device: str,
    wl_window: tuple[float, float] | None = None,
):
    """Build objective function for skopt. Also records optimization history."""
    color_idx = torch.tensor([COLOR_MAP[color]], dtype=torch.long, device=device)
    is_red_idx = torch.tensor([RED_PHASE.is_red], dtype=torch.long, device=device)
    ceiling = window_ceiling(target_wl, target_int, intensity_weight) if wl_window else None

    history = []

    def objective(values: list) -> float:
        params = expand_params(param_names, values)
        gray = params["gray"]
        thickness = params["thickness"]
        stretch = params["stretch"]

        gray_norm = torch.tensor([[normalize("gray", gray)]], dtype=torch.float32, device=device)
        thickness_norm = torch.tensor([[normalize("thickness", thickness)]], dtype=torch.float32, device=device)
        stretch_norm = torch.tensor([[normalize("stretch", stretch)]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_raw = model.predict_spectrum(color_idx, gray_norm, thickness_norm, stretch_norm, is_red_idx)
        pred = pred_raw.squeeze().detach().cpu().numpy()

        peak_idx = int(np.argmax(np.abs(pred)))
        peak_wl = float(WAVELENGTHS[peak_idx])
        peak_int = float(pred[peak_idx])

        loss = 0.0
        if target_wl is not None:
            loss += (peak_wl - target_wl) ** 2
        if target_int is not None:
            loss += intensity_weight * (peak_int - target_int) ** 2

        in_window = True
        if wl_window is not None:
            lo, hi = wl_window
            in_window = lo <= peak_wl <= hi
            if not in_window:
                excursion = (lo - peak_wl) if peak_wl < lo else (peak_wl - hi)
                loss = ceiling + excursion ** 2

        history.append({
            "gray": float(gray),
            "thickness": float(thickness),
            "stretch": float(stretch),
            "peak_wl": peak_wl,
            "peak_int": peak_int,
            "objective": loss,
            "in_window": bool(in_window),
        })

        return loss

    return objective, history


def main():
    args = parse_args()
    profile = RED_PHASE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wl_tag = f"{int(args.target_wavelength)}nm" if args.target_wavelength is not None else "none"
    int_tag = f"{int(args.target_intensity)}int" if args.target_intensity is not None else "none"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path.cwd() / (
            f"inverse_discrete_gray_{profile.name}_{args.color}_{wl_tag}_{int_tag}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    model = load_model(args.device)
    device = str(next(model.parameters()).device)

    gray_choices = [profile.gray_fixed]
    thickness_choices = list(profile.thickness_choices)
    space, param_names = build_discrete_search_space()

    objective, history = make_objective(
        model,
        args.color,
        param_names,
        args.target_wavelength,
        args.target_intensity,
        args.intensity_weight,
        device,
        wl_window=tuple(args.wl_window) if args.wl_window else None,
    )

    print(f"\nStarting Bayesian Optimization...")
    if args.target_wavelength is not None:
        print(f"  Target wavelength: {args.target_wavelength} nm")
    if args.target_intensity is not None:
        print(f"  Target intensity: {args.target_intensity} mdeg (weight={args.intensity_weight})")
    print(f"  Fixed color: {args.color}, is_red: {profile.is_red}")
    print(
        f"  Search space: gray fixed at {profile.gray_fixed}, "
        f"thickness choices {thickness_choices}, "
        f"stretch {profile.stretch_norm.min}-{profile.stretch_norm.max}"
    )
    print(f"  n_calls: {args.n_calls}, n_random_starts: {args.n_random_starts}, acq_func: {args.acq_func}")

    t0 = time.time()
    result = gp_minimize(
        objective,
        space,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts,
        acq_func=args.acq_func,
        verbose=False,
        n_jobs=args.skopt_n_jobs,
        random_state=42,
        callback=make_progress_callback(args.n_calls, t0),
    )
    elapsed = time.time() - t0

    best_params = expand_params(param_names, result.x)
    best_gray = best_params["gray"]
    best_thickness = best_params["thickness"]
    best_stretch = best_params["stretch"]
    best_loss = float(result.fun)

    best_gray_norm = torch.tensor([[normalize("gray", best_gray)]], dtype=torch.float32, device=device)
    best_thickness_norm = torch.tensor([[normalize("thickness", best_thickness)]], dtype=torch.float32, device=device)
    best_stretch_norm = torch.tensor([[normalize("stretch", best_stretch)]], dtype=torch.float32, device=device)
    color_idx = torch.tensor([COLOR_MAP[args.color]], dtype=torch.long, device=device)
    is_red_idx = torch.tensor([profile.is_red], dtype=torch.long, device=device)
    with torch.no_grad():
        best_spectrum_raw = model.predict_spectrum(
            color_idx, best_gray_norm, best_thickness_norm, best_stretch_norm, is_red_idx
        )
    best_spectrum = best_spectrum_raw.squeeze().detach().cpu().numpy()
    best_peak_idx = int(np.argmax(np.abs(best_spectrum)))
    best_peak_wl = float(WAVELENGTHS[best_peak_idx])
    best_peak_int = float(best_spectrum[best_peak_idx])

    print(f"\n=== Optimization finished in {elapsed:.1f}s ===")
    print(f"Best objective: {best_loss:.4f}")
    print(f"Optimal parameters:")
    print(f"  gray       = {best_gray:.4f}")
    print(f"  thickness  = {best_thickness:.4f}")
    print(f"  stretch    = {best_stretch:.4f}")
    print(f"Predicted peak wavelength: {best_peak_wl:.2f} nm")
    print(f"Predicted peak intensity:  {best_peak_int:.2f} mdeg")

    results = {
        **profile.to_json(),
        "checkpoint_path": str(profile.checkpoint),
        "spectrum_transform": getattr(model, "spectrum_transform", "asinh"),
        "spectrum_scale": getattr(model, "spectrum_scale", None),
        "color": args.color,
        "thickness_choices_nm": thickness_choices,
        "searched_params": param_names,
        "target_wavelength_nm": args.target_wavelength,
        "target_intensity_mdeg": args.target_intensity,
        "intensity_weight": args.intensity_weight,
        "wl_window_nm": list(args.wl_window) if args.wl_window else None,
        "optimal_gray": best_gray,
        "optimal_thickness": best_thickness,
        "optimal_stretch": best_stretch,
        "best_objective": best_loss,
        "predicted_peak_wavelength_nm": best_peak_wl,
        "predicted_peak_intensity_mdeg": best_peak_int,
        "n_calls": args.n_calls,
        "n_random_starts": args.n_random_starts,
        "acq_func": args.acq_func,
        "skopt_n_jobs": args.skopt_n_jobs,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    result_path = output_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {result_path}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(result.func_vals, color="#0072B2", linewidth=1.2, alpha=0.7, label="Objective value")
    ax.plot(np.minimum.accumulate(result.func_vals), color="#D55E00", linewidth=1.5, label="Best so far")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    conv_path = output_dir / "convergence.png"
    fig.savefig(conv_path)
    plt.close(fig)
    print(f"Saved convergence plot to: {conv_path}")

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(WAVELENGTHS, best_spectrum, color="#0072B2", linewidth=1.2, label="Predicted")
    ax.axvline(best_peak_wl, color="#D55E00", linestyle="--", linewidth=1.0, label=f"Peak={best_peak_wl:.1f} nm")
    if args.target_wavelength is not None:
        ax.axvline(
            args.target_wavelength,
            color="#009E73",
            linestyle="--",
            linewidth=1.0,
            label=f"Target WL={args.target_wavelength:.1f} nm",
        )
    if args.target_intensity is not None:
        ax.axhline(
            args.target_intensity,
            color="#CC79A7",
            linestyle="--",
            linewidth=1.0,
            label=f"Target Int={args.target_intensity:.1f} mdeg",
        )
        ax.plot(best_peak_wl, best_peak_int, "o", color="red", markersize=6, label=f"Peak Int={best_peak_int:.1f} mdeg")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("CD (mdeg)")
    ax.legend(loc="best")
    spec_path = output_dir / "best_spectrum.png"
    fig.savefig(spec_path)
    plt.close(fig)
    print(f"Saved best spectrum plot to: {spec_path}")

    grays = [h["gray"] for h in history]
    thicknesses = [h["thickness"] for h in history]
    stretches = [h["stretch"] for h in history]
    fig = plt.figure(figsize=(6, 5))
    ax3d = fig.add_subplot(111, projection="3d")
    iterations = np.arange(len(grays))
    scatter = ax3d.scatter(
        grays,
        thicknesses,
        stretches,
        c=iterations,
        cmap="plasma",
        s=50,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.4,
        zorder=3,
    )
    ax3d.plot(
        [best_gray],
        [best_thickness],
        [best_stretch],
        marker="*",
        markersize=15,
        color="red",
        markeredgecolor="black",
        markeredgewidth=0.8,
        linestyle="none",
        label="Best",
        zorder=20,
    )
    ax3d.set_xlabel("Gray")
    ax3d.set_ylabel("Thickness")
    ax3d.set_zlabel("Stretch")
    cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.1)
    cbar.set_label("Iteration")
    ax3d.legend(loc="upper left")
    ax3d.set_xlim(profile.gray_norm.min, profile.gray_norm.max)
    ax3d.set_xticks(gray_choices)
    ax3d.set_ylim(profile.thickness_norm.min, profile.thickness_norm.max)
    ax3d.set_yticks(thickness_choices)
    ax3d.set_zlim(profile.stretch_norm.min, profile.stretch_norm.max)
    space3d_path = output_dir / "parameter_space_3d.png"
    fig.savefig(space3d_path)
    plt.close(fig)
    print(f"Saved 3D parameter space plot to: {space3d_path}")

    if args.json:
        summary = {
            "success": True,
            "output_dir": str(output_dir),
            "optimal": {
                "gray": round(best_gray, 4),
                "thickness": round(best_thickness, 4),
                "stretch": round(best_stretch, 4),
            },
            "predicted": {
                "peak_wavelength_nm": round(best_peak_wl, 2),
                "peak_intensity_mdeg": round(best_peak_int, 2),
            },
            "target": {
                "wavelength_nm": args.target_wavelength,
                "intensity_mdeg": args.target_intensity,
            },
            "metrics": {
                "best_objective": round(best_loss, 4),
                "elapsed_seconds": round(elapsed, 2),
                "n_calls": args.n_calls,
            },
        }
        print("\n--- JSON_SUMMARY ---")
        print(json.dumps(summary, ensure_ascii=False))
        print("--- END_JSON_SUMMARY ---")


if __name__ == "__main__":
    main()
