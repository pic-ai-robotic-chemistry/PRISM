"""Run the red-phase CPL search for four wavelength targets and three colors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from cpl_profile import RED_PHASE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVERSE_DESIGN = PROJECT_ROOT / "scripts" / "inverse_design.py"

TARGETS = [
    (475.0, 5000.0),
    (525.0, 5000.0),
    (575.0, 5000.0),
    (625.0, 5000.0),
]


COLORS = ["R", "G", "B"]


def intensity_tag(value: float) -> str:
    prefix = "p" if value >= 0 else "m"
    return f"{prefix}{int(abs(value))}mdeg"


def target_tag(wavelength_nm: float, intensity_mdeg: float) -> str:
    return f"target_{int(wavelength_nm)}nm_{intensity_tag(intensity_mdeg)}"


def subprocess_env() -> dict:
    """Cap BLAS/OpenMP threads for the child process.

    Each run is dominated by skopt's single-threaded Python work, so extra
    threads buy nothing and cost a lot: with PyTorch's default thread count,
    N concurrent runs each claim every core on the machine, which degrades
    every one of them. One thread per run keeps N runs genuinely parallel.
    """
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    return env


def read_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_normalization(normalization: dict | None) -> str:
    """Render the audit ranges compactly so they survive both CSV and JSON."""
    if not normalization:
        return ""
    return " ".join(
        f"{param}[{bounds[0]},{bounds[1]}]"
        for param, bounds in normalization.items()
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    )


def flatten_result(target_id: str, color: str, output_dir: Path, result: dict) -> dict:
    target_wl = float(result["target_wavelength_nm"])
    target_int = float(result["target_intensity_mdeg"])
    pred_wl = float(result["predicted_peak_wavelength_nm"])
    pred_int = float(result["predicted_peak_intensity_mdeg"])
    return {
        "target_id": target_id,
        "target_wavelength_nm": target_wl,
        "target_intensity_mdeg": target_int,
        "color": color,
        "is_red": int(result["is_red"]),
        "profile": result.get("profile"),
        "checkpoint": result.get("checkpoint_path") or result.get("checkpoint"),
        "normalization": format_normalization(result.get("normalization")),
        "optimal_gray": float(result["optimal_gray"]),
        "optimal_thickness_nm": float(result["optimal_thickness"]),
        "thickness_choices_nm": ",".join(str(v) for v in result.get("thickness_choices_nm", [])),
        "optimal_stretch": float(result["optimal_stretch"]),
        "predicted_peak_wavelength_nm": pred_wl,
        "predicted_peak_intensity_mdeg": pred_int,
        "wavelength_error_nm": pred_wl - target_wl,
        "intensity_error_mdeg": pred_int - target_int,
        "abs_wavelength_error_nm": abs(pred_wl - target_wl),
        "abs_intensity_error_mdeg": abs(pred_int - target_int),
        "best_objective": float(result["best_objective"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "n_calls": int(result["n_calls"]),
        "n_random_starts": int(result["n_random_starts"]),
        "acq_func": result["acq_func"],
        "skopt_n_jobs": int(result.get("skopt_n_jobs", 0)),
        "intensity_weight": float(result["intensity_weight"]),
        "output_dir": str(output_dir),
        "status": "ok",
        "gray_choices": ",".join(str(v) for v in result.get("gray_choices", [])),
    }


def job_directory(output_root: Path, target_id: str, color: str) -> Path:
    return output_root / f"profile_{RED_PHASE.name}" / target_id / f"color_{color}"


def existing_result(
    target: tuple[float, float], color: str, args: argparse.Namespace, output_root: Path
) -> dict | None:
    tid = target_tag(*target)
    run_dir = job_directory(output_root, tid, color)
    result_path = run_dir / "results.json"
    if not result_path.is_file():
        return None

    result = read_results(result_path)
    recorded_checkpoint = result.get("checkpoint_path") or result.get("checkpoint")
    expected = {
        "profile": RED_PHASE.name,
        "is_red": RED_PHASE.is_red,
        "color": color,
        "target_wavelength_nm": target[0],
        "target_intensity_mdeg": target[1],
        "n_calls": args.n_calls,
        "n_random_starts": args.n_random_starts,
        "acq_func": args.acq_func,
        "skopt_n_jobs": args.skopt_n_jobs,
        "intensity_weight": args.intensity_weight,
    }
    if (
        any(result.get(key) != value for key, value in expected.items())
        or not recorded_checkpoint
        or Path(recorded_checkpoint).resolve() != RED_PHASE.checkpoint.resolve()
    ):
        raise ValueError(f"{result_path} uses a different configuration. Choose a new --output-root.")
    output_files = ("convergence.png", "best_spectrum.png", "parameter_space_3d.png")
    if not all((run_dir / name).is_file() for name in output_files):
        return None
    return flatten_result(tid, color, run_dir, result)


def run_design(target: tuple[float, float], color: str, args: argparse.Namespace, output_root: Path) -> tuple[str, dict]:
    wavelength_nm, intensity_mdeg = target
    tid = target_tag(wavelength_nm, intensity_mdeg)
    run_dir = job_directory(output_root, tid, color)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(INVERSE_DESIGN),
        "--color",
        color,
        "--target_wavelength",
        str(wavelength_nm),
        "--target_intensity",
        str(intensity_mdeg),
        "--intensity_weight",
        str(args.intensity_weight),
        "--n_calls",
        str(args.n_calls),
        "--n_random_starts",
        str(args.n_random_starts),
        "--acq_func",
        args.acq_func,
        "--device",
        args.device,
        "--skopt_n_jobs",
        str(args.skopt_n_jobs),
        "--output-dir",
        str(run_dir),
        "--json",
    ]

    print(f"Running {tid} color={color} -> {run_dir}", flush=True)

    log_path = run_dir / "run.log"
    env = subprocess_env()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            cmd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        returncode = process.wait()

    if returncode != 0:
        return "failure", {
            "target_id": tid,
            "color": color,
            "returncode": returncode,
            "output_dir": str(run_dir),
        }

    result_path = run_dir / "results.json"
    if not result_path.exists():
        return "failure", {
            "target_id": tid,
            "color": color,
            "returncode": returncode,
            "output_dir": str(run_dir),
            "error": "results.json missing",
        }

    return "success", flatten_result(tid, color, run_dir, read_results(result_path))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, rows: list[dict], recommendations: list[dict]) -> None:
    lines = [
        "# CPL Material Design Batch Summary",
        "",
        "Selection rule: lowest `best_objective` among R/G/B for the same target.",
        f"All runs use the red phase (`is_red={RED_PHASE.is_red}`).",
        "",
        "## Recommendations",
        "",
        "| Target | Color | gray | thickness (nm) | stretch | Pred WL (nm) | Pred Int (mdeg) | Objective | Output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in recommendations:
        lines.append(
            "| {target_id} | {color} | {optimal_gray:.4f} | {optimal_thickness_nm:.4f} | "
            "{optimal_stretch:.4f} | {predicted_peak_wavelength_nm:.2f} | "
            "{predicted_peak_intensity_mdeg:.2f} | {best_objective:.4f} | {output_dir} |".format(**r)
        )
    lines.extend([
        "",
        "## All Runs",
        "",
        "| Target | Color | WL err (nm) | Int err (mdeg) | Objective | Output |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for r in rows:
        lines.append(
            "| {target_id} | {color} | {wavelength_error_nm:.2f} | "
            "{intensity_error_mdeg:.2f} | {best_objective:.4f} | {output_dir} |".format(**r)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run red-phase CPL inverse design for B/G/R at 475, 525, 575, and 625 nm using assets/best.ckpt."
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Root directory for all batch outputs.")
    parser.add_argument("--n_calls", type=int, default=100, help="Bayesian optimization calls per run.")
    parser.add_argument("--n_random_starts", type=int, default=20, help="Random starts per run.")
    parser.add_argument("--intensity_weight", type=float, default=1e-2, help="Intensity term weight.")
    parser.add_argument("--device", default="auto", help="Device passed to inverse_design.py.")
    parser.add_argument("--acq_func", default="EI", choices=["LCB", "EI", "PI"], help="Acquisition function.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent inverse-design subprocesses.")
    parser.add_argument("--skopt_n_jobs", type=int, default=1, help="n_jobs passed to each skopt.gp_minimize.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root
        or PROJECT_ROOT
        / "outputs"
        / f"material_design_{RED_PHASE.name}_rgb_discrete_gray_thickness_wl475_625_5000mdeg_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []

    jobs: list[tuple[tuple[float, float], str]] = []
    for target in TARGETS:
        for color in COLORS:
            try:
                row = existing_result(target, color, args, output_root)
            except ValueError as exc:
                print(f"Batch configuration conflict: {exc}", file=sys.stderr)
                return 2
            if row is None:
                jobs.append((target, color))
            else:
                rows.append(row)
                print(f"SKIP existing {row['target_id']} color={color}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_job = {
            executor.submit(run_design, target, color, args, output_root): (target, color)
            for target, color in jobs
        }
        for future in as_completed(future_to_job):
            target, color = future_to_job[future]
            tid = target_tag(*target)
            try:
                status, payload = future.result()
            except Exception as exc:
                payload = {
                    "target_id": tid,
                    "color": color,
                    "output_dir": str(job_directory(output_root, tid, color)),
                    "error": repr(exc),
                }
                failures.append(payload)
                print(f"FAILED {tid} color={color}: {exc!r}", flush=True)
                continue

            if status == "success":
                rows.append(payload)
                print(f"DONE {tid} color={color} objective={payload['best_objective']:.4f}", flush=True)
            else:
                failures.append(payload)
                print(f"FAILED {tid} color={color} returncode={payload.get('returncode')}", flush=True)

    target_order = {target_tag(*target): idx for idx, target in enumerate(TARGETS)}
    color_order = {color: idx for idx, color in enumerate(COLORS)}
    rows.sort(key=lambda r: (target_order[r["target_id"]], color_order[r["color"]]))

    recommendations: list[dict] = []
    for wavelength_nm, intensity_mdeg in TARGETS:
        tid = target_tag(wavelength_nm, intensity_mdeg)
        candidates = [r for r in rows if r["target_id"] == tid]
        if candidates:
            recommendations.append(min(candidates, key=lambda r: r["best_objective"]))

    write_csv(output_root / "all_results.csv", rows)
    write_csv(output_root / "recommendations.csv", recommendations)
    (output_root / "all_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "recommendations.json").write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
    (output_root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    write_markdown_summary(output_root / "summary.md", rows, recommendations)

    print(f"Batch output root: {output_root}", flush=True)
    print(f"Successful runs: {len(rows)}", flush=True)
    print(f"Failed runs: {len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
