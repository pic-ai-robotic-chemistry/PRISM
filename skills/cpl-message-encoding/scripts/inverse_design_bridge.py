"""Resolve the material parameters required by a message's CPL layout."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cpl_codec import (
    DESIGN_ID_BY_PAIR,
    INTENSITY_INTERVAL_LABELS,
    INTENSITY_INTERVAL_NAMES,
    IS_RED_REQUIRED,
    MESSAGE_LIBRARY_TYPE,
    MODEL_PREDICTION_SOURCE,
    ProtocolError,
    classify_intensity,
    protocol_metadata,
)
from create_config_library import build_catalog, canonical_hash, file_sha256, read_dataset


TARGET_INTENSITIES = (-6500, -2500, 2500, 6500)


def build_message_library(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    inverse_skill_dir: Path,
    measurements: Path | None,
    device: str,
    stretch_steps: int,
) -> dict[str, Any]:
    counts = Counter(pair for row in rows for pair in zip(row["colour_slots"], row["intensity_slots"]))
    targets = {
        pair: {
            "design_id": DESIGN_ID_BY_PAIR[pair],
            "colour_symbol": pair[0],
            "intensity_digit": pair[1],
            "intensity_interval": INTENSITY_INTERVAL_LABELS[pair[1]],
            "target_peak_intensity_mdeg": TARGET_INTENSITIES[pair[1]],
            "device_count": counts[pair],
            "status": "pending",
        }
        for pair in sorted(counts)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "design_targets.json"
    report = {"profile": "red", "is_red": IS_RED_REQUIRED, "targets": list(targets.values())}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    measured_rows = read_dataset(measurements.expanduser().resolve()) if measurements else []
    measured_pairs = {(row["color"], row["intensity_digit"]) for row in measured_rows if row["is_red"]}
    catalog = build_catalog(measured_rows, minimum_margin=500.0, required_pairs=measured_pairs & targets.keys())
    for target in targets.values():
        if target["design_id"] in catalog:
            target["status"] = "measured"

    pending = {pair: target for pair, target in targets.items() if target["status"] == "pending"}
    if pending:
        scripts = inverse_skill_dir.expanduser().resolve() / "scripts"
        if not (scripts / "inverse_design.py").is_file():
            raise ProtocolError(f"Cannot find the inverse-design skill at {scripts.parent}.")
        sys.path.insert(0, str(scripts))
        try:
            from cpl_profile import RED_PHASE
            from inverse_design import evaluate_process_grid, load_model
        except ImportError as exc:
            raise ProtocolError(f"Cannot load the red-phase cof-inverse-design skill: {exc}") from exc
        finally:
            sys.path.pop(0)

        weights = RED_PHASE.checkpoint
        model = load_model(device)
        report["model_checkpoint"] = {"name": weights.name, "sha256": file_sha256(weights)}
        report["stretch_steps"] = stretch_steps

        for color in sorted({pair[0] for pair in pending}):
            peaks = evaluate_process_grid(model, color, stretch_steps=stretch_steps)
            by_digit: dict[int, list[dict[str, float]]] = {digit: [] for digit in range(len(TARGET_INTENSITIES))}
            for peak in peaks:
                digit = classify_intensity(peak["peak_intensity_mdeg"], location=f"{color} model prediction")
                by_digit[digit].append(peak)
            intensities = [peak["peak_intensity_mdeg"] for peak in peaks]
            for digit, candidates in by_digit.items():
                target = pending.get((color, digit))
                if target is None:
                    continue
                if not candidates:
                    target["status"] = "no_candidate_in_interval"
                    target["scanned_peak_intensity_range_mdeg"] = [min(intensities), max(intensities)]
                    continue
                preferred = target["target_peak_intensity_mdeg"]
                peak = min(candidates, key=lambda item: abs(item["peak_intensity_mdeg"] - preferred))
                design_id = target["design_id"]
                catalog[design_id] = {
                    "design_id": design_id,
                    "colour_symbol": color,
                    "intensity_digit": digit,
                    "intensity_interval": {
                        "name": INTENSITY_INTERVAL_NAMES[digit],
                        "label": INTENSITY_INTERVAL_LABELS[digit],
                    },
                    "material_parameters": {
                        "color": color,
                        "gray": peak["gray"],
                        "thickness_nm": peak["thickness"],
                        "stretch": peak["stretch"],
                        "is_red": IS_RED_REQUIRED,
                    },
                    "observables": {
                        "colour_symbol": color,
                        "peak_cd_intensity": peak["peak_intensity_mdeg"],
                        "peak_wavelength_nm": peak["peak_wavelength_nm"],
                        "source": MODEL_PREDICTION_SOURCE,
                    },
                }
                target["status"] = "model_candidate"

    missing = []
    for target in targets.values():
        design_id = target["design_id"]
        if design_id in catalog:
            target["selected_configuration"] = catalog[design_id]
        else:
            missing.append(design_id)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise ProtocolError(
            f"No qualifying candidate for {', '.join(missing)} in the model grid. "
            f"See {report_path}. Supply matching measured rows with --measurements."
        )

    library = {
        "library_type": MESSAGE_LIBRARY_TYPE,
        "protocol": protocol_metadata(),
        "design_catalog": catalog,
    }
    return {**library, "library_id": f"sha256:{canonical_hash(library)}"}
