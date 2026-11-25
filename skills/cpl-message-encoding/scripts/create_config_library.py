#!/usr/bin/env python3
"""Build the reusable quaternary CPL library from measured peak CSV rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cpl_codec import (
    DESIGN_ID_BY_PAIR,
    INTENSITY_INTERVAL_LABELS,
    INTENSITY_INTERVAL_NAMES,
    PAIR_BY_DESIGN_ID,
    ProtocolError,
    classify_intensity,
    protocol_metadata,
    validate_library,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SKILL_ROOT / "outputs" / "dataset_quaternary_red_library"
REQUIRED_COLUMNS = {
    "color",
    "gray",
    "thickness",
    "stretch",
    "is_red",
    "peak_cd_intensity",
    "peak_wavelength",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one measured red-phase configuration for each of the twelve "
            "colour/intensity pairs."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True, help="CSV of measured CD peaks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--minimum-boundary-margin-mdeg",
        type=float,
        default=500.0,
        help="Prefer selected measured peaks at least this far from an interval boundary.",
    )
    return parser.parse_args()


def parse_bool(value: str, *, location: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ProtocolError(f"{location}: is_red must be True or False.")


def read_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                raise ProtocolError(
                    f"Dataset columns must include {sorted(REQUIRED_COLUMNS)}; got {reader.fieldnames}."
                )
            rows: list[dict[str, Any]] = []
            for csv_line, raw in enumerate(reader, start=2):
                try:
                    color = raw["color"]
                    gray = float(raw["gray"])
                    thickness = float(raw["thickness"])
                    stretch = float(raw["stretch"])
                    is_red = parse_bool(raw["is_red"], location=f"CSV line {csv_line}")
                    intensity = float(raw["peak_cd_intensity"])
                    wavelength = float(raw["peak_wavelength"])
                except (TypeError, ValueError) as exc:
                    raise ProtocolError(f"CSV line {csv_line}: non-numeric field.") from exc
                if color not in ("B", "G", "R"):
                    raise ProtocolError(f"CSV line {csv_line}: unsupported color {color!r}.")
                rows.append(
                    {
                        "csv_line": csv_line,
                        "color": color,
                        "gray": gray,
                        "thickness": thickness,
                        "stretch": stretch,
                        "is_red": is_red,
                        "peak_cd_intensity": intensity,
                        "peak_wavelength": wavelength,
                        "intensity_digit": classify_intensity(
                            intensity, location=f"CSV line {csv_line}"
                        ),
                    }
                )
    except OSError as exc:
        raise ProtocolError(f"Cannot read dataset {path}: {exc}") from exc
    if not rows:
        raise ProtocolError(f"Dataset {path} has no rows.")
    return rows


def interval_boundary_margin(intensity: float) -> float:
    digit = classify_intensity(intensity, location="boundary-margin calculation")
    if digit == 0:
        return -5000.0 - intensity
    if digit == 1:
        return min(intensity + 5000.0, -intensity)
    if digit == 2:
        return min(intensity, 5000.0 - intensity)
    return intensity - 5000.0


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "csv_line": row["csv_line"],
        "color": row["color"],
        "gray": row["gray"],
        "thickness": row["thickness"],
        "stretch": row["stretch"],
        "is_red": row["is_red"],
        "peak_cd_intensity": row["peak_cd_intensity"],
        "peak_wavelength": row["peak_wavelength"],
        "intensity_digit": row["intensity_digit"],
    }


def analyze_dataset(rows: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    phase_gray = Counter(("red" if row["is_red"] else "non_red", row["gray"]) for row in rows)
    coverage: dict[str, dict[str, list[int]]] = {}
    for phase_name, phase_value in (("red", True), ("non_red", False)):
        coverage[phase_name] = {}
        for color in "BGR":
            coverage[phase_name][color] = [
                sum(
                    row["is_red"] == phase_value
                    and row["color"] == color
                    and row["intensity_digit"] == digit
                    for row in rows
                )
                for digit in range(4)
            ]

    by_full_parameters: dict[tuple[Any, ...], set[bool]] = defaultdict(set)
    by_partial_parameters: dict[tuple[Any, ...], set[bool]] = defaultdict(set)
    for row in rows:
        by_full_parameters[
            (row["color"], row["gray"], row["thickness"], row["stretch"])
        ].add(row["is_red"])
        by_partial_parameters[
            (row["color"], row["thickness"], row["stretch"])
        ].add(row["is_red"])
    exact_pairs = sum(len(phases) == 2 for phases in by_full_parameters.values())
    partial_pairs = sum(len(phases) == 2 for phases in by_partial_parameters.values())
    if exact_pairs:
        pairing_limitation = "Exact parameter matches take priority. The search uses proxy matches when needed."
    elif partial_pairs:
        pairing_limitation = "Phase comparisons use proxy matches with the same colour, thickness, and stretch."
    else:
        pairing_limitation = "No matching non-red phase records are available for an intensity comparison."
    return {
        "dataset_path": str(dataset_path),
        "row_count": len(rows),
        "peak_definition": "peak_cd_intensity column from the supplied peak CSV",
        "phase_gray_counts": {
            f"{phase}/gray={gray:g}": count
            for (phase, gray), count in sorted(phase_gray.items())
        },
        "coverage_counts_by_digit": coverage,
        "exact_full_parameter_red_non_red_pair_count": exact_pairs,
        "same_color_thickness_stretch_phase_pair_count": partial_pairs,
        "pairing_limitation": pairing_limitation,
    }


def best_phase_proxy(
    red_row: dict[str, Any],
    non_red_by_full: dict[tuple[Any, ...], list[dict[str, Any]]],
    non_red_by_partial: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    full_key = (
        red_row["color"],
        red_row["gray"],
        red_row["thickness"],
        red_row["stretch"],
    )
    exact_candidates = non_red_by_full.get(full_key, [])
    if exact_candidates:
        candidates = exact_candidates
        comparison_kind = "exact_same_color_gray_thickness_stretch"
    else:
        partial_key = (red_row["color"], red_row["thickness"], red_row["stretch"])
        candidates = non_red_by_partial.get(partial_key, [])
        comparison_kind = "proxy_same_color_thickness_stretch_gray_differs"
    if not candidates:
        return None, None
    different_interval = [
        row for row in candidates if row["intensity_digit"] != red_row["intensity_digit"]
    ]
    pool = different_interval or candidates
    return (
        max(
            pool,
            key=lambda row: (
                abs(row["peak_cd_intensity"] - red_row["peak_cd_intensity"]),
                abs(row["intensity_digit"] - red_row["intensity_digit"]),
            ),
        ),
        comparison_kind,
    )


def select_measured_row(
    *,
    color: str,
    digit: int,
    rows: list[dict[str, Any]],
    non_red_by_full: dict[tuple[Any, ...], list[dict[str, Any]]],
    non_red_by_partial: dict[tuple[Any, ...], list[dict[str, Any]]],
    minimum_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["is_red"] and row["color"] == color and row["intensity_digit"] == digit
    ]
    if not candidates:
        raise ProtocolError(f"No measured red-phase candidate for color={color}, digit={digit}.")

    evaluated: list[
        tuple[dict[str, Any], dict[str, Any] | None, str | None, float, float, bool]
    ] = []
    for red_row in candidates:
        proxy, comparison_kind = best_phase_proxy(
            red_row,
            non_red_by_full,
            non_red_by_partial,
        )
        margin = interval_boundary_margin(red_row["peak_cd_intensity"])
        contrast = (
            abs(proxy["peak_cd_intensity"] - red_row["peak_cd_intensity"])
            if proxy is not None
            else -1.0
        )
        intervals_differ = (
            proxy is not None and proxy["intensity_digit"] != red_row["intensity_digit"]
        )
        evaluated.append(
            (red_row, proxy, comparison_kind, margin, contrast, intervals_differ)
        )

    margin_eligible = [item for item in evaluated if item[3] >= minimum_margin]
    pool = margin_eligible or evaluated
    selected = max(
        pool,
        key=lambda item: (
            item[2] == "exact_same_color_gray_thickness_stretch",
            item[5],
            item[4],
            item[3],
        ),
    )
    red_row, proxy, comparison_kind, margin, contrast, intervals_differ = selected
    selection = {
        "candidate_count_in_interval": len(candidates),
        "minimum_requested_boundary_margin_mdeg": minimum_margin,
        "selected_boundary_margin_mdeg": margin,
        "minimum_margin_satisfied": margin >= minimum_margin,
        "selection_priority": [
            "red-phase row lies in required interval",
            "boundary margin meets requested minimum when possible",
            "exact same-parameter red/non-red pair when available",
            "same color/thickness/stretch non-red proxy lies in a different interval",
            "maximum absolute red/non-red proxy intensity difference",
            "maximum remaining boundary margin",
        ],
        "phase_contrast": (
            {
                "comparison_kind": comparison_kind,
                "matched_fields": (
                    ["color", "gray", "thickness", "stretch"]
                    if comparison_kind == "exact_same_color_gray_thickness_stretch"
                    else ["color", "thickness", "stretch"]
                ),
                "unmatched_field": (
                    None
                    if comparison_kind == "exact_same_color_gray_thickness_stretch"
                    else "gray"
                ),
                "red_row": compact_row(red_row),
                "non_red_proxy_row": compact_row(proxy),
                "intensity_difference_mdeg": contrast,
                "intervals_differ": intervals_differ,
                "red_digit": digit,
                "non_red_digit": proxy["intensity_digit"],
            }
            if proxy is not None
            else None
        ),
    }
    return red_row, selection


def build_catalog(
    rows: list[dict[str, Any]],
    *,
    minimum_margin: float,
    required_pairs: set[tuple[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    non_red_by_full: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    non_red_by_partial: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["is_red"]:
            non_red_by_full[
                (row["color"], row["gray"], row["thickness"], row["stretch"])
            ].append(row)
            non_red_by_partial[
                (row["color"], row["thickness"], row["stretch"])
            ].append(row)

    catalog: dict[str, dict[str, Any]] = {}
    for design_id in sorted(PAIR_BY_DESIGN_ID):
        color, digit = PAIR_BY_DESIGN_ID[design_id]
        if required_pairs is not None and (color, digit) not in required_pairs:
            continue
        base = {
            "design_id": design_id,
            "colour_symbol": color,
            "intensity_digit": digit,
            "intensity_interval": {
                "name": INTENSITY_INTERVAL_NAMES[digit],
                "label": INTENSITY_INTERVAL_LABELS[digit],
            },
        }
        selected, selection = select_measured_row(
            color=color,
            digit=digit,
            rows=rows,
            non_red_by_full=non_red_by_full,
            non_red_by_partial=non_red_by_partial,
            minimum_margin=minimum_margin,
        )
        catalog[design_id] = {
            **base,
            "material_parameters": {
                "color": color,
                "gray": selected["gray"],
                "thickness_nm": selected["thickness"],
                "stretch": selected["stretch"],
                "is_red": 1,
            },
            "observables": {
                "colour_symbol": color,
                "peak_cd_intensity": selected["peak_cd_intensity"],
                "peak_wavelength_nm": selected["peak_wavelength"],
                "source": "peak_csv_measurement",
            },
            "source_csv_row": compact_row(selected),
            "selection": selection,
        }
    return catalog


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(library: dict[str, Any]) -> str:
    evidence = library["data_evidence"]
    lines = [
        "# Red Phase CPL Configuration Library",
        "",
        f"- Library ID `{library['library_id']}`",
        "- Intensity intervals in mdeg are `0: < -5000`, `1: [-5000, 0)`, `2: [0, 5000]`, and `3: > 5000`.",
        f"- {len(library['design_catalog'])} measured configurations",
        "",
        "## CSV Coverage",
        "",
        "| Phase | Colour | Digit 0 | Digit 1 | Digit 2 | Digit 3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for phase, label in (("red", "Red phase"), ("non_red", "Non-red phase")):
        for color in "BGR":
            counts = evidence["coverage_counts_by_digit"][phase][color]
            lines.append(
                f"| {label} | {color} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |"
            )
    lines.extend(
        [
            "",
            f"> {evidence['pairing_limitation']}",
            "",
            "## D001–D012",
            "",
            "| ID | Colour/Morse | Digit | Interval | Red phase peak (mdeg) | Red phase parameters (gray/thickness/stretch) | Non-red phase match (gray/peak/digit) | Phase difference (mdeg) |",
            "|---|---|---:|---|---:|---|---|---:|",
        ]
    )
    morse = {"B": "dot `.`", "G": "dash `-`", "R": "separator or space"}
    for design_id in sorted(library["design_catalog"]):
        design = library["design_catalog"][design_id]
        material = design["material_parameters"]
        contrast = design["selection"]["phase_contrast"]
        proxy_summary = "—"
        difference = "—"
        if contrast is not None:
            proxy = contrast["non_red_proxy_row"]
            proxy_summary = f"{proxy['gray']:g}/{proxy['peak_cd_intensity']:.1f}/{proxy['intensity_digit']}"
            difference = f"{contrast['intensity_difference_mdeg']:.1f}"
        lines.append(
            f"| {design_id} | {design['colour_symbol']} / {morse[design['colour_symbol']]} | "
            f"{design['intensity_digit']} | {design['intensity_interval']['label']} | "
            f"{design['observables']['peak_cd_intensity']:.1f} | "
            f"{material['gray']:g}/{material['thickness_nm']:g}/{material['stretch']:g} | "
            f"{proxy_summary} | {difference} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        if args.minimum_boundary_margin_mdeg < 0:
            raise ProtocolError("minimum-boundary-margin-mdeg must be non-negative.")
        dataset_path = args.dataset.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        rows = read_dataset(dataset_path)
        evidence = analyze_dataset(rows, dataset_path)
        catalog = build_catalog(rows, minimum_margin=args.minimum_boundary_margin_mdeg)
        protocol = protocol_metadata()
        identity_catalog = {
            design_id: {
                "colour_symbol": design["colour_symbol"],
                "intensity_digit": design["intensity_digit"],
                "material_parameters": design["material_parameters"],
                "observables": design["observables"],
            }
            for design_id, design in catalog.items()
        }
        dataset_sha256 = file_sha256(dataset_path)
        identity = {
            "protocol": protocol,
            "design_catalog": identity_catalog,
            "dataset_sha256": dataset_sha256,
        }
        library = {
            "library_type": "cpl_joint_configuration_library",
            "library_id": f"sha256:{canonical_hash(identity)}",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "source_dataset": {
                "path": str(dataset_path),
                "sha256": dataset_sha256,
            },
            "design_catalog": catalog,
            "data_evidence": evidence,
            "selection_settings": {
                "minimum_boundary_margin_mdeg": args.minimum_boundary_margin_mdeg,
                "phase_proxy_match_fields": ["color", "thickness", "stretch"],
            },
        }
        validate_library(library)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "config_library.json"
        markdown_path = output_dir / "configuration_library.md"
        json_path.write_text(
            json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_markdown(library), encoding="utf-8")
        print(f"Configuration library: {json_path}")
        print(f"Human-readable report: {markdown_path}")
        print(f"Measured configurations: {len(catalog)}")
        print(
            "Exact full-parameter red/non-red pairs: "
            f"{evidence['exact_full_parameter_red_non_red_pair_count']}"
        )
        return 0
    except (OSError, ProtocolError, KeyError, TypeError, ValueError) as exc:
        print(f"LIBRARY ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
