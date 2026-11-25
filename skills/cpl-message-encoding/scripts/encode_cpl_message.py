#!/usr/bin/env python3
"""Encode text with a measured quaternary CPL configuration library."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from cpl_codec import (
    FIXED_ROW_CONFIGURATION_COUNT,
    MODEL_PREDICTION_SOURCE,
    MORSE_BY_LETTER,
    RADIX,
    ProtocolError,
    ascii_digits,
    attach_configurations,
    decode_document,
    normalize_message,
    plan_symbol_rows,
)
from inverse_design_bridge import build_message_library


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_LIBRARY = (
    SKILL_ROOT / "outputs" / "dataset_quaternary_red_library" / "config_library.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode A-Z text into spatially multiplexed red-phase CPL rows using "
            "two Morse characters and two fixed-width base-4 ASCII characters per "
            "eight-device row."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message")
    source.add_argument("--message-file", type=Path)
    configurations = parser.add_mutually_exclusive_group()
    configurations.add_argument("--library", type=Path, help="Use an existing configuration library.")
    configurations.add_argument(
        "--inverse-design", action="store_true", help="Resolve the configurations required by this message."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    design = parser.add_argument_group("inverse design options")
    design.add_argument("--measurements", type=Path, help="Measured peak CSV for any known configurations.")
    design.add_argument("--inverse-skill-dir", type=Path, help="Path to cof-inverse-design (defaults to a sibling skill).")
    design.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Inference device (default: auto).")
    design.add_argument("--stretch-steps", type=int, help="Number of stretch samples per thickness (default: 267).")
    args = parser.parse_args()
    if not args.inverse_design and any(
        value is not None for value in (args.measurements, args.inverse_skill_dir, args.device, args.stretch_steps)
    ):
        parser.error("Inverse design options require --inverse-design.")
    if args.stretch_steps is not None and args.stretch_steps < 2:
        parser.error("--stretch-steps must be at least 2.")
    return args


def read_message(args: argparse.Namespace) -> str:
    if args.message is not None:
        return args.message
    try:
        return args.message_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"Cannot read message file {args.message_file}: {exc}") from exc


def default_output_dir(normalized: str) -> Path:
    slug = "".join(character.lower() if "A" <= character <= "Z" else "_" for character in normalized)
    slug = slug.strip("_")[:64] or "message"
    return SKILL_ROOT / "outputs" / f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _render_character(character: str) -> str:
    return "space" if character == " " else character


def render_configuration_sequence_markdown(
    normalized: str,
    library: dict[str, Any],
    plans: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    """Render the plaintext, matching fabrication parameters, and peak provenance."""
    catalog = library["design_catalog"]
    lines = [
        "# CPL Configuration Sequence",
        "",
        f"Message `{normalized}`",
        "",
        f"Configuration library `{library['library_id']}`",
        "",
        "Each configuration carries a Morse colour token and one of four peak-intensity digits.",
        f"Each row has {FIXED_ROW_CONFIGURATION_COUNT} configurations and carries up to two Morse "
        "and two four-digit ASCII characters. Unused channel blocks in the last row contain padding.",
        "Each Morse character uses four slots. Short codes end at the first R, and four-mark codes end at the block boundary.",
        "",
        "## Configuration Mapping",
        "",
        "| ID | Colour token | Digit | Intensity interval | Peak intensity (mdeg) | Peak wavelength (nm) | Gray | Thickness (nm) | Stretch ratio | Source |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    colour_meaning = {"B": "dot `.`", "G": "dash `-`", "R": "separator or space"}
    for design_id in sorted(catalog):
        design = catalog[design_id]
        observations = design["observables"]
        material = design["material_parameters"]
        source = "Model prediction" if observations.get("source") == MODEL_PREDICTION_SOURCE else "Measurement"
        values = (observations.get("peak_wavelength_nm"), material.get("gray"),
                  material.get("thickness_nm"), material.get("stretch"))
        parameters = " | ".join(f"{value:g}" if isinstance(value, (int, float)) else "—" for value in values)
        lines.append(
            f"| {design_id} | {design['colour_symbol']} = "
            f"{colour_meaning[design['colour_symbol']]} | {design['intensity_digit']} | "
            f"{design['intensity_interval']['label']} | {observations['peak_cd_intensity']:.1f} | "
            f"{parameters} | {source} |"
        )

    counts: Counter[str] = Counter()
    total = 0
    for plan in plans:
        row_number = plan["row_index"] + 1
        lines.extend(
            [
                "",
                f"## Row {row_number}",
                "",
                f"Plaintext `{plan['plaintext']}`",
                "",
                "Colour/Morse characters "
                + ", ".join(
                    f"{_render_character(character)}="
                    f"{MORSE_BY_LETTER[character] if character != ' ' else '[R]'}"
                    for character in plan["morse_characters"]
                ),
                "",
                "Intensity/base-4 ASCII characters "
                + ", ".join(
                    f"{_render_character(character)}="
                    f"{''.join(map(str, ascii_digits(character, RADIX)))}"
                    for character in plan["ascii_characters"]
                ),
                "",
                "Encoded colour blocks `"
                + " ".join(plan["colour_stream"])
                + "`",
                "",
                "Eight colour slots including padding `"
                + " ".join(plan["colour_slots"])
                + "`",
                "",
                f"Intensity payload ({len(plan['ascii_characters'])} characters × 4 digits) `"
                + " ".join(map(str, plan["intensity_stream"]))
                + "`",
                "",
                "Eight intensity slots including padding `"
                + " ".join(map(str, plan["intensity_slots"]))
                + "`",
                "",
                "Configuration sequence `" + " ".join(cell["design_id"] for cell in plan["cells"]) + "`",
                "",
                "| Column | Configuration | Colour | Digit |",
                "|---:|---|---|---:|",
            ]
        )
        for cell in plan["cells"]:
            design_id = cell["design_id"]
            counts[design_id] += 1
            total += 1
            lines.append(
                f"| {cell['column'] + 1} | {design_id} | "
                f"{cell['intended_pair']['colour_symbol']} | "
                f"{cell['intended_pair']['intensity_digit']} |"
            )

    lines.extend(["", "## Counts", ""])
    for design_id in sorted(catalog):
        lines.append(f"{design_id} × {counts[design_id]}")
    lines.extend(
        [
            "",
            f"Total of {total} configurations in {len(plans)} rows.",
            "",
            "## Verification",
            "",
            f"Decoded message `{result['decoded_message']}`",
            "",
            f"Round-trip verification {'PASS' if result['success'] else 'FAIL'}",
            f"Devices using predicted peaks {result['predicted_cell_count']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        normalized = normalize_message(read_message(args))
        output_dir = (args.output_dir or default_output_dir(normalized)).expanduser().resolve()
        symbol_rows = plan_symbol_rows(normalized)
        if args.inverse_design:
            library_path = output_dir / "config_library.json"
            library = build_message_library(
                symbol_rows, output_dir,
                inverse_skill_dir=args.inverse_skill_dir or SKILL_ROOT.parent / "cof-inverse-design",
                measurements=args.measurements,
                device=args.device or "auto",
                stretch_steps=args.stretch_steps or 267,
            )
        else:
            library_path = (args.library or DEFAULT_LIBRARY).expanduser().resolve()
            library = json.loads(library_path.read_text(encoding="utf-8"))
        plans = attach_configurations(symbol_rows, library)

        rows = [
            {
                "row_index": plan["row_index"],
                "payload_character_count": plan["payload_character_count"],
                "morse_character_count": plan["morse_character_count"],
                "ascii_character_count": plan["ascii_character_count"],
                "cell_count": len(plan["cells"]),
                "cells": plan["cells"],
            }
            for plan in plans
        ]
        document: dict[str, Any] = {
            "document_type": "cpl_spatially_multiplexed_message",
            "protocol": library["protocol"],
            "library_reference": {
                "library_id": library["library_id"],
                "path_used_for_encoding": str(library_path),
            },
            "plaintext_policy": {
                "plaintext_stored_in_json": False,
                "plaintext_present_in_markdown_sidecar": True,
            },
            "rows": rows,
        }
        result = decode_document(document, expected=normalized)
        if not result["success"] or not result["matches_expected"]:
            raise ProtocolError(
                f"Internal round-trip mismatch: encoded {normalized!r}, "
                f"decoded {result['decoded_message']!r}."
            )
        document["verification"] = {
            "round_trip_passed": True,
            "comparison_plaintext_stored": False,
            "row_count": result["row_count"],
            "configuration_count": result["configuration_count"],
            "predicted_cell_count": result["predicted_cell_count"],
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "encoded_message.json"
        markdown_path = output_dir / "configuration_sequence.md"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(
            render_configuration_sequence_markdown(normalized, library, plans, result),
            encoding="utf-8",
        )
        if args.inverse_design:
            library_path.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Design targets: {output_dir / 'design_targets.json'}")
        print(f"Normalized input: {normalized}")
        print(f"Decoded message: {result['decoded_message']}")
        print(f"Rows: {result['row_count']}")
        print(f"Configurations: {result['configuration_count']}")
        print(f"Configurations using predicted peaks: {result['predicted_cell_count']}")
        print(f"Configuration library: {library_path}")
        print(f"Machine JSON: {json_path}")
        print(f"Human-readable sequence: {markdown_path}")
        print("Round-trip verification: PASS")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ENCODE ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
