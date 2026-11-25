#!/usr/bin/env python3
"""Decode CPL rows from colour and four-interval peak intensities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cpl_codec import ProtocolError, decode_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode spatially multiplexed CPL rows by reading colour and quaternary "
            "peak-intensity intervals separately."
        )
    )
    parser.add_argument("input_json", type=Path, help="Path to encoded_message.json.")
    parser.add_argument("--expected", default=None, help="Optional external plaintext for comparison.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON.")
    parser.add_argument("--show-rows", action="store_true", help="Show both decoded channels per row.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document = json.loads(args.input_json.read_text(encoding="utf-8"))
        result = decode_document(document, expected=args.expected)
    except (OSError, json.JSONDecodeError, ProtocolError, KeyError, TypeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"DECODE ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Decoded message: {result['decoded_message']!r}")
        print(f"Rows: {result['row_count']}")
        print(f"Configurations: {result['configuration_count']}")
        print(f"Configurations using predicted peaks: {result['predicted_cell_count']}")
        if result["comparison_performed"]:
            print(f"Expected message: {result['expected_message']!r}")
            print(f"Comparison: {'PASS' if result['matches_expected'] else 'FAIL'}")
        else:
            print("Comparison: NOT REQUESTED")
        if args.show_rows:
            for row in result["rows"]:
                print(
                    f"Row {row['row_index'] + 1}: colour/Morse={row['morse_characters']!r}, "
                    f"intensity/ASCII={row['ascii_characters']!r}, "
                    f"merged={row['interleaved']!r}"
                )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
