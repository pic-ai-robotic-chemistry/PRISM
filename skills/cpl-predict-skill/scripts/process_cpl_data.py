#!/usr/bin/env python3
"""Parse CPL database txt files and save g(lum) / CD[mdeg] as pickle dicts."""

import argparse
import pickle
from pathlib import Path


def parse_filename(filepath: Path, is_red: bool) -> tuple:
    """Extract (color, gray, thickness, stretch) from filename."""
    # e.g. B-1-30-67.txt  ->  ('B', 1, 30, 67)
    # e.g. B-5-30-100.txt (red phase)
    stem = filepath.stem
    parts = stem.split("-")
    color = parts[0]
    gray = int(parts[1])
    thickness = int(parts[2])
    stretch = int(parts[3])
    return color, gray, thickness, stretch


def parse_txt_file(filepath: Path) -> tuple:
    """
    Parse JASCO CPL txt file.
    Returns (wavelength_arr, g_lum_arr, cd_arr)
    """
    wavelengths = []
    g_lum = []
    cd = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find XYDATA line index
    xydata_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "XYDATA":
            xydata_idx = i
            break

    if xydata_idx is None:
        raise ValueError(f"XYDATA not found in {filepath}")

    # Parse data lines after XYDATA
    for line in lines[xydata_idx + 1 :]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#####"):
            break  # end of data section
        parts = line.split()
        if len(parts) < 5:
            continue
        wavelengths.append(float(parts[0]))
        g_lum.append(float(parts[1]))
        cd.append(float(parts[3]))  # CD [mdeg] is the 4th column (index 3)

    return wavelengths, g_lum, cd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of the JASCO TXT exports.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Directory for the two pickle files.")
    parser.add_argument(
        "--red-phase-dir", default="red phase", help="Name of the red-phase subdirectory (default: %(default)s)."
    )
    args = parser.parse_args()

    g_factor_dict = {}
    cd_dict = {}

    # Collect all txt files
    txt_files = sorted(args.data_dir.rglob("*.txt"))
    if not txt_files:
        parser.error(f"No TXT files found in {args.data_dir}")
    print(f"Found {len(txt_files)} txt files.")

    for filepath in txt_files:
        # Determine if red phase from path
        is_red = args.red_phase_dir in filepath.relative_to(args.data_dir).parent.parts

        color, gray, thickness, stretch = parse_filename(filepath, is_red)
        key = f"{color}_{gray}_{thickness}_{stretch}_{is_red}"

        wavelengths, g_lum, cd = parse_txt_file(filepath)

        g_factor_dict[key] = {
            "wavelength": wavelengths,
            "g_lum": g_lum,
        }
        cd_dict[key] = {
            "wavelength": wavelengths,
            "cd": cd,
        }

    print(f"Parsed {len(g_factor_dict)} entries.")

    # Save pickle files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g_factor_path = args.output_dir / "cpl_g_factor.pkl"
    cd_path = args.output_dir / "cpl_cd.pkl"

    with open(g_factor_path, "wb") as f:
        pickle.dump(g_factor_dict, f)
    print(f"Saved: {g_factor_path} ({len(g_factor_dict)} keys)")

    with open(cd_path, "wb") as f:
        pickle.dump(cd_dict, f)
    print(f"Saved: {cd_path} ({len(cd_dict)} keys)")

    # Print some sample keys
    print("\nSample keys:")
    for k in sorted(g_factor_dict.keys())[:5]:
        print(f"  {k}")


if __name__ == "__main__":
    main()
