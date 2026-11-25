#!/usr/bin/env python3
"""
PubChem PUG REST SMILES lookup script.
Input: compound name (CLI arg, stdin, JSON list, or file)
Output: JSON or TSV with name, CID, canonical_smiles

Respects PubChem rate limits: max 5 req/sec.
Includes smart fallback for common spelling variants (e.g. hydrazide -> ohydrazide).
"""

import argparse
import json
import sys
import time
import urllib.parse

import requests

PUGREST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_LIMIT_DELAY = 0.21  # seconds between requests (5 req/sec max)

# Common spelling / nomenclature fallbacks when exact name fails
FALLBACK_VARIANTS = {
    "hydrazide": ["ohydrazide"],          # IUPAC vs common spelling
}


def _apply_fallbacks(name: str) -> list[str]:
    """Generate candidate name variants for PubChem lookup."""
    candidates = [name]
    lowered = name.lower()
    for pattern, replacements in FALLBACK_VARIANTS.items():
        if pattern in lowered:
            for repl in replacements:
                candidates.append(name.lower().replace(pattern, repl))
    return candidates


def get_canonical_smiles(name: str, try_fallbacks: bool = True) -> dict:
    """
    Search PubChem by compound name and return CID + Canonical SMILES.
    Returns dict with 'error' key if the compound is not found or an error occurs.
    """
    candidates = _apply_fallbacks(name) if try_fallbacks else [name]

    for cand in candidates:
        safe_name = urllib.parse.quote(cand, safe="")
        url = f"{PUGREST}/compound/name/{safe_name}/property/ConnectivitySMILES/JSON"

        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as e:
            time.sleep(RATE_LIMIT_DELAY)
            continue  # try next candidate

        if response.status_code == 404:
            time.sleep(RATE_LIMIT_DELAY)
            continue  # try next candidate
        elif response.status_code == 503:
            time.sleep(RATE_LIMIT_DELAY)
            continue
        elif response.status_code != 200:
            time.sleep(RATE_LIMIT_DELAY)
            continue

        try:
            data = response.json()
            props = data["PropertyTable"]["Properties"][0]
            return {
                "query": name,
                "matched_name": cand,
                "cid": props["CID"],
                "canonical_smiles": props["ConnectivitySMILES"],
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            time.sleep(RATE_LIMIT_DELAY)
            continue

    return {"query": name, "error": "Not found in PubChem (tried variants)"}


def main():
    parser = argparse.ArgumentParser(
        description="Lookup canonical SMILES from PubChem by compound name."
    )
    parser.add_argument("names", nargs="*", help="Compound name(s) to look up")
    parser.add_argument("--json", action="store_true", help="Output as JSON array")
    parser.add_argument("--file", "-f", type=str, help="File with one compound name per line")
    parser.add_argument("--no-fallbacks", action="store_true", help="Disable spelling fallback variants")
    args = parser.parse_args()

    names = args.names or []
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            names.extend(line.strip() for line in f if line.strip())

    if not names:
        for line in sys.stdin:
            line = line.strip()
            if line:
                names.append(line)

    results = []
    try_fallbacks = not args.no_fallbacks
    for name in names:
        result = get_canonical_smiles(name, try_fallbacks=try_fallbacks)
        results.append(result)
        time.sleep(RATE_LIMIT_DELAY)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['query']}\tERROR\t{r['error']}")
            else:
                print(f"{r['query']}\t{r['cid']}\t{r['canonical_smiles']}")


if __name__ == "__main__":
    main()
