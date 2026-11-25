#!/usr/bin/env python3
"""Create stratified test and cross-validation key splits for CPL data."""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CPL stratified test + K-fold CV splits.")
    parser.add_argument("--data_path", default="data/cpl_cd.pkl", help="Pickled CPL data path.")
    parser.add_argument("--out_dir", default="cv_splits/seed_42", help="Output directory for key files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Held-out test ratio per group.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    return parser.parse_args()


def group_key(key: str) -> str:
    color, _gray, _thickness, _stretch, is_red = key.split("_")
    return f"{color}_{is_red}"


def write_keys(path: Path, keys: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(keys)) + "\n")


def count_groups(keys: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(group_key(k) for k in keys).items()))


def balanced_group_parts(keys: list[str], n_folds: int, fold_val_counts: list[int]) -> list[list[str]]:
    base_size, remainder = divmod(len(keys), n_folds)
    sizes = [base_size] * n_folds
    extra_folds = sorted(range(n_folds), key=lambda idx: (fold_val_counts[idx], idx))[:remainder]
    for idx in extra_folds:
        sizes[idx] += 1

    parts = []
    start = 0
    for idx, size in enumerate(sizes):
        part = keys[start : start + size]
        parts.append(part)
        fold_val_counts[idx] += len(part)
        start += size
    return parts


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)

    with data_path.open("rb") as f:
        raw_data = pickle.load(f)

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1.")
    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2.")

    rng = np.random.RandomState(args.seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in raw_data:
        groups[group_key(key)].append(key)

    trainval_keys: list[str] = []
    test_keys: list[str] = []
    grouped_trainval: dict[str, list[str]] = {}

    for grp, keys in sorted(groups.items()):
        keys = list(rng.permutation(sorted(keys)))
        n_trainval = int(len(keys) * (1 - args.test_ratio))
        if n_trainval <= 0 or n_trainval >= len(keys):
            raise ValueError(f"Group {grp} has invalid train/test split size: {len(keys)}")
        grouped_trainval[grp] = keys[:n_trainval]
        trainval_keys.extend(keys[:n_trainval])
        test_keys.extend(keys[n_trainval:])

    write_keys(out_dir / "trainval_keys.txt", trainval_keys)
    write_keys(out_dir / "test_keys.txt", test_keys)

    fold_summaries = []
    fold_val_counts = [0] * args.n_folds
    fold_parts = {}
    for grp, keys in sorted(grouped_trainval.items()):
        fold_parts[grp] = balanced_group_parts(keys, args.n_folds, fold_val_counts)

    all_fold_val_sets = []
    for fold_idx in range(args.n_folds):
        val_keys: list[str] = []
        fold_train_keys: list[str] = []
        for grp, parts in sorted(fold_parts.items()):
            val_keys.extend(parts[fold_idx])
            for idx, part in enumerate(parts):
                if idx != fold_idx:
                    fold_train_keys.extend(part)

        fold_dir = out_dir / f"fold_{fold_idx + 1}"
        write_keys(fold_dir / "train_keys.txt", fold_train_keys)
        write_keys(fold_dir / "val_keys.txt", val_keys)
        all_fold_val_sets.extend(val_keys)

        fold_summaries.append({
            "fold": fold_idx + 1,
            "train_count": len(fold_train_keys),
            "val_count": len(val_keys),
            "train_groups": count_groups(fold_train_keys),
            "val_groups": count_groups(val_keys),
        })

    if set(trainval_keys) & set(test_keys):
        raise RuntimeError("Trainval/test overlap detected.")
    if Counter(all_fold_val_sets) != Counter(trainval_keys):
        raise RuntimeError("CV validation folds do not cover trainval keys exactly once.")

    summary = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_folds": args.n_folds,
        "total_count": len(raw_data),
        "trainval_count": len(trainval_keys),
        "test_count": len(test_keys),
        "all_groups": count_groups(raw_data.keys()),
        "trainval_groups": count_groups(trainval_keys),
        "test_groups": count_groups(test_keys),
        "folds": fold_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote CV splits to: {out_dir}")
    print(f"Trainval: {len(trainval_keys)}, Test: {len(test_keys)}")
    for fold in fold_summaries:
        print(f"Fold {fold['fold']}: train={fold['train_count']}, val={fold['val_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
