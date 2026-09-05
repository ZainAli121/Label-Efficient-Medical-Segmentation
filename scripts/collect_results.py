#!/usr/bin/env python3
"""Collect completed controlled experiments into one master CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "outputs" / "experiments"
OUTPUT_PATH = EXPERIMENTS_ROOT / "master_results.csv"
FIELDS = (
    "train_size",
    "subset_seed",
    "augmentation",
    "best_val_dice",
    "best_val_iou",
    "best_epoch",
    "epochs_trained",
    "checkpoint_path",
)


def validate_result(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    missing = set(FIELDS) - set(result)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    checkpoint = Path(str(result["checkpoint_path"]))
    checkpoint = checkpoint if checkpoint.is_absolute() else REPOSITORY_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint referenced by {path} does not exist: {checkpoint}")
    return {field: result[field] for field in FIELDS}


def main() -> int:
    if not EXPERIMENTS_ROOT.is_dir():
        raise FileNotFoundError(f"Experiments directory does not exist: {EXPERIMENTS_ROOT}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for result_path in EXPERIMENTS_ROOT.glob("train*_seed*_*/result.json"):
        with result_path.open("r", encoding="utf-8") as handle:
            row = validate_result(result_path, json.load(handle))
        identity = (int(row["train_size"]), int(row["subset_seed"]), str(row["augmentation"]))
        if identity in seen:
            raise ValueError(f"Duplicate experiment identity found: {identity}")
        seen.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed result.json files found under {EXPERIMENTS_ROOT}")
    rows.sort(key=lambda row: (int(row["train_size"]), int(row["subset_seed"]), str(row["augmentation"])))
    temporary = OUTPUT_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT_PATH)
    print(f"Collected {len(rows)} completed experiments.")
    print(f"Saved: {OUTPUT_PATH}")
    if len(rows) != 60:
        print(f"WARNING: expected 60 controlled experiments, currently found {len(rows)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
