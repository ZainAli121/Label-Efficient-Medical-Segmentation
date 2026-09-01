#!/usr/bin/env python3
"""Run the four train-25, seed-42 augmentation pilots and compare them."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.augmentations import AUGMENTATION_MODES, verify_geometric_alignment


def main() -> int:
    pilots_root = REPOSITORY_ROOT / "outputs" / "pilots"
    alignment_path = pilots_root / "geometric_alignment_check.png"
    minimum_iou = verify_geometric_alignment(alignment_path)
    print(f"Geometric alignment check passed (minimum synthetic IoU={minimum_iou:.4f}).")
    print(f"Alignment figure: {alignment_path}")

    results: list[dict[str, str | int | float]] = []
    for augmentation in AUGMENTATION_MODES:
        output_dir = pilots_root / f"train25_seed42_{augmentation}"
        print(f"\nRunning augmentation pilot: {augmentation}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train_baseline.py"),
                "--augmentation",
                augmentation,
                "--output-dir",
                str(output_dir),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        with (output_dir / "history.json").open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        results.append(
            {
                "augmentation": augmentation,
                "best_validation_dice": history["best_validation_dice"],
                "best_validation_iou": history["best_validation_iou"],
                "best_epoch": history["best_epoch"],
                "epochs_trained": history["epochs_trained"],
            }
        )

    comparison_path = pilots_root / "train25_seed42_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nAUGMENTATION PILOT COMPARISON")
    print("augmentation  best val Dice  best val IoU  best epoch  epochs trained")
    print("------------  -------------  ------------  ----------  --------------")
    for result in results:
        print(
            f"{result['augmentation']:<12}  "
            f"{float(result['best_validation_dice']):>13.4f}  "
            f"{float(result['best_validation_iou']):>12.4f}  "
            f"{int(result['best_epoch']):>10d}  "
            f"{int(result['epochs_trained']):>14d}"
        )
    print(f"\nSaved comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
