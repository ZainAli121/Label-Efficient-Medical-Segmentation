from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print(
        "ERROR: This script requires Pillow, NumPy, and matplotlib.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


QC_SEED = 42
QC_SAMPLE_COUNT = 5
EXPECTED_IMAGE_SIZE = (128, 128)
TINY_MASK_THRESHOLD = 1.0
LARGE_MASK_THRESHOLD = 90.0
REQUIRED_COLUMNS = {"sample_id", "image_path", "mask_path"}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass
class DatasetQC:
    label: str
    rows: list[ManifestRow] = field(default_factory=list)
    area_percentages: list[float] = field(default_factory=list)
    empty_masks: list[str] = field(default_factory=list)
    full_masks: list[str] = field(default_factory=list)
    tiny_masks: list[str] = field(default_factory=list)
    large_masks: list[str] = field(default_factory=list)
    nonbinary_masks: list[str] = field(default_factory=list)
    dimension_errors: list[str] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    valid_visual_rows: list[ManifestRow] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    manifest_name: str
    figure_name: str
    seed_offset: int


DATASETS = (
    DatasetConfig("ISIC TRAIN", "isic_train_pool.csv", "isic_train_qc.png", 0),
    DatasetConfig("ISIC VALIDATION", "isic_val.csv", "isic_val_qc.png", 1),
    DatasetConfig("ISIC TEST", "isic_test.csv", "isic_test_qc.png", 2),
    DatasetConfig("PH2 EXTERNAL TEST", "ph2_test.csv", "ph2_qc.png", 3),
)


def resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def load_manifest(path: Path, repository_root: Path) -> list[ManifestRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows: list[ManifestRow] = []
        for line_number, record in enumerate(reader, start=2):
            sample_id = (record.get("sample_id") or "").strip()
            image_value = (record.get("image_path") or "").strip()
            mask_value = (record.get("mask_path") or "").strip()
            if not sample_id or not image_value or not mask_value:
                raise ValueError(f"{path}, line {line_number}: blank required value")
            rows.append(
                ManifestRow(
                    sample_id,
                    resolve_path(image_value, repository_root),
                    resolve_path(mask_value, repository_root),
                )
            )
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    normalized_ids = [row.sample_id.casefold() for row in rows]
    duplicate_ids = sorted({value for value in normalized_ids if normalized_ids.count(value) > 1})
    if duplicate_ids:
        raise ValueError(f"{path} contains duplicate sample IDs: {', '.join(duplicate_ids)}")
    return rows


def inspect_dataset(label: str, rows: list[ManifestRow]) -> DatasetQC:
    result = DatasetQC(label=label, rows=rows)
    for row in rows:
        image_size: tuple[int, int] | None = None
        mask_size: tuple[int, int] | None = None
        image_ok = False
        mask_ok = False
        try:
            with Image.open(row.image_path) as image:
                image.load()
                image_size = image.size
                image_ok = True
                if image.mode != "RGB":
                    result.dimension_errors.append(
                        f"{row.sample_id}: image mode={image.mode}, expected RGB"
                    )
        except (OSError, ValueError, SyntaxError) as exc:
            result.unreadable_files.append(f"{row.sample_id} image ({row.image_path}): {exc}")

        try:
            with Image.open(row.mask_path) as mask:
                mask.load()
                mask_size = mask.size
                mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                mask_ok = True
                values = set(np.unique(mask_array).tolist())
                if not values.issubset({0, 255}):
                    result.nonbinary_masks.append(f"{row.sample_id}: values={sorted(values)}")
                lesion_pixels = int(np.count_nonzero(mask_array))
                total_pixels = int(mask_array.size)
                area = lesion_pixels / total_pixels * 100.0
                result.area_percentages.append(area)
                if lesion_pixels == 0:
                    result.empty_masks.append(row.sample_id)
                if lesion_pixels == total_pixels:
                    result.full_masks.append(row.sample_id)
                if area < TINY_MASK_THRESHOLD:
                    result.tiny_masks.append(row.sample_id)
                if area > LARGE_MASK_THRESHOLD:
                    result.large_masks.append(row.sample_id)
        except (OSError, ValueError, SyntaxError) as exc:
            result.unreadable_files.append(f"{row.sample_id} mask ({row.mask_path}): {exc}")

        if image_size is not None and image_size != EXPECTED_IMAGE_SIZE:
            result.dimension_errors.append(
                f"{row.sample_id}: image size={image_size}, expected={EXPECTED_IMAGE_SIZE}"
            )
        if mask_size is not None and mask_size != EXPECTED_IMAGE_SIZE:
            result.dimension_errors.append(
                f"{row.sample_id}: mask size={mask_size}, expected={EXPECTED_IMAGE_SIZE}"
            )
        if image_size is not None and mask_size is not None and image_size != mask_size:
            result.dimension_errors.append(
                f"{row.sample_id}: image size={image_size}, mask size={mask_size}"
            )
        if image_ok and mask_ok and image_size == mask_size:
            result.valid_visual_rows.append(row)
    return result


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    lesion = mask > 0
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = 0.38
    overlay[lesion] = (1.0 - alpha) * overlay[lesion] + alpha * red[lesion]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_qc_figure(result: DatasetQC, destination: Path, seed: int) -> None:
    if not result.valid_visual_rows:
        raise ValueError(f"{result.label}: no readable, aligned pairs for the QC figure")
    count = min(QC_SAMPLE_COUNT, len(result.valid_visual_rows))
    selected = random.Random(seed).sample(result.valid_visual_rows, count)
    figure, axes = plt.subplots(count, 3, figsize=(10, 3.2 * count), squeeze=False)
    for row_index, row in enumerate(selected):
        with Image.open(row.image_path) as image_file, Image.open(row.mask_path) as mask_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
            mask = np.asarray(mask_file.convert("L"), dtype=np.uint8)
        panels = (image, mask, make_overlay(image, mask))
        titles = ("Processed RGB image", "Binary ground-truth mask", "Mask overlay")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axis = axes[row_index, column]
            axis.imshow(panel, cmap="gray" if column == 1 else None, vmin=0, vmax=255)
            axis.set_title(f"{row.sample_id}\n{title}", fontsize=9)
            axis.axis("off")
    figure.suptitle(f"{result.label} - processed-data QC", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def statistics_for(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return (float("nan"),) * 5
    return (
        min(values),
        statistics.median(values),
        statistics.mean(values),
        max(values),
        statistics.pstdev(values),
    )


def print_flagged(result: DatasetQC) -> None:
    categories: tuple[tuple[str, list[str]], ...] = (
        ("Empty masks", result.empty_masks),
        ("Completely full masks", result.full_masks),
        ("Tiny masks (<1%)", result.tiny_masks),
        ("Large masks (>90%)", result.large_masks),
        ("Non-binary masks", result.nonbinary_masks),
        ("Dimension/mode errors", result.dimension_errors),
        ("Unreadable files", result.unreadable_files),
    )
    flagged = [(name, values) for name, values in categories if values]
    if flagged:
        print(f"\n{result.label} flagged samples:")
        for name, values in flagged:
            print(f"  {name} ({len(values)}):")
            for value in values:
                print(f"    - {value}")


def print_summary(results: list[DatasetQC], fatal_errors: list[str]) -> str:
    has_failures = bool(fatal_errors) or any(
        result.unreadable_files or result.dimension_errors or result.nonbinary_masks
        for result in results
    )
    has_warnings = any(
        result.empty_masks or result.full_masks or result.tiny_masks or result.large_masks
        for result in results
    )
    status = "FAIL" if has_failures else "WARNING" if has_warnings else "PASS"
    print("\n========================================")
    print("PROCESSED DATA QC SUMMARY")
    print("========================================")
    for result in results:
        minimum, median, mean, maximum, stddev = statistics_for(result.area_percentages)
        print(f"\n{result.label}")
        print(f"Samples:             {len(result.rows)}")
        print(f"Min lesion area:     {minimum:.4f}%")
        print(f"Median lesion area:  {median:.4f}%")
        print(f"Mean lesion area:    {mean:.4f}%")
        print(f"Max lesion area:     {maximum:.4f}%")
        print(f"Std. deviation:      {stddev:.4f}")
        print(f"Empty masks:         {len(result.empty_masks)}")
        print(f"Full masks:          {len(result.full_masks)}")
        print(f"Tiny masks (<1%):    {len(result.tiny_masks)}")
        print(f"Large masks (>90%):  {len(result.large_masks)}")
        print(f"Dimension errors:    {len(result.dimension_errors)}")
        print(f"Unreadable files:    {len(result.unreadable_files)}")
    if fatal_errors:
        print("\nFatal QC errors:")
        for error in fatal_errors:
            print(f"  - {error}")
    print(f"\nFINAL QC STATUS: {status}")
    print("========================================")
    return status


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=repository_root / "data" / "processed" / "manifests",
        help="Processed manifest directory",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=repository_root / "reports" / "qc",
        help="QC figure output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    manifests_dir = args.manifests_dir.expanduser().resolve()
    reports_dir = args.reports_dir.expanduser().resolve()
    results: list[DatasetQC] = []
    fatal_errors: list[str] = []
    for config in DATASETS:
        try:
            rows = load_manifest(manifests_dir / config.manifest_name, repository_root)
            result = inspect_dataset(config.label, rows)
            results.append(result)
            print_flagged(result)
            figure_path = reports_dir / config.figure_name
            create_qc_figure(result, figure_path, QC_SEED + config.seed_offset)
            print(f"Saved {figure_path}")
        except (OSError, ValueError, csv.Error) as exc:
            fatal_errors.append(f"{config.label}: {exc}")
    status = print_summary(results, fatal_errors)
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
