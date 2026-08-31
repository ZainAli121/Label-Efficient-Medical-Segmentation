from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow is required. Install the project's requirements first.", file=sys.stderr)
    raise SystemExit(2)


# Reproducibility and experiment configuration.
IMAGE_SIZE = 128
SPLIT_SEED = 42
TRAIN_POOL_SIZE = 720
VAL_SIZE = 180
ISIC_TEST_SIZE = 379
PH2_TEST_SIZE = 200
SUBSET_SIZES = [25, 50, 100, 200, 400]
SUBSET_SEEDS = [42, 123, 2026]
SANITY_SAMPLE_COUNT = 5
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CSV_FIELDS = ("sample_id", "image_path", "mask_path")


@dataclass(frozen=True)
class RawPair:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class ProcessedPair:
    sample_id: str
    image_path: Path
    mask_path: Path


def supported_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def isic_sample_id(path: Path, mask: bool = False) -> str:
    stem = path.stem
    suffix = "_segmentation"
    if mask and stem.casefold().endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def unique_map(paths: Iterable[Path], *, masks: bool, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        sample_id = isic_sample_id(path, mask=masks)
        key = sample_id.casefold()
        if key in result:
            duplicates.append(sample_id)
        else:
            result[key] = path
    if duplicates:
        raise ValueError(f"Duplicate {label} sample IDs: {', '.join(sorted(duplicates))}")
    return result


def collect_isic_pairs(image_dir: Path, mask_dir: Path, expected: int, label: str) -> list[RawPair]:
    images = unique_map(supported_files(image_dir), masks=False, label=f"{label} image")
    masks = unique_map(supported_files(mask_dir), masks=True, label=f"{label} mask")
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"{label} pairing failed; missing masks={missing_masks}, missing images={missing_images}"
        )
    if len(images) != expected:
        raise ValueError(f"Expected {expected} {label} pairs, found {len(images)}")
    return [
        RawPair(isic_sample_id(images[key]), images[key], masks[key])
        for key in sorted(images)
    ]


def collect_ph2_pairs(cases_root: Path) -> list[RawPair]:
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {cases_root}")
    cases = sorted((path for path in cases_root.iterdir() if path.is_dir()), key=lambda p: p.name.casefold())
    if len(cases) != PH2_TEST_SIZE:
        raise ValueError(f"Expected {PH2_TEST_SIZE} PH2 cases, found {len(cases)}")

    pairs: list[RawPair] = []
    errors: list[str] = []
    for case in cases:
        # Deliberately inspect only these directories. PH2 ROI content is ignored.
        images = supported_files(case / f"{case.name}_Dermoscopic_Image")
        masks = supported_files(case / f"{case.name}_lesion")
        if len(images) != 1 or len(masks) != 1:
            errors.append(f"{case.name}: images={len(images)}, masks={len(masks)}")
        else:
            pairs.append(RawPair(case.name, images[0], masks[0]))
    if errors:
        raise ValueError("Invalid PH2 cases: " + "; ".join(errors))
    return pairs


def split_isic_training(pairs: Sequence[RawPair]) -> tuple[list[RawPair], list[RawPair]]:
    if len(pairs) != TRAIN_POOL_SIZE + VAL_SIZE:
        raise ValueError(f"Expected {TRAIN_POOL_SIZE + VAL_SIZE} ISIC training pairs, found {len(pairs)}")
    shuffled = list(pairs)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    # The validation membership is fixed before any experiment-specific subset selection.
    validation = sorted(shuffled[:VAL_SIZE], key=lambda pair: pair.sample_id.casefold())
    training_pool = sorted(shuffled[VAL_SIZE:], key=lambda pair: pair.sample_id.casefold())
    return training_pool, validation


def process_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        processed = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
        )
        processed.save(destination, format="PNG")


def process_mask(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as mask:
        grayscale = mask.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST
        )
        # Any non-background value is lesion; point() produces a strict 0/255 L-mode mask.
        binary = grayscale.point(lambda value: 255 if value > 0 else 0, mode="L")
        binary.save(destination, format="PNG")


def process_split(
    pairs: Sequence[RawPair], output_dir: Path, repository_root: Path
) -> list[ProcessedPair]:
    processed: list[ProcessedPair] = []
    for index, pair in enumerate(pairs, start=1):
        image_path = output_dir / "images" / f"{pair.sample_id}.png"
        mask_path = output_dir / "masks" / f"{pair.sample_id}.png"
        process_image(pair.image_path, image_path)
        process_mask(pair.mask_path, mask_path)
        processed.append(ProcessedPair(pair.sample_id, image_path, mask_path))
        if index % 100 == 0 or index == len(pairs):
            print(f"  Processed {index}/{len(pairs)} pairs into {output_dir.relative_to(repository_root)}")
    return processed


def portable_path(path: Path, repository_root: Path) -> str:
    return path.relative_to(repository_root).as_posix()


def write_manifest(path: Path, pairs: Sequence[ProcessedPair], repository_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sample_id": pair.sample_id,
                    "image_path": portable_path(pair.image_path, repository_root),
                    "mask_path": portable_path(pair.mask_path, repository_root),
                }
            )
    temporary.replace(path)


def create_subset_manifests(
    training_pool: Sequence[ProcessedPair], manifests_dir: Path, repository_root: Path
) -> dict[int, dict[int, list[ProcessedPair]]]:
    all_subsets: dict[int, dict[int, list[ProcessedPair]]] = {}
    for seed in SUBSET_SEEDS:
        shuffled = list(training_pool)
        random.Random(seed).shuffle(shuffled)
        seed_subsets: dict[int, list[ProcessedPair]] = {}
        for size in SUBSET_SIZES:
            # Prefixes of one deterministic shuffle guarantee nesting within this seed.
            subset = shuffled[:size]
            seed_subsets[size] = subset
            write_manifest(
                manifests_dir / "subsets" / f"seed_{seed}" / f"train_{size}.csv",
                subset,
                repository_root,
            )
        all_subsets[seed] = seed_subsets
    return all_subsets


def verify_processed_pair(pair: ProcessedPair) -> list[str]:
    errors: list[str] = []
    try:
        with Image.open(pair.image_path) as image:
            image.load()
            if image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
                errors.append(
                    f"{pair.image_path}: expected {IMAGE_SIZE}x{IMAGE_SIZE} RGB, "
                    f"found {image.size} {image.mode}"
                )
    except (OSError, ValueError, SyntaxError) as exc:
        errors.append(f"{pair.image_path}: unreadable ({exc})")

    try:
        with Image.open(pair.mask_path) as mask:
            mask.load()
            if mask.size != (IMAGE_SIZE, IMAGE_SIZE) or mask.mode != "L":
                errors.append(
                    f"{pair.mask_path}: expected {IMAGE_SIZE}x{IMAGE_SIZE} L, "
                    f"found {mask.size} {mask.mode}"
                )
            values = set(mask.getdata())
            if not values.issubset({0, 255}):
                errors.append(f"{pair.mask_path}: non-binary pixel values {sorted(values)}")
    except (OSError, ValueError, SyntaxError) as exc:
        errors.append(f"{pair.mask_path}: unreadable ({exc})")
    return errors


def verify_outputs(
    training: Sequence[ProcessedPair],
    validation: Sequence[ProcessedPair],
    isic_test: Sequence[ProcessedPair],
    ph2_test: Sequence[ProcessedPair],
    subsets: dict[int, dict[int, list[ProcessedPair]]],
) -> list[str]:
    errors: list[str] = []
    expected_splits = (
        ("ISIC training pool", training, TRAIN_POOL_SIZE),
        ("ISIC validation", validation, VAL_SIZE),
        ("ISIC official test", isic_test, ISIC_TEST_SIZE),
        ("PH2 external test", ph2_test, PH2_TEST_SIZE),
    )
    for label, pairs, expected in expected_splits:
        if len(pairs) != expected:
            errors.append(f"{label}: expected {expected} pairs, found {len(pairs)}")
        ids = [pair.sample_id.casefold() for pair in pairs]
        if len(ids) != len(set(ids)):
            errors.append(f"{label}: duplicate sample IDs")
        for pair in pairs:
            errors.extend(verify_processed_pair(pair))

    train_ids = {pair.sample_id.casefold() for pair in training}
    val_ids = {pair.sample_id.casefold() for pair in validation}
    test_ids = {pair.sample_id.casefold() for pair in isic_test}
    if train_ids & val_ids:
        errors.append("ISIC training pool overlaps validation")
    if train_ids & test_ids:
        errors.append("ISIC training pool overlaps official test")
    if val_ids & test_ids:
        errors.append("ISIC validation overlaps official test")

    for seed, seed_subsets in subsets.items():
        previous_ids: set[str] = set()
        for size in SUBSET_SIZES:
            subset_ids = {pair.sample_id.casefold() for pair in seed_subsets[size]}
            if len(subset_ids) != size:
                errors.append(f"seed {seed}, train_{size}: expected {size} unique samples, found {len(subset_ids)}")
            if not subset_ids.issubset(train_ids):
                errors.append(f"seed {seed}, train_{size}: contains samples outside the training pool")
            if not previous_ids.issubset(subset_ids):
                errors.append(f"seed {seed}: train_{size} does not contain the preceding subset")
            previous_ids = subset_ids
    return errors


def create_sanity_figure(
    pairs: Sequence[ProcessedPair], destination: Path, repository_root: Path
) -> None:
    selected = random.Random(SPLIT_SEED).sample(list(pairs), SANITY_SAMPLE_COUNT)
    label_height = 22
    row_gap = 8
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * 2, (IMAGE_SIZE + label_height + row_gap) * len(selected) - row_gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, pair in enumerate(selected):
        top = row * (IMAGE_SIZE + label_height + row_gap)
        with Image.open(pair.image_path) as image, Image.open(pair.mask_path) as mask:
            canvas.paste(image.convert("RGB"), (0, top + label_height))
            canvas.paste(mask.convert("RGB"), (IMAGE_SIZE, top + label_height))
        draw.text((4, top + 4), f"{pair.sample_id}: image", fill="black")
        draw.text((IMAGE_SIZE + 4, top + 4), "binary mask", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    print(f"Saved sanity-check figure to {destination.relative_to(repository_root)}")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository_root / "data",
        help="Data directory (default: <repository>/data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    processed_root = data_root / "processed"
    manifests_dir = processed_root / "manifests"
    isic_raw = data_root / "isic-2016"

    print("Collecting raw image-mask pairs...")
    official_training = collect_isic_pairs(
        isic_raw / "ISBI2016_ISIC_Part1_Training_Data",
        isic_raw / "ISBI2016_ISIC_Part1_Training_GroundTruth",
        TRAIN_POOL_SIZE + VAL_SIZE,
        "ISIC training",
    )
    official_test = collect_isic_pairs(
        isic_raw / "ISBI2016_ISIC_Part1_Test_Data",
        isic_raw / "ISBI2016_ISIC_Part1_Test_GroundTruth",
        ISIC_TEST_SIZE,
        "ISIC test",
    )
    ph2_raw = collect_ph2_pairs(data_root / "ph2" / "PH2 Dataset images")
    training_raw, validation_raw = split_isic_training(official_training)

    print("Preprocessing datasets...")
    training = process_split(training_raw, processed_root / "isic2016" / "train", repository_root)
    validation = process_split(validation_raw, processed_root / "isic2016" / "val", repository_root)
    isic_test = process_split(official_test, processed_root / "isic2016" / "test", repository_root)
    ph2_test = process_split(ph2_raw, processed_root / "ph2" / "test", repository_root)

    print("Writing manifests...")
    write_manifest(manifests_dir / "isic_train_pool.csv", training, repository_root)
    write_manifest(manifests_dir / "isic_val.csv", validation, repository_root)
    write_manifest(manifests_dir / "isic_test.csv", isic_test, repository_root)
    write_manifest(manifests_dir / "ph2_test.csv", ph2_test, repository_root)
    subsets = create_subset_manifests(training, manifests_dir, repository_root)

    print("Validating processed outputs...")
    errors = verify_outputs(training, validation, isic_test, ph2_test, subsets)
    if errors:
        print("\nPREPROCESSING VALIDATION: FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    create_sanity_figure(training, repository_root / "reports" / "preprocessing_samples.png", repository_root)
    print("\nPREPROCESSING VALIDATION: PASS")
    print(f"  ISIC training pool: {len(training)}")
    print(f"  ISIC validation:    {len(validation)}")
    print(f"  ISIC official test: {len(isic_test)}")
    print(f"  PH2 external test:  {len(ph2_test)}")
    print("  Subset sizes:       " + ", ".join(map(str, SUBSET_SIZES)))
    print("  Subset seeds:       " + ", ".join(map(str, SUBSET_SEEDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
