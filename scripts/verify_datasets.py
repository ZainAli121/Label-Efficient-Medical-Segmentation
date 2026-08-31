from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print(
        "ERROR: Pillow is required to check whether image files are readable. "
        "Install Pillow in your project environment and run this script again.",
        file=sys.stderr,
    )
    raise SystemExit(2)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    "ISIC Training": (900, 900),
    "ISIC Test": (379, 379),
    "PH2": (200, 200),
}
EXAMPLE_LIMIT = 5


@dataclass
class IsicResult:
    label: str
    images: list[Path] = field(default_factory=list)
    masks: list[Path] = field(default_factory=list)
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    missing_masks: list[Path] = field(default_factory=list)
    missing_images: list[Path] = field(default_factory=list)
    duplicate_image_ids: dict[str, list[Path]] = field(default_factory=dict)
    duplicate_mask_ids: dict[str, list[Path]] = field(default_factory=dict)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


@dataclass
class Ph2Result:
    cases: list[Path] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    masks: list[Path] = field(default_factory=list)
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    missing_images: list[Path] = field(default_factory=list)
    missing_masks: list[Path] = field(default_factory=list)
    multiple_images: dict[str, list[Path]] = field(default_factory=dict)
    multiple_masks: dict[str, list[Path]] = field(default_factory=dict)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


def image_files(directory: Path) -> list[Path]:
    """Return supported image files recursively, without changing them."""
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def check_readable(paths: Iterable[Path]) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    for path in paths:
        try:
            # verify() checks integrity; reopening and load() also exercises decoding.
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
    return failures


def isic_id(path: Path, is_mask: bool) -> str:
    stem = path.stem
    if is_mask and stem.casefold().endswith("_segmentation"):
        stem = stem[: -len("_segmentation")]
    return stem.casefold()


def group_by_id(paths: Iterable[Path], is_mask: bool) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(isic_id(path, is_mask), []).append(path)
    return grouped


def verify_isic(label: str, image_dir: Path, mask_dir: Path) -> IsicResult:
    result = IsicResult(label=label)
    result.images = image_files(image_dir)
    result.masks = image_files(mask_dir)
    image_map = group_by_id(result.images, is_mask=False)
    mask_map = group_by_id(result.masks, is_mask=True)

    result.duplicate_image_ids = {key: value for key, value in image_map.items() if len(value) > 1}
    result.duplicate_mask_ids = {key: value for key, value in mask_map.items() if len(value) > 1}

    for key in sorted(image_map.keys() & mask_map.keys()):
        # A valid pair must be unambiguous on both sides.
        if len(image_map[key]) == 1 and len(mask_map[key]) == 1:
            result.pairs.append((image_map[key][0], mask_map[key][0]))
    for key in sorted(image_map.keys() - mask_map.keys()):
        result.missing_masks.extend(image_map[key])
    for key in sorted(mask_map.keys() - image_map.keys()):
        result.missing_images.extend(mask_map[key])

    result.unreadable = check_readable([*result.images, *result.masks])
    return result


def verify_ph2(cases_root: Path) -> Ph2Result:
    result = Ph2Result()
    if cases_root.is_dir():
        result.cases = sorted(
            (path for path in cases_root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )

    for case in result.cases:
        # Search only the two relevant PH2 subtrees; ROI files are never included.
        dermoscopic_dir = case / f"{case.name}_Dermoscopic_Image"
        lesion_dir = case / f"{case.name}_lesion"
        case_images = image_files(dermoscopic_dir)
        case_masks = image_files(lesion_dir)
        result.images.extend(case_images)
        result.masks.extend(case_masks)

        if not case_images:
            result.missing_images.append(case)
        elif len(case_images) > 1:
            result.multiple_images[case.name] = case_images
        if not case_masks:
            result.missing_masks.append(case)
        elif len(case_masks) > 1:
            result.multiple_masks[case.name] = case_masks
        if len(case_images) == 1 and len(case_masks) == 1:
            result.pairs.append((case_images[0], case_masks[0]))

    result.unreadable = check_readable([*result.images, *result.masks])
    return result


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_items(title: str, paths: Iterable[Path], root: Path) -> None:
    paths = list(paths)
    if paths:
        print(f"\n{title} ({len(paths)}):")
        for path in paths:
            print(f"  - {relative(path, root)}")


def print_duplicates(title: str, groups: dict[str, list[Path]], root: Path) -> None:
    if groups:
        print(f"\n{title} ({len(groups)} IDs/cases):")
        for key, paths in sorted(groups.items()):
            print(f"  - {key}:")
            for path in paths:
                print(f"      {relative(path, root)}")


def print_unreadable(title: str, failures: list[tuple[Path, str]], root: Path) -> None:
    if failures:
        print(f"\n{title} ({len(failures)}):")
        for path, error in failures:
            print(f"  - {relative(path, root)}: {error}")


def print_examples(title: str, pairs: list[tuple[Path, Path]], root: Path) -> None:
    print(f"\n{title} example pairs:")
    if not pairs:
        print("  (none)")
        return
    for image, mask in pairs[:EXAMPLE_LIMIT]:
        print(f"  - {relative(image, root)}  <->  {relative(mask, root)}")


def isic_passes(result: IsicResult, expected: tuple[int, int]) -> bool:
    return all(
        (
            len(result.images) == expected[0],
            len(result.masks) == expected[1],
            len(result.pairs) == expected[0],
            not result.missing_masks,
            not result.missing_images,
            not result.duplicate_image_ids,
            not result.duplicate_mask_ids,
            not result.unreadable,
        )
    )


def ph2_passes(result: Ph2Result, expected: tuple[int, int]) -> bool:
    return all(
        (
            len(result.cases) == expected[0],
            len(result.images) == expected[0],
            len(result.masks) == expected[1],
            len(result.pairs) == expected[0],
            not result.missing_images,
            not result.missing_masks,
            not result.multiple_images,
            not result.multiple_masks,
            not result.unreadable,
        )
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository_root / "data",
        help="Raw data directory (default: <repository>/data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    isic_root = data_root / "isic-2016"
    ph2_root = data_root / "ph2" / "PH2 Dataset images"

    required_directories = [
        isic_root / "ISBI2016_ISIC_Part1_Training_Data",
        isic_root / "ISBI2016_ISIC_Part1_Training_GroundTruth",
        isic_root / "ISBI2016_ISIC_Part1_Test_Data",
        isic_root / "ISBI2016_ISIC_Part1_Test_GroundTruth",
        ph2_root,
    ]
    missing_directories = [path for path in required_directories if not path.is_dir()]

    training = verify_isic("ISIC Training", required_directories[0], required_directories[1])
    test = verify_isic("ISIC Test", required_directories[2], required_directories[3])
    ph2 = verify_ph2(ph2_root)

    if missing_directories:
        print_items("Missing required directories", missing_directories, data_root)

    for result in (training, test):
        print_examples(result.label, result.pairs, data_root)
        print_items(f"{result.label} images missing masks", result.missing_masks, data_root)
        print_items(f"{result.label} masks missing images", result.missing_images, data_root)
        print_duplicates(f"{result.label} duplicate image IDs", result.duplicate_image_ids, data_root)
        print_duplicates(f"{result.label} duplicate mask IDs", result.duplicate_mask_ids, data_root)
        print_unreadable(f"{result.label} unreadable files", result.unreadable, data_root)

    print_examples("PH2", ph2.pairs, data_root)
    print_items("PH2 cases missing dermoscopic images", ph2.missing_images, data_root)
    print_items("PH2 cases missing lesion masks", ph2.missing_masks, data_root)
    print_duplicates("PH2 cases with multiple dermoscopic images", ph2.multiple_images, data_root)
    print_duplicates("PH2 cases with multiple lesion masks", ph2.multiple_masks, data_root)
    print_unreadable("PH2 unreadable files", ph2.unreadable, data_root)

    passed = (
        not missing_directories
        and isic_passes(training, EXPECTED_COUNTS[training.label])
        and isic_passes(test, EXPECTED_COUNTS[test.label])
        and ph2_passes(ph2, EXPECTED_COUNTS["PH2"])
    )

    print("\n========================================")
    print("DATASET VERIFICATION SUMMARY")
    print("========================================")
    for result in (training, test):
        print(f"\n{result.label}")
        print(f"Images:           {len(result.images)}")
        print(f"Masks:            {len(result.masks)}")
        print(f"Matched pairs:    {len(result.pairs)}")
        print(f"Missing masks:    {len(result.missing_masks)}")
        print(f"Missing images:   {len(result.missing_images)}")
        print(f"Unreadable files: {len(result.unreadable)}")
    print("\nPH2")
    print(f"Cases:            {len(ph2.cases)}")
    print(f"Images:           {len(ph2.images)}")
    print(f"Masks:            {len(ph2.masks)}")
    print(f"Valid pairs:      {len(ph2.pairs)}")
    print(f"Missing images:   {len(ph2.missing_images)}")
    print(f"Missing masks:    {len(ph2.missing_masks)}")
    print(f"Unreadable files: {len(ph2.unreadable)}")
    print(f"\nFINAL STATUS: {'PASS' if passed else 'FAIL'}")
    print("========================================")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
