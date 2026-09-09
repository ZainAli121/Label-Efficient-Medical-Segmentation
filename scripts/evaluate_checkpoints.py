#!/usr/bin/env python3
"""Evaluate the 48 completed ISIC experiments on fixed ISIC and PH2 test sets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.datasets import SegmentationDataset
from src.models import LightweightUNet
from src.training import SMOOTH, seed_worker, set_deterministic


TRAIN_SIZES = (25, 50, 100, 200)
SUBSET_SEEDS = (42, 123, 2026)
AUGMENTATIONS = ("none", "geometric", "photometric", "combined")
TEST_DATASETS = {
    "isic_test": ("data/processed/manifests/isic_test.csv", 379),
    "ph2_test": ("data/processed/manifests/ph2_test.csv", 200),
}
EVALUATION_SEED = 42
PREDICTION_THRESHOLD = 0.5
RESULT_FIELDS = (
    "train_size",
    "subset_seed",
    "augmentation",
    "test_dataset",
    "dice",
    "iou",
    "precision",
    "recall",
    "checkpoint_path",
)
SUMMARY_FIELDS = (
    "train_size",
    "augmentation",
    "test_dataset",
    "seed_count",
    "mean_dice",
    "std_dice",
    "mean_iou",
    "std_iou",
    "mean_precision",
    "mean_recall",
)
COLORS = {
    "none": "#4c4c4c",
    "geometric": "#1f77b4",
    "photometric": "#ff7f0e",
    "combined": "#2ca02c",
}


@dataclass(frozen=True)
class Experiment:
    train_size: int
    subset_seed: int
    augmentation: str
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing evaluation CSVs instead of stopping",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all 48 checkpoints and test manifests without running inference",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def expected_identities() -> set[tuple[int, int, str]]:
    return {
        (size, seed, augmentation)
        for size in TRAIN_SIZES
        for seed in SUBSET_SEEDS
        for augmentation in AUGMENTATIONS
    }


def resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def discover_experiments(experiments_root: Path) -> tuple[list[Experiment], list[str]]:
    expected = expected_identities()
    discovered: dict[tuple[int, int, str], Experiment] = {}
    errors: list[str] = []
    for result_path in sorted(experiments_root.glob("train*_seed*_*/result.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            identity = (
                int(result["train_size"]),
                int(result["subset_seed"]),
                str(result["augmentation"]),
            )
            if identity not in expected:
                # This intentionally excludes all 400-image and other non-main experiments.
                continue
            if identity in discovered:
                errors.append(f"Duplicate completed experiment: {identity}")
                continue
            checkpoint = resolve_repository_path(str(result["checkpoint_path"]))
            if not checkpoint.is_file():
                errors.append(f"Missing checkpoint for {identity}: {checkpoint}")
                continue
            discovered[identity] = Experiment(*identity, checkpoint)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid result file {result_path}: {exc}")
    for identity in sorted(expected - set(discovered)):
        errors.append(f"Missing completed experiment/checkpoint: {identity}")
    experiments = [discovered[identity] for identity in sorted(discovered)]
    return experiments, errors


def make_test_loaders(batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, (manifest_value, expected_count) in TEST_DATASETS.items():
        dataset = SegmentationDataset(REPOSITORY_ROOT / manifest_value, REPOSITORY_ROOT)
        if len(dataset) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} samples, found {len(dataset)}")
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            drop_last=False,
        )
    return loaders


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Return macro metrics: calculate each image independently, then average."""
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    sample_count = 0
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = masks.to(device, non_blocking=True) >= 0.5
            predictions = torch.sigmoid(model(images)) >= PREDICTION_THRESHOLD
            dimensions = (1, 2, 3)
            true_positive = (predictions & targets).sum(dim=dimensions).float()
            predicted_positive = predictions.sum(dim=dimensions).float()
            actual_positive = targets.sum(dim=dimensions).float()
            union = predicted_positive + actual_positive - true_positive
            metrics = {
                "dice": (2.0 * true_positive + SMOOTH)
                / (predicted_positive + actual_positive + SMOOTH),
                "iou": (true_positive + SMOOTH) / (union + SMOOTH),
                "precision": (true_positive + SMOOTH) / (predicted_positive + SMOOTH),
                "recall": (true_positive + SMOOTH) / (actual_positive + SMOOTH),
            }
            for name, values in metrics.items():
                totals[name] += values.sum().item()
            sample_count += images.size(0)
    if sample_count == 0:
        raise ValueError("Test DataLoader produced no samples")
    return {name: value / sample_count for name, value in totals.items()}


def write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["train_size"]), str(row["augmentation"]), str(row["test_dataset"]))].append(row)
    summary: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[2], item[0], AUGMENTATIONS.index(item[1]))):
        train_size, augmentation, test_dataset = key
        group = groups[key]
        if len(group) != len(SUBSET_SEEDS):
            raise ValueError(f"Expected 3 seeds for {key}, found {len(group)}")
        values = {metric: [float(row[metric]) for row in group] for metric in ("dice", "iou", "precision", "recall")}
        summary.append(
            {
                "train_size": train_size,
                "augmentation": augmentation,
                "test_dataset": test_dataset,
                "seed_count": len(group),
                "mean_dice": statistics.mean(values["dice"]),
                "std_dice": statistics.stdev(values["dice"]),
                "mean_iou": statistics.mean(values["iou"]),
                "std_iou": statistics.stdev(values["iou"]),
                "mean_precision": statistics.mean(values["precision"]),
                "mean_recall": statistics.mean(values["recall"]),
            }
        )
    return summary


def summary_lookup(summary: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        (int(row["train_size"]), str(row["augmentation"]), str(row["test_dataset"])): row
        for row in summary
    }


def plot_learning_curves(summary: list[dict[str, Any]], dataset_name: str, destination: Path) -> None:
    lookup = summary_lookup(summary)
    figure, axis = plt.subplots(figsize=(8, 5))
    for augmentation in AUGMENTATIONS:
        means = [float(lookup[(size, augmentation, dataset_name)]["mean_dice"]) for size in TRAIN_SIZES]
        stds = [float(lookup[(size, augmentation, dataset_name)]["std_dice"]) for size in TRAIN_SIZES]
        axis.plot(TRAIN_SIZES, means, marker="o", label=augmentation, color=COLORS[augmentation])
        axis.fill_between(TRAIN_SIZES, np.subtract(means, stds), np.add(means, stds), color=COLORS[augmentation], alpha=0.14)
    axis.set(
        xlabel="Number of training labels",
        ylabel="Mean per-image Dice",
        title=f"{'ISIC 2016' if dataset_name == 'isic_test' else 'PH2 external'} test learning curves",
        xticks=TRAIN_SIZES,
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Augmentation")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def paired_values(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str, str], float]:
    return {
        (int(row["train_size"]), int(row["subset_seed"]), str(row["augmentation"]), str(row["test_dataset"])): float(row["dice"])
        for row in rows
    }


def plot_augmentation_effect(rows: list[dict[str, Any]], destination: Path) -> None:
    lookup = paired_values(rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, dataset_name, title in zip(axes, TEST_DATASETS, ("ISIC 2016 test", "PH2 external test")):
        for augmentation in AUGMENTATIONS[1:]:
            means, stds = [], []
            for size in TRAIN_SIZES:
                differences = [
                    lookup[(size, seed, augmentation, dataset_name)]
                    - lookup[(size, seed, "none", dataset_name)]
                    for seed in SUBSET_SEEDS
                ]
                means.append(statistics.mean(differences))
                stds.append(statistics.stdev(differences))
            axis.errorbar(TRAIN_SIZES, means, yerr=stds, marker="o", capsize=3, label=augmentation, color=COLORS[augmentation])
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set(xlabel="Number of training labels", title=title, xticks=TRAIN_SIZES)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Paired Dice difference vs no augmentation")
    axes[1].legend(title="Augmentation")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_generalization_gap(rows: list[dict[str, Any]], destination: Path) -> None:
    lookup = paired_values(rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    for augmentation in AUGMENTATIONS:
        means, stds = [], []
        for size in TRAIN_SIZES:
            gaps = [
                lookup[(size, seed, augmentation, "isic_test")]
                - lookup[(size, seed, augmentation, "ph2_test")]
                for seed in SUBSET_SEEDS
            ]
            means.append(statistics.mean(gaps))
            stds.append(statistics.stdev(gaps))
        axis.errorbar(TRAIN_SIZES, means, yerr=stds, marker="o", capsize=3, label=augmentation, color=COLORS[augmentation])
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        xlabel="Number of training labels",
        ylabel="ISIC test Dice - PH2 test Dice",
        title="In-domain to external-domain generalization gap",
        xticks=TRAIN_SIZES,
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Augmentation")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_terminal_summary(summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lookup = summary_lookup(summary)
    for dataset_name, title in (("isic_test", "ISIC TEST"), ("ph2_test", "PH2 TEST")):
        print(f"\n{title}")
        print("size  augmentation   mean Dice +/- std")
        print("----  -------------  -----------------")
        for size in TRAIN_SIZES:
            for augmentation in AUGMENTATIONS:
                row = lookup[(size, augmentation, dataset_name)]
                print(f"{size:>4}  {augmentation:<13}  {float(row['mean_dice']):.4f} +/- {float(row['std_dice']):.4f}")
            best = max(AUGMENTATIONS, key=lambda aug: float(lookup[(size, aug, dataset_name)]["mean_dice"]))
            best_row = lookup[(size, best, dataset_name)]
            print(f"      Highest mean: {best} ({float(best_row['mean_dice']):.4f} +/- {float(best_row['std_dice']):.4f})")

    paired = paired_values(rows)
    print("\nDESCRIPTIVE AUGMENTATION CHECKS")
    for dataset_name in TEST_DATASETS:
        consistently_positive: list[str] = []
        for augmentation in AUGMENTATIONS[1:]:
            size_means = []
            for size in TRAIN_SIZES:
                differences = [paired[(size, seed, augmentation, dataset_name)] - paired[(size, seed, "none", dataset_name)] for seed in SUBSET_SEEDS]
                size_means.append(statistics.mean(differences))
            if all(value > 0 for value in size_means):
                consistently_positive.append(augmentation)
        message = ", ".join(consistently_positive) if consistently_positive else "none"
        print(f"{dataset_name}: augmentation modes with positive mean Dice change at every size: {message}.")

    domain_differences: list[float] = []
    isic_effects: list[float] = []
    ph2_effects: list[float] = []
    for size in TRAIN_SIZES:
        for seed in SUBSET_SEEDS:
            for augmentation in AUGMENTATIONS[1:]:
                isic_delta = paired[(size, seed, augmentation, "isic_test")] - paired[(size, seed, "none", "isic_test")]
                ph2_delta = paired[(size, seed, augmentation, "ph2_test")] - paired[(size, seed, "none", "ph2_test")]
                isic_effects.append(isic_delta)
                ph2_effects.append(ph2_delta)
                domain_differences.append(ph2_delta - isic_delta)
    print(f"Mean paired augmentation Dice change: ISIC={statistics.mean(isic_effects):+.4f}, PH2={statistics.mean(ph2_effects):+.4f}.")
    print(f"Mean PH2-minus-ISIC augmentation effect: {statistics.mean(domain_differences):+.4f} (SD {statistics.stdev(domain_differences):.4f}).")
    print("These are descriptive comparisons; effects small relative to seed variation should not be interpreted as reliable improvements.")


def main() -> int:
    args = parse_args()
    experiments_root = REPOSITORY_ROOT / "outputs" / "experiments"
    evaluation_root = REPOSITORY_ROOT / "outputs" / "evaluation"
    results_path = evaluation_root / "test_results.csv"
    summary_path = evaluation_root / "test_results_summary.csv"
    plots_root = evaluation_root / "plots"
    if (results_path.exists() or summary_path.exists()) and not args.overwrite and not args.validate_only:
        raise FileExistsError("Evaluation outputs already exist. Use --overwrite to replace them intentionally.")

    experiments, errors = discover_experiments(experiments_root)
    if errors:
        print("CHECKPOINT DISCOVERY FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    loaders = make_test_loaders(args.batch_size, args.num_workers)
    print(f"Validated {len(experiments)} required checkpoints; 400-image experiments are excluded.")
    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)} samples")
    if args.validate_only:
        print("Validation-only mode complete. No inference or output writing was performed.")
        return 0

    set_deterministic(EVALUATION_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    rows: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments, start=1):
        checkpoint = torch.load(experiment.checkpoint_path, map_location=device, weights_only=False)
        model = LightweightUNet(base_channels=16).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        for dataset_name, loader in loaders.items():
            metrics = evaluate_model(model, loader, device)
            rows.append(
                {
                    "train_size": experiment.train_size,
                    "subset_seed": experiment.subset_seed,
                    "augmentation": experiment.augmentation,
                    "test_dataset": dataset_name,
                    **metrics,
                    "checkpoint_path": experiment.checkpoint_path.relative_to(REPOSITORY_ROOT).as_posix(),
                }
            )
        print(f"[{index:02d}/{len(experiments)}] evaluated train{experiment.train_size} seed{experiment.subset_seed} {experiment.augmentation}")
    write_csv(results_path, RESULT_FIELDS, rows)
    summary = aggregate_results(rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary)
    plots_root.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(summary, "isic_test", plots_root / "isic_test_learning_curves.png")
    plot_learning_curves(summary, "ph2_test", plots_root / "ph2_test_learning_curves.png")
    plot_augmentation_effect(rows, plots_root / "augmentation_effect.png")
    plot_generalization_gap(rows, plots_root / "generalization_gap.png")
    print_terminal_summary(summary, rows)
    print(f"\nDetailed results: {results_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Plots:            {plots_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
