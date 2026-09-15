#!/usr/bin/env python3
"""Create final paired-effect and annotation-efficiency analyses from test metrics."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPOSITORY_ROOT / "outputs" / "evaluation" / "test_results.csv"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "paper_analysis"
PLOTS_ROOT = OUTPUT_ROOT / "plots"
TRAIN_SIZES = (25, 50, 100, 200)
RECOVERY_SIZES = (25, 50, 100)
SUBSET_SEEDS = (42, 123, 2026)
AUGMENTATIONS = ("none", "geometric", "photometric", "combined")
ACTIVE_AUGMENTATIONS = AUGMENTATIONS[1:]
TEST_DATASETS = ("isic_test", "ph2_test")
SMALL_DENOMINATOR_THRESHOLD = 0.01
COLORS = {
    "none": "#333333",
    "geometric": "#0072B2",
    "photometric": "#E69F00",
    "combined": "#009E73",
}


def read_results() -> list[dict[str, Any]]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Corrected test results not found: {INPUT_PATH}")
    with INPUT_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "train_size",
        "subset_seed",
        "augmentation",
        "test_dataset",
        "dice",
        "iou",
        "precision",
        "recall",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{INPUT_PATH} is empty or missing required columns")
    expected = {
        (size, seed, augmentation, dataset)
        for size in TRAIN_SIZES
        for seed in SUBSET_SEEDS
        for augmentation in AUGMENTATIONS
        for dataset in TEST_DATASETS
    }
    identities = {
        (int(row["train_size"]), int(row["subset_seed"]), row["augmentation"], row["test_dataset"])
        for row in rows
    }
    if identities != expected or len(rows) != len(expected):
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(f"Expected exactly 96 main test rows; missing={missing}, extra={extra}")
    return rows


def lookup_dice(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str, str], float]:
    return {
        (int(row["train_size"]), int(row["subset_seed"]), row["augmentation"], row["test_dataset"]): float(row["dice"])
        for row in rows
    }


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def direction_label(values: list[float]) -> tuple[bool, str]:
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values]
    agree = all(sign == signs[0] for sign in signs)
    direction = "positive" if agree and signs[0] > 0 else "negative" if agree and signs[0] < 0 else "zero" if agree else "mixed"
    return agree, direction


def paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dice = lookup_dice(rows)
    output: list[dict[str, Any]] = []
    for dataset in TEST_DATASETS:
        for size in TRAIN_SIZES:
            baseline_values = [dice[(size, seed, "none", dataset)] for seed in SUBSET_SEEDS]
            for augmentation in ACTIVE_AUGMENTATIONS:
                augmentation_values = [dice[(size, seed, augmentation, dataset)] for seed in SUBSET_SEEDS]
                differences = [aug - base for aug, base in zip(augmentation_values, baseline_values)]
                agree, direction = direction_label(differences)
                output.append(
                    {
                        "test_dataset": dataset,
                        "train_size": size,
                        "augmentation": augmentation,
                        "none_mean_dice": statistics.mean(baseline_values),
                        "none_sd_dice": sample_sd(baseline_values),
                        "augmentation_mean_dice": statistics.mean(augmentation_values),
                        "augmentation_sd_dice": sample_sd(augmentation_values),
                        "seed_42_difference": differences[0],
                        "seed_123_difference": differences[1],
                        "seed_2026_difference": differences[2],
                        "mean_paired_difference": statistics.mean(differences),
                        "sd_paired_difference": sample_sd(differences),
                        "all_seeds_agree": agree,
                        "effect_direction": direction,
                    }
                )
    return output


def recovery_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dice = lookup_dice(rows)
    output: list[dict[str, Any]] = []
    for dataset in TEST_DATASETS:
        for size in RECOVERY_SIZES:
            for augmentation in ACTIVE_AUGMENTATIONS:
                ratios: list[float] = []
                denominators: list[float] = []
                for seed in SUBSET_SEEDS:
                    baseline = dice[(size, seed, "none", dataset)]
                    denominator = dice[(2 * size, seed, "none", dataset)] - baseline
                    numerator = dice[(size, seed, augmentation, dataset)] - baseline
                    denominators.append(denominator)
                    ratios.append(numerator / denominator if denominator != 0 else float("nan"))
                finite = [value for value in ratios if math.isfinite(value)]
                output.append(
                    {
                        "test_dataset": dataset,
                        "train_size": size,
                        "doubled_train_size": 2 * size,
                        "augmentation": augmentation,
                        "seed_42_denominator": denominators[0],
                        "seed_42_recovery_ratio": ratios[0],
                        "seed_123_denominator": denominators[1],
                        "seed_123_recovery_ratio": ratios[1],
                        "seed_2026_denominator": denominators[2],
                        "seed_2026_recovery_ratio": ratios[2],
                        "mean_recovery_ratio": statistics.mean(finite) if finite else float("nan"),
                        "median_recovery_ratio": statistics.median(finite) if finite else float("nan"),
                        "sd_recovery_ratio": sample_sd(finite),
                        "small_denominator_flag": any(abs(value) < SMALL_DENOMINATOR_THRESHOLD for value in denominators),
                        "undefined_ratio_count": len(ratios) - len(finite),
                    }
                )
    return output


def final_summary(rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["test_dataset"], int(row["train_size"]), row["augmentation"])].append(row)
    effect_lookup = {
        (row["test_dataset"], int(row["train_size"]), row["augmentation"]): row
        for row in effects
    }
    output: list[dict[str, Any]] = []
    for dataset in TEST_DATASETS:
        for size in TRAIN_SIZES:
            for augmentation in AUGMENTATIONS:
                group = groups[(dataset, size, augmentation)]
                metrics = {
                    name: [float(row[name]) for row in group]
                    for name in ("dice", "iou", "precision", "recall")
                }
                effect = effect_lookup.get((dataset, size, augmentation))
                output.append(
                    {
                        "test_dataset": dataset,
                        "train_size": size,
                        "augmentation": augmentation,
                        "mean_dice": statistics.mean(metrics["dice"]),
                        "sd_dice": sample_sd(metrics["dice"]),
                        "mean_iou": statistics.mean(metrics["iou"]),
                        "sd_iou": sample_sd(metrics["iou"]),
                        "mean_precision": statistics.mean(metrics["precision"]),
                        "sd_precision": sample_sd(metrics["precision"]),
                        "mean_recall": statistics.mean(metrics["recall"]),
                        "sd_recall": sample_sd(metrics["recall"]),
                        "mean_paired_dice_difference_vs_none": 0.0 if effect is None else effect["mean_paired_difference"],
                        "all_seeds_agree": True if effect is None else effect["all_seeds_agree"],
                        "effect_direction": "reference" if effect is None else effect["effect_direction"],
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def grouped_summary(summary: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {(row["test_dataset"], int(row["train_size"]), row["augmentation"]): row for row in summary}


def plot_learning_curves(summary: list[dict[str, Any]], dataset: str, path: Path) -> None:
    values = grouped_summary(summary)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for augmentation in AUGMENTATIONS:
        means = [float(values[(dataset, size, augmentation)]["mean_dice"]) for size in TRAIN_SIZES]
        standard_deviations = [float(values[(dataset, size, augmentation)]["sd_dice"]) for size in TRAIN_SIZES]
        axis.errorbar(TRAIN_SIZES, means, yerr=standard_deviations, marker="o", capsize=3, linewidth=1.8, color=COLORS[augmentation], label=augmentation)
    axis.set(xlabel="Number of training labels", ylabel="Mean per-image Dice", xticks=TRAIN_SIZES, ylim=(0, 1))
    axis.set_title("ISIC 2016 official test" if dataset == "isic_test" else "PH2 external test")
    axis.grid(alpha=0.22)
    axis.legend(title="Augmentation", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_effects(effects: list[dict[str, Any]], path: Path) -> None:
    lookup = {(row["test_dataset"], int(row["train_size"]), row["augmentation"]): row for row in effects}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, dataset, title in zip(axes, TEST_DATASETS, ("ISIC 2016 test", "PH2 external test")):
        for augmentation in ACTIVE_AUGMENTATIONS:
            means = [float(lookup[(dataset, size, augmentation)]["mean_paired_difference"]) for size in TRAIN_SIZES]
            standard_deviations = [float(lookup[(dataset, size, augmentation)]["sd_paired_difference"]) for size in TRAIN_SIZES]
            axis.errorbar(TRAIN_SIZES, means, yerr=standard_deviations, marker="o", capsize=3, color=COLORS[augmentation], label=augmentation)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Number of training labels", xticks=TRAIN_SIZES)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Paired Dice difference vs none")
    axes[1].legend(title="Augmentation", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_recovery(recovery: list[dict[str, Any]], path: Path) -> None:
    lookup = {(row["test_dataset"], int(row["train_size"]), row["augmentation"]): row for row in recovery}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, dataset, title in zip(axes, TEST_DATASETS, ("ISIC 2016 test", "PH2 external test")):
        for augmentation in ACTIVE_AUGMENTATIONS:
            means = [float(lookup[(dataset, size, augmentation)]["mean_recovery_ratio"]) for size in RECOVERY_SIZES]
            standard_deviations = [float(lookup[(dataset, size, augmentation)]["sd_recovery_ratio"]) for size in RECOVERY_SIZES]
            axis.errorbar(RECOVERY_SIZES, means, yerr=standard_deviations, marker="o", capsize=3, color=COLORS[augmentation], label=augmentation)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.axhline(1, color="gray", linestyle=":", linewidth=1)
        axis.set(title=title, xlabel="Starting label budget", xticks=RECOVERY_SIZES)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Fraction of label-doubling Dice gain recovered")
    axes[1].legend(title="Augmentation", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_results_table(summary: list[dict[str, Any]]) -> None:
    lookup = grouped_summary(summary)
    for dataset, title in zip(TEST_DATASETS, ("ISIC OFFICIAL TEST", "PH2 EXTERNAL TEST")):
        print(f"\n{title}")
        print("size | none          | geometric     | photometric   | combined")
        print("-----+---------------+---------------+---------------+---------------")
        for size in TRAIN_SIZES:
            cells = []
            for augmentation in AUGMENTATIONS:
                row = lookup[(dataset, size, augmentation)]
                cells.append(f"{float(row['mean_dice']):.4f} +/- {float(row['sd_dice']):.4f}")
            print(f"{size:>4} | " + " | ".join(cells))


def main() -> int:
    rows = read_results()
    effects = paired_effects(rows)
    recovery = recovery_analysis(rows)
    summary = final_summary(rows, effects)
    write_csv(OUTPUT_ROOT / "paired_augmentation_effects.csv", effects)
    write_csv(OUTPUT_ROOT / "label_doubling_recovery.csv", recovery)
    write_csv(OUTPUT_ROOT / "final_summary.csv", summary)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(summary, "isic_test", PLOTS_ROOT / "isic_test_learning_curves.png")
    plot_learning_curves(summary, "ph2_test", PLOTS_ROOT / "ph2_test_learning_curves.png")
    plot_effects(effects, PLOTS_ROOT / "augmentation_minus_baseline_dice.png")
    plot_recovery(recovery, PLOTS_ROOT / "label_doubling_recovery.png")
    print_results_table(summary)
    flagged = [row for row in recovery if row["small_denominator_flag"]]
    print(f"\nRecovery rows with |label-doubling gain| < {SMALL_DENOMINATOR_THRESHOLD:.2f}: {len(flagged)}")
    print("Recovery ratios are not clipped; negative values and values above 1 are retained.")
    print(f"Saved paper analysis: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
