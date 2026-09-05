#!/usr/bin/env python3
"""Run one controlled ISIC training-size/seed/augmentation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.augmentations import AUGMENTATION_MODES, SegmentationAugmentation
from src.datasets import SegmentationDataset
from src.models import LightweightUNet
from src.training import BCEDiceLoss, run_epoch, seed_worker, set_deterministic


TRAIN_SIZES = (25, 50, 100, 200, 400)
SUBSET_SEEDS = (42, 123, 2026)
TRAINING_SEED = 42
IMAGE_SIZE = 128
EXPECTED_VAL_SAMPLES = 180
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, choices=TRAIN_SIZES, required=True)
    parser.add_argument("--subset-seed", type=int, choices=SUBSET_SEEDS, required=True)
    parser.add_argument("--augmentation", choices=AUGMENTATION_MODES, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace outputs when a completed result.json already exists",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate arguments and manifests, then exit without training or writing outputs",
    )
    args = parser.parse_args()
    for name in ("epochs", "patience", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def experiment_name(train_size: int, subset_seed: int, augmentation: str) -> str:
    return f"train{train_size}_seed{subset_seed}_{augmentation}"


def manifest_paths(train_size: int, subset_seed: int) -> tuple[Path, Path]:
    manifests = REPOSITORY_ROOT / "data" / "processed" / "manifests"
    train_manifest = manifests / "subsets" / f"seed_{subset_seed}" / f"train_{train_size}.csv"
    validation_manifest = manifests / "isic_val.csv"
    return train_manifest, validation_manifest


def make_loaders(
    train_size: int,
    subset_seed: int,
    augmentation: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, SegmentationDataset, Path, Path]:
    train_manifest, validation_manifest = manifest_paths(train_size, subset_seed)
    train_dataset = SegmentationDataset(
        train_manifest,
        REPOSITORY_ROOT,
        augmentation=SegmentationAugmentation(augmentation),
    )
    validation_dataset = SegmentationDataset(validation_manifest, REPOSITORY_ROOT)
    if len(train_dataset) != train_size:
        raise ValueError(f"Expected {train_size} training samples, found {len(train_dataset)}")
    if len(validation_dataset) != EXPECTED_VAL_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_SAMPLES} validation samples, found {len(validation_dataset)}"
        )
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, validation_dataset, train_manifest, validation_manifest


def write_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def save_history_plot(history: dict[str, list[float]], destination: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="BCE + Dice loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["val_dice"], color="tab:green", label="Validation Dice")
    axes[1].set(xlabel="Epoch", ylabel="Dice", title="Validation Dice", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_prediction_figure(
    model: torch.nn.Module,
    dataset: SegmentationDataset,
    device: torch.device,
    destination: Path,
    count: int = 5,
) -> None:
    generator = np.random.default_rng(TRAINING_SEED)
    indices = generator.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    figure, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), squeeze=False)
    model.eval()
    with torch.inference_mode():
        for row, index in enumerate(indices):
            image, mask, sample_id = dataset[int(index)]
            logits = model(image.unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
            panels = (image.permute(1, 2, 0).numpy(), mask[0].numpy(), prediction)
            for column, (panel, title) in enumerate(
                zip(panels, ("Image", "Ground truth", "Prediction"))
            ):
                axis = axes[row, column]
                axis.imshow(panel, cmap="gray" if column else None, vmin=0, vmax=1)
                axis.set_title(f"{sample_id}\n{title}", fontsize=9)
                axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    name = experiment_name(args.train_size, args.subset_seed, args.augmentation)
    output_dir = REPOSITORY_ROOT / "outputs" / "experiments" / name
    result_path = output_dir / "result.json"
    train_manifest, validation_manifest = manifest_paths(args.train_size, args.subset_seed)

    if not train_manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")
    if not validation_manifest.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {validation_manifest}")
    if args.validate_config:
        train_dataset = SegmentationDataset(train_manifest, REPOSITORY_ROOT)
        validation_dataset = SegmentationDataset(validation_manifest, REPOSITORY_ROOT)
        if len(train_dataset) != args.train_size or len(validation_dataset) != EXPECTED_VAL_SAMPLES:
            raise ValueError(
                f"Unexpected manifest sizes: train={len(train_dataset)}, val={len(validation_dataset)}"
            )
        print(f"Configuration valid: {name}")
        print(f"Training manifest: {train_manifest}")
        print(f"Validation manifest: {validation_manifest}")
        print("No training was started and no output files were written.")
        return 0
    if result_path.exists() and not args.overwrite:
        print(f"SKIP: completed result already exists: {result_path}")
        print("Use --overwrite only if you intentionally want to replace this experiment.")
        return 0

    if result_path.exists() and args.overwrite:
        # Remove only the old completion marker. A failed replacement must not look complete.
        result_path.unlink()
        print(f"Overwrite requested; removed previous completion marker: {result_path}")

    set_deterministic(TRAINING_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    history_plot_path = output_dir / "training_history.png"
    prediction_plot_path = output_dir / "validation_predictions.png"
    config_path = output_dir / "experiment_config.json"
    config = {
        "experiment_name": name,
        "train_size": args.train_size,
        "subset_seed": args.subset_seed,
        "augmentation": args.augmentation,
        "training_seed": TRAINING_SEED,
        "image_size": IMAGE_SIZE,
        "epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "model": "LightweightUNet(base_channels=16)",
        "optimizer": "Adam",
        "loss": "0.5 * BCEWithLogitsLoss + 0.5 * soft Dice loss",
        "train_manifest": train_manifest.relative_to(REPOSITORY_ROOT).as_posix(),
        "validation_manifest": validation_manifest.relative_to(REPOSITORY_ROOT).as_posix(),
    }
    write_json(config_path, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, validation_dataset, _, _ = make_loaders(
        args.train_size,
        args.subset_seed,
        args.augmentation,
        args.batch_size,
        args.num_workers,
    )
    model = LightweightUNet(base_channels=16).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Experiment: {name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {parameter_count:,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(validation_metrics.loss)
        history["val_dice"].append(validation_metrics.dice)
        history["val_iou"].append(validation_metrics.iou)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train loss {train_metrics.loss:.4f} | "
            f"val loss {validation_metrics.loss:.4f} | val Dice {validation_metrics.dice:.4f} | "
            f"val IoU {validation_metrics.iou:.4f}"
        )
        if validation_metrics.dice > best_dice:
            best_dice = validation_metrics.dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_dice": best_dice,
                    "validation_iou": validation_metrics.iou,
                    "config": config,
                    "parameter_count": parameter_count,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    epochs_trained = len(history["train_loss"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation = run_epoch(model, validation_loader, criterion, device)
    history_document: dict[str, Any] = {
        **history,
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "best_validation_iou": best_validation.iou,
        "epochs_trained": epochs_trained,
    }
    write_json(history_path, history_document)
    save_history_plot(history, history_plot_path)
    save_prediction_figure(model, validation_dataset, device, prediction_plot_path)
    result = {
        "train_size": args.train_size,
        "subset_seed": args.subset_seed,
        "augmentation": args.augmentation,
        "best_val_dice": best_dice,
        "best_val_iou": best_validation.iou,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "checkpoint_path": checkpoint_path.relative_to(REPOSITORY_ROOT).as_posix(),
    }
    # Written last: its presence marks the experiment as successfully completed.
    write_json(result_path, result)

    print("\nEXPERIMENT COMPLETE")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
