#!/usr/bin/env python3
"""Train a 25-image, seed-42 ISIC pilot with selectable augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


SEED = 42
IMAGE_SIZE = 128
EXPECTED_TRAIN_SAMPLES = 25
EXPECTED_VAL_SAMPLES = 180
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augmentation", choices=AUGMENTATION_MODES, default="none")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "baseline" / "isic_train25_seed42",
    )
    args = parser.parse_args()
    for name in ("epochs", "patience", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.num_workers < 0:
        parser.error("learning rate must be positive and num-workers cannot be negative")
    return args


def make_loaders(
    batch_size: int, num_workers: int, augmentation: str
) -> tuple[DataLoader, DataLoader, SegmentationDataset]:
    train_manifest = (
        REPOSITORY_ROOT
        / "data"
        / "processed"
        / "manifests"
        / "subsets"
        / "seed_42"
        / "train_25.csv"
    )
    val_manifest = REPOSITORY_ROOT / "data" / "processed" / "manifests" / "isic_val.csv"
    train_dataset = SegmentationDataset(
        train_manifest,
        REPOSITORY_ROOT,
        augmentation=SegmentationAugmentation(augmentation),
    )
    val_dataset = SegmentationDataset(val_manifest, REPOSITORY_ROOT)
    if len(train_dataset) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_SAMPLES} training samples, found {len(train_dataset)}")
    if len(val_dataset) != EXPECTED_VAL_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_VAL_SAMPLES} validation samples, found {len(val_dataset)}")
    generator = torch.Generator().manual_seed(SEED)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, val_dataset


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
    generator = np.random.default_rng(SEED)
    indices = generator.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    figure, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), squeeze=False)
    model.eval()
    with torch.inference_mode():
        for row, index in enumerate(indices):
            image, mask, sample_id = dataset[int(index)]
            logits = model(image.unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
            panels = (image.permute(1, 2, 0).numpy(), mask[0].numpy(), prediction)
            titles = ("Image", "Ground truth", "Prediction")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axis = axes[row, column]
                axis.imshow(panel, cmap="gray" if column else None, vmin=0, vmax=1)
                axis.set_title(f"{sample_id}\n{title}", fontsize=9)
                axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    set_deterministic(SEED)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_plot_path = output_dir / "training_history.png"
    prediction_plot_path = output_dir / "validation_predictions.png"
    history_path = output_dir / "history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, val_dataset = make_loaders(
        args.batch_size, args.num_workers, args.augmentation
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

    print(f"Device: {device}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Trainable parameters: {parameter_count:,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)
        history["val_dice"].append(val_metrics.dice)
        history["val_iou"].append(val_metrics.iou)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train loss {train_metrics.loss:.4f} | "
            f"val loss {val_metrics.loss:.4f} | val Dice {val_metrics.dice:.4f} | "
            f"val IoU {val_metrics.iou:.4f}"
        )
        if val_metrics.dice > best_dice:
            best_dice = val_metrics.dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_dice": best_dice,
                    "validation_iou": val_metrics.iou,
                    "config": vars(args),
                    "seed": SEED,
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
    final_validation = run_epoch(model, val_loader, criterion, device)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **history,
                "best_epoch": best_epoch,
                "best_validation_dice": best_dice,
                "best_validation_iou": final_validation.iou,
                "best_checkpoint_validation_iou": final_validation.iou,
                "epochs_trained": epochs_trained,
                "augmentation": args.augmentation,
            },
            handle,
            indent=2,
        )
    save_history_plot(history, history_plot_path)
    save_prediction_figure(model, val_dataset, device, prediction_plot_path)

    print("\n========================================")
    print("BASELINE TRAINING SUMMARY")
    print("========================================")
    print(f"Model parameters:       {parameter_count:,}")
    print(f"Epochs trained:         {epochs_trained}")
    print(f"Best validation Dice:   {best_dice:.4f} (epoch {best_epoch})")
    print(f"Final validation IoU:   {final_validation.iou:.4f}")
    print(f"Saved checkpoint:       {checkpoint_path}")
    print(f"Saved history plot:     {history_plot_path}")
    print(f"Saved prediction plot:  {prediction_plot_path}")
    print(f"Saved metric history:   {history_path}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
