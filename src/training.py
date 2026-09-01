"""Losses, metrics, and reusable training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


SMOOTH = 1e-6


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice_loss = 1.0 - ((2.0 * intersection + SMOOTH) / (denominator + SMOOTH)).mean()
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * dice_loss


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.sigmoid(logits) >= 0.5
    targets_bool = targets >= 0.5
    dimensions = (1, 2, 3)
    intersection = (predictions & targets_bool).sum(dim=dimensions).float()
    prediction_area = predictions.sum(dim=dimensions).float()
    target_area = targets_bool.sum(dim=dimensions).float()
    union = prediction_area + target_area - intersection
    dice = (2.0 * intersection + SMOOTH) / (prediction_area + target_area + SMOOTH)
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return dice, iou


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    dice: float
    iou: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            dice, iou = binary_metrics(logits.detach(), masks)
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_dice += dice.sum().item()
            total_iou += iou.sum().item()
    if total_samples == 0:
        raise ValueError("DataLoader produced no samples")
    return EpochMetrics(
        loss=total_loss / total_samples,
        dice=total_dice / total_samples,
        iou=total_iou / total_samples,
    )
