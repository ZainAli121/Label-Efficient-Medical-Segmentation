"""Mild paired augmentations for dermoscopic lesion segmentation."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


AugmentationMode = Literal["none", "geometric", "photometric", "combined"]
AUGMENTATION_MODES = ("none", "geometric", "photometric", "combined")


class SegmentationAugmentation:
    """Apply mild augmentation while keeping image and mask geometry synchronized."""

    def __init__(self, mode: AugmentationMode) -> None:
        if mode not in AUGMENTATION_MODES:
            raise ValueError(f"Unknown augmentation mode: {mode}")
        self.mode = mode

    @staticmethod
    def _affine_coefficients(
        size: tuple[int, int], angle_degrees: float, scale: float, tx: float, ty: float
    ) -> tuple[float, float, float, float, float, float]:
        """Return inverse affine coefficients expected by Pillow."""
        width, height = size
        cx, cy = width / 2.0, height / 2.0
        angle = math.radians(angle_degrees)
        cosine = math.cos(angle) / scale
        sine = math.sin(angle) / scale
        a, b = cosine, sine
        d, e = -sine, cosine
        c = cx - a * (cx + tx) - b * (cy + ty)
        f = cy - d * (cx + tx) - e * (cy + ty)
        return a, b, c, d, e, f

    def _geometric(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.8:
            width, height = image.size
            coefficients = self._affine_coefficients(
                image.size,
                angle_degrees=random.uniform(-12.0, 12.0),
                scale=random.uniform(0.92, 1.08),
                tx=random.uniform(-0.04, 0.04) * width,
                ty=random.uniform(-0.04, 0.04) * height,
            )
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            mask = mask.transform(
                mask.size,
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
        return image, mask

    @staticmethod
    def _photometric(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.mode in ("geometric", "combined"):
            image, mask = self._geometric(image, mask)
        if self.mode in ("photometric", "combined"):
            image = self._photometric(image)
        return image, mask


def verify_geometric_alignment(output_path: Path, seed: int = 42, trials: int = 24) -> float:
    """Programmatically verify paired geometry and save representative examples."""
    previous_state = random.getstate()
    random.seed(seed)
    augmenter = SegmentationAugmentation("geometric")
    minimum_iou = 1.0
    examples: list[tuple[Image.Image, Image.Image]] = []
    try:
        for trial in range(trials):
            mask = Image.new("L", (128, 128), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((29, 37, 99, 91), fill=255)
            image = Image.new("RGB", mask.size, (0, 0, 0))
            image_array = np.asarray(image).copy()
            image_array[np.asarray(mask) > 0] = (255, 255, 255)
            image = Image.fromarray(image_array, mode="RGB")
            transformed_image, transformed_mask = augmenter(image, mask)
            image_region = np.asarray(transformed_image)[:, :, 0] >= 127
            mask_region = np.asarray(transformed_mask) > 0
            intersection = np.logical_and(image_region, mask_region).sum()
            union = np.logical_or(image_region, mask_region).sum()
            iou = float(intersection / union) if union else 1.0
            minimum_iou = min(minimum_iou, iou)
            if trial < 4:
                examples.append((transformed_image, transformed_mask))
    finally:
        random.setstate(previous_state)
    if minimum_iou < 0.94:
        raise RuntimeError(f"Geometric alignment check failed: minimum synthetic IoU={minimum_iou:.4f}")
    canvas = Image.new("RGB", (128 * 3, 128 * len(examples)), "white")
    for row, (image, mask) in enumerate(examples):
        mask_array = np.asarray(mask) > 0
        overlay = np.asarray(image).copy()
        overlay[mask_array] = (0.6 * overlay[mask_array] + 0.4 * np.array([255, 0, 0])).astype(
            np.uint8
        )
        canvas.paste(image, (0, row * 128))
        canvas.paste(mask.convert("RGB"), (128, row * 128))
        canvas.paste(Image.fromarray(overlay), (256, row * 128))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return minimum_iou
