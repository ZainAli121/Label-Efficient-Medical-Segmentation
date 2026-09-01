"""Manifest-backed datasets for segmentation experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.augmentations import SegmentationAugmentation


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def _resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load preprocessed RGB images and binary masks listed in a CSV manifest."""

    REQUIRED_COLUMNS = {"sample_id", "image_path", "mask_path"}

    def __init__(
        self,
        manifest_path: Path,
        repository_root: Path,
        augmentation: SegmentationAugmentation | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repository_root = Path(repository_root)
        self.augmentation = augmentation
        self.samples = self._read_manifest()

    def _read_manifest(self) -> list[SegmentationSample]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        samples: list[SegmentationSample] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Manifest {self.manifest_path} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                sample_id = (row.get("sample_id") or "").strip()
                image_value = (row.get("image_path") or "").strip()
                mask_value = (row.get("mask_path") or "").strip()
                if not sample_id or not image_value or not mask_value:
                    raise ValueError(
                        f"Manifest {self.manifest_path}, line {line_number}: blank required value"
                    )
                samples.append(
                    SegmentationSample(
                        sample_id=sample_id,
                        image_path=_resolve_path(image_value, self.repository_root),
                        mask_path=_resolve_path(mask_value, self.repository_root),
                    )
                )
        if not samples:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")
        ids = [sample.sample_id.casefold() for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Manifest contains duplicate sample IDs: {self.manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image_file:
            image_pil = image_file.convert("RGB")
        with Image.open(sample.mask_path) as mask_file:
            mask_pil = mask_file.convert("L")
        if self.augmentation is not None:
            image_pil, mask_pil = self.augmentation(image_pil, mask_pil)
        image = np.asarray(image_pil, dtype=np.float32).copy() / 255.0
        mask = np.asarray(mask_pil, dtype=np.uint8).copy()
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Dimension mismatch for {sample.sample_id}: image={image.shape[:2]}, mask={mask.shape}"
            )
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, sample.sample_id
