# Label-Efficient-Medical-Segmentation
Quantifying how data augmentation reduces annotation requirements in low-data medical image segmentation using lightweight U-Net models on ISIC.

## Dataset protocol

ISIC 2016 is the main dataset. Its 900 official training pairs are split deterministically into a 720-image training pool and a fixed 180-image validation set. The 379 official ISIC test pairs remain a separate in-domain test set and are not used for training or model selection.

All 200 PH2 image-mask pairs are reserved exclusively for final external-domain testing. PH2 is not used for training, validation, augmentation selection, or hyperparameter tuning.

Low-data experiments use nested subsets of 25, 50, 100, 200, and 400 samples drawn from the same 720-image ISIC training pool. Independent nested subset manifests are generated with seeds 42, 123, and 2026. The fixed validation split is shared by every experiment.

To generate standardized 128x128 PNG data and the reproducible manifests, run:

```powershell
python scripts/preprocess_data.py
```

Raw dataset files are read only and remain unchanged.
