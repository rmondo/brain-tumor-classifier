"""
data/dataset.py
───────────────
Dataset class, image transforms, DataLoader factory, and dataset download
helpers for the Brain Tumor MRI Classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from brain_tumor.config import (
    BATCH_SIZE,
    HAS_CUDA,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    LABEL_MAP,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    SEED,
)

# ── Supported image extensions ────────────────────────────────────────────────
_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = IMG_SIZE) -> T.Compose:
    """Augmented transform used during training."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def get_val_transform(image_size: int = IMG_SIZE) -> T.Compose:
    """Deterministic transform used for validation and test."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    Reads a directory of class sub-folders and maps folder names to integer
    labels via *label_map*.

    Each ``__getitem__`` returns ``(image_tensor, label, path_str)``.
    The path is included so downstream evaluation code can open the original
    file for Grad-CAM or misclassification panels.
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[T.Compose] = None,
        label_map: Optional[dict[str, int]] = None,
    ) -> None:
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = self.label_map.get(cls_dir.name.lower())
            if label is None:
                print(f"  [warn] Unknown class folder: {cls_dir.name!r}")
                continue
            self.samples.extend(
                (p, label)
                for p in cls_dir.iterdir()
                if p.suffix in _EXTS
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    train_root: Path,
    test_root: Path,
    val_split: float = 0.15,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    persistent_workers: bool = PERSISTENT_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, BrainTumorDataset]:
    """
    Build train, val, and test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, full_train_aug
        ``full_train_aug`` is the un-split augmented training dataset; its
        ``.samples`` attribute is used by the class-weight computation.
    """
    train_transform = get_train_transform()
    val_transform   = get_val_transform()

    full_train_aug = BrainTumorDataset(train_root, transform=train_transform)
    full_train_val = BrainTumorDataset(train_root, transform=val_transform)
    test_ds        = BrainTumorDataset(test_root,  transform=val_transform)

    indices = np.arange(len(full_train_aug))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, shuffle=True
    )

    train_ds = Subset(full_train_aug, train_idx.tolist())
    val_ds   = Subset(full_train_val, val_idx.tolist())

    loader_kw: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kw["persistent_workers"] = persistent_workers
        if HAS_CUDA:
            loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, full_train_aug


# ── Kaggle download ───────────────────────────────────────────────────────────

def download_dataset(
    data_dir: Path,
    dataset_slug: str,
) -> tuple[Path, Path]:
    """
    Download and unzip the Kaggle dataset from Kaggle, then locate the
    Training and Testing roots.

    The download is skipped only when ``data_dir`` already exists **and**
    contains a ``Training/`` sub-folder with at least one image.  An empty
    directory created by ``make_dirs()`` is not treated as "already present"
    so the dataset will always be fetched on a fresh setup.

    Credentials must be in the environment (KAGGLE_USERNAME + KAGGLE_KEY) or
    in ``~/.kaggle/kaggle.json`` before calling this function.

    Returns
    -------
    train_root, test_root
    """
    import os
    import kaggle  # imported lazily so the module works without kaggle installed

    # ── Pre-flight credential check ───────────────────────────────────────────
    _user = os.environ.get("KAGGLE_USERNAME", "").strip()
    _key  = os.environ.get("KAGGLE_KEY", "").strip()
    if not _user or not _key:
        from pathlib import Path as _Path
        _kj = _Path.home() / ".kaggle" / "kaggle.json"
        if not _kj.exists():
            raise RuntimeError(
                "Kaggle credentials not found.\n"
                "Either set KAGGLE_USERNAME and KAGGLE_KEY environment "
                "variables, or place your kaggle.json at ~/.kaggle/kaggle.json.\n"
                "Run Section 2 (Kaggle Authentication) first."
            )

    # ── Decide whether a download is actually needed ──────────────────────────
    # A directory is considered "already downloaded" only when it contains a
    # Training/ sub-folder that itself contains at least one image file.
    # An empty directory (created by make_dirs()) does NOT count.
    _training_candidate = data_dir / "Training"
    _already_present = (
        _training_candidate.exists()
        and any(
            True for p in _training_candidate.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    )

    if _already_present:
        print(f"Dataset already present ✔  ({data_dir})")
    else:
        print(f"Downloading {dataset_slug} from Kaggle …")
        print(f"  Destination: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset_slug, path=str(data_dir), unzip=True
        )
        print("Download complete ✔")

    matches = list(data_dir.rglob("Training"))
    if not matches:
        raise FileNotFoundError(
            f"'Training' folder not found inside {data_dir}.\n"
            "The download may have failed — check your Kaggle credentials "
            "and that the dataset slug is correct: " + repr(dataset_slug)
        )

    train_root = matches[0]
    test_root  = train_root.parent / "Testing"

    print(f"\nTrain root : {train_root}")
    print(f"Test root  : {test_root}")

    # Per-class image counts
    print("\nClass distribution:")
    for split, root in [("Train", train_root), ("Test", test_root)]:
        for cls_dir in sorted(root.iterdir()):
            n = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            print(f"  {split}/{cls_dir.name:<22} {n:>4} images")

    return train_root, test_root
