"""
models/checkpoint.py
────────────────────
Helpers for saving and loading model checkpoints with full metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from brain_tumor.config import (
    CLASS_NAMES,
    DEVICE,
    DROPOUT,
    IMG_MEAN,
    IMG_STD,
    IMAGE_SIZE,
    NUM_CLASSES,
    REPORT_DIR,
)
from brain_tumor.models.classifier import BrainTumorClassifier


def save_model(
    model: BrainTumorClassifier,
    path: Path,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """
    Save model state dict plus all metadata needed for inference.

    Parameters
    ----------
    model     : trained BrainTumorClassifier
    path      : destination .pth file
    extra_meta: any additional key/value pairs to embed in the checkpoint
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_names"     : CLASS_NAMES,
        "image_size"      : IMAGE_SIZE,
        "num_classes"     : NUM_CLASSES,
        "dropout"         : DROPOUT,
        "mean"            : IMG_MEAN,
        "std"             : IMG_STD,
    }
    if extra_meta:
        payload.update(extra_meta)

    torch.save(payload, path)
    size_mb = path.stat().st_size / 1e6
    print(f"Model saved → {path}  ({size_mb:.1f} MB)")


def load_model(path: Path, device: torch.device = DEVICE) -> BrainTumorClassifier:
    """
    Restore a ``BrainTumorClassifier`` from a checkpoint produced by
    :func:`save_model`.

    Returns the model in eval mode on *device*.
    """
    ckpt = torch.load(path, map_location=device)
    model = BrainTumorClassifier(
        num_classes=ckpt["num_classes"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def save_metrics(
    metrics: dict[str, Any],
    path: Path = REPORT_DIR / "metrics_summary.json",
) -> None:
    """Persist an evaluation metrics dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved → {path}")
