"""
evaluation/metrics.py
─────────────────────
Inference, class-weight computation, and misclassification analysis.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from brain_tumor.config import (
    AMP_DTYPE,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    REPORT_DIR,
    SEED,
    USE_AMP,
)


def compute_class_weights(
    dataset,
    indices,
    num_classes: int = NUM_CLASSES,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Compute balanced class weights from a training split and return them as a
    ``torch.Tensor`` on *device* (for use with ``CrossEntropyLoss``).

    Parameters
    ----------
    dataset : BrainTumorDataset  (has a ``.samples`` list of (path, label) tuples)
    indices : array-like of integer indices into *dataset.samples*
    """
    labels = [dataset.samples[i][1] for i in indices]
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(labels),
    )
    weights = torch.tensor(cw, dtype=torch.float32, device=device)

    print("Class weights:")
    for i, cls in enumerate(CLASS_NAMES[:num_classes]):
        print(f"  [{i}] {cls:<12} count={labels.count(i):>4}  weight={cw[i]:.4f}")

    return weights


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run inference over *loader* and collect ground-truth labels, predicted
    labels, class probabilities, and image paths.

    Returns
    -------
    y_true, y_pred, y_prob, img_paths
    """
    model.eval()
    labels_all, preds_all, probs_all, paths_all = [], [], [], []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
        if USE_AMP and device.type == "cuda"
        else nullcontext()
    )

    for imgs, labels, paths in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        with amp_ctx:
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        labels_all.extend(labels.numpy())
        preds_all.extend(logits.argmax(1).cpu().numpy())
        probs_all.extend(probs)
        paths_all.extend(paths)

    return (
        np.array(labels_all),
        np.array(preds_all),
        np.array(probs_all),
        paths_all,
    )


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> None:
    """Print a formatted sklearn classification report."""
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def build_error_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    img_paths: list[str],
    class_names: list[str] = CLASS_NAMES,
    save_path: Path = REPORT_DIR / "misclassified.csv",
) -> pd.DataFrame:
    """
    Build a DataFrame of misclassified samples, save it to CSV, and return it.

    Columns: path, true_label, pred_label, pred_confidence, true_confidence
    """
    errors = [
        {
            "path"            : img_paths[i],
            "true_label"      : class_names[y_true[i]],
            "pred_label"      : class_names[y_pred[i]],
            "pred_confidence" : float(y_prob[i, y_pred[i]]),
            "true_confidence" : float(y_prob[i, y_true[i]]),
        }
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    df = pd.DataFrame(errors)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Misclassified: {len(df)} / {len(y_true)}")
    return df
