"""
evaluation/plots.py
───────────────────
All visualisation functions: training curves, confusion matrix, ROC curves,
misclassification panel, and augmented-sample preview.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from brain_tumor.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMG_MEAN,
    IMG_STD,
    NUM_CLASSES,
    REPORT_DIR,
    SEED,
)

_ROC_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]


# ── Training curves ───────────────────────────────────────────────────────────

def plot_history(
    h1: dict,
    h2: dict,
    save_path: Path = REPORT_DIR / "training_curves.png",
) -> None:
    """
    Overlay Stage-1 and Stage-2 loss & accuracy curves with a red divider at
    the fine-tuning boundary.
    """
    s1 = len(h1["train_loss"])
    s2 = len(h2["train_loss"])
    epochs   = list(range(1, s1 + s2 + 1))
    tr_loss  = h1["train_loss"] + h2["train_loss"]
    vl_loss  = h1["val_loss"]   + h2["val_loss"]
    tr_acc   = h1["train_acc"]  + h2["train_acc"]
    vl_acc   = h1["val_acc"]    + h2["val_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tr, vl, title in zip(
        axes,
        [tr_loss, tr_acc],
        [vl_loss, vl_acc],
        ["Loss", "Accuracy"],
    ):
        ax.plot(epochs, tr, label="Train",      linewidth=2)
        ax.plot(epochs, vl, label="Validation", linewidth=2, linestyle="--")
        ax.axvline(s1 + 0.5, color="red", linestyle=":", linewidth=1.5,
                   label="Fine-tune start")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Training History — Stage 1 → Stage 2",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    save_path: Path = REPORT_DIR / "confusion_matrix.png",
) -> None:
    """Plot raw-count and row-normalised confusion matrices side-by-side."""
    cm      = confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Row-Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    colors: list[str] = _ROC_COLORS,
    save_path: Path = REPORT_DIR / "roc_curves.png",
) -> dict[str, float]:
    """
    Plot per-class One-vs-Rest ROC curves.

    Returns
    -------
    roc_scores : dict mapping class name → AUC
    """
    y_true_oh  = label_binarize(y_true, classes=np.arange(len(class_names)))
    roc_scores = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (cls, color, ax) in enumerate(zip(class_names, colors, axes.ravel())):
        fpr, tpr, _ = roc_curve(y_true_oh[:, idx], y_prob[:, idx])
        roc_scores[cls] = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"AUC = {roc_scores[cls]:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"ROC — {cls}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    macro_auc = float(np.mean(list(roc_scores.values())))
    plt.suptitle("Per-Class ROC Curves (One-vs-Rest)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Macro-average AUC: {macro_auc:.4f}")

    return roc_scores


# ── Misclassification panel ───────────────────────────────────────────────────

def plot_misclassified(
    errors_df: pd.DataFrame,
    n: int = 9,
    seed: int = SEED,
    save_path: Path = REPORT_DIR / "misclassified_panel.png",
) -> None:
    """Plot a random grid of misclassified MRI images."""
    if len(errors_df) == 0:
        print("No misclassified samples!")
        return

    sample = errors_df.sample(min(n, len(errors_df)),
                              random_state=seed).reset_index(drop=True)
    rows = int(np.ceil(len(sample) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    for ax, (_, row) in zip(np.array(axes).ravel(), sample.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f'True: {row["true_label"]}\n'
            f'Pred: {row["pred_label"]}\n'
            f'Conf: {row["pred_confidence"]:.3f}',
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.show()


# ── Augmentation preview ──────────────────────────────────────────────────────

def plot_augmented_samples(
    loader,
    class_names: list[str] = CLASS_NAMES,
    n_cols: int = 6,
    n_rows: int = 2,
    save_path: Path = REPORT_DIR / "sample_augmented.png",
) -> None:
    """Display a grid of augmented training samples (de-normalised)."""
    mean_t = torch.tensor(IMG_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(IMG_STD).view(3, 1, 1)

    batch_imgs, batch_labels, _ = next(iter(loader))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for ax, img, lbl in zip(axes.ravel(), batch_imgs[:n_rows * n_cols], batch_labels):
        disp = (img * std_t + mean_t).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(disp)
        ax.set_title(class_names[lbl.item()], fontsize=9)
        ax.axis("off")

    plt.suptitle("Augmented Training Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
