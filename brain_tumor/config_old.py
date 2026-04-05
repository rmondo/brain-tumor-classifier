"""
config.py
─────────
Central configuration for the Brain Tumor MRI Classifier.
All hyperparameters, path roots, and runtime flags live here so every
other module can import them without circular dependencies.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Environment detection ─────────────────────────────────────────────────────
IN_COLAB  = "google.colab" in sys.modules
ON_MACOS  = platform.system() == "Darwin"
ON_ARM64  = platform.machine() == "arm64"
HAS_CUDA  = torch.cuda.is_available()
HAS_MPS   = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# ── Device selection: CUDA → MPS → CPU ───────────────────────────────────────
if HAS_CUDA:
    DEVICE = torch.device("cuda")
elif HAS_MPS:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

USE_AMP   = HAS_CUDA
AMP_DTYPE = torch.float16

if HAS_CUDA:
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """Set all global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_CUDA:
        torch.cuda.manual_seed_all(seed)


# ── Paths ─────────────────────────────────────────────────────────────────────
if IN_COLAB:
    BASE_DIR = Path("/content/brain_tumor_project")
else:
    BASE_DIR = Path(".").resolve()

DATA_DIR   = BASE_DIR / "data" / "brain_tumor_mri"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR    = BASE_DIR / "logs"
TB_LOG_DIR = BASE_DIR / "runs" / "brain_tumor"

print(f'BASE_DIR: {BASE_DIR}')
print(f'DATA_DIR: {DATA_DIR}')

# ── Checkpoint & model output paths ──────────────────────────────────────────
CKPT_S1    = MODEL_DIR / "best_stage1.pth"       # best weights from Stage 1
CKPT_S2    = MODEL_DIR / "best_stage2.pth"       # best weights from Stage 2
FINAL_PATH = MODEL_DIR / "brain_tumor_efficientnetb0_final.pth"  # final export


def make_dirs() -> None:
    """Create all output directories if they don't already exist."""
    for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES  = len(CLASS_NAMES)
DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"

#: Folder-name aliases present in the raw Kaggle download
LABEL_MAP: dict[str, int] = {
    "glioma": 0,           "glioma_tumor": 0,
    "meningioma": 1,       "meningioma_tumor": 1,
    "pituitary": 2,        "pituitary_tumor": 2,
    "notumor": 3,          "no_tumor": 3,          "no tumor": 3,
}

# ImageNet normalisation stats (used by EfficientNet)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Model / training hyperparameters ─────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
DROPOUT    = 0.40

EPOCHS_S1  = 15
LR_S1      = 1e-3
EPOCHS_S2  = 25
LR_S2      = 5e-5

#: Number of backbone params to unfreeze in Stage 2
UNFREEZE_N = 30

#: Early-stopping patience (epochs without val-acc improvement)
PATIENCE   = 8

# ── DataLoader tuning ─────────────────────────────────────────────────────────
# macOS/Jupyter: num_workers > 0 causes spawn/pickle errors with inline classes
NUM_WORKERS = (
    2 if IN_COLAB
    else (0 if ON_MACOS else min(4, os.cpu_count() or 2))
)
PIN_MEMORY         = HAS_CUDA
PERSISTENT_WORKERS = NUM_WORKERS > 0 and not IN_COLAB and not ON_MACOS

# ── Flask server ──────────────────────────────────────────────────────────────
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5001
