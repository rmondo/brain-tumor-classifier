"""
config.py
─────────
Central configuration for the Brain Tumor MRI Classifier.
All hyperparameters, path roots, and runtime flags live here so every
other module can import them without circular dependencies.

Location: notebooks/brain_tumor/config.py
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
# BASE_DIR resolves to the project root (brain-tumor-classifier/) regardless of
# whether the package sits in root/brain_tumor or notebooks/brain_tumor.
if IN_COLAB:
    BASE_DIR = Path("/content/brain_tumor_project")
else:
    _cfg_file = Path(__file__).resolve()
    _cfg_dir = _cfg_file.parent
    _project_root = next(
        (p for p in [_cfg_dir, *_cfg_dir.parents] if (p / "pyproject.toml").exists()),
        None,
    )
    BASE_DIR = _project_root if _project_root is not None else _cfg_dir.parent

DATA_DIR   = BASE_DIR / "data" / "brain_tumor_mri"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR    = BASE_DIR / "logs"
TB_LOG_DIR = BASE_DIR / "runs" / "brain_tumor"

# Expected canonical dataset split paths used across notebooks/modules.
TRAIN_ROOT = DATA_DIR / "Training"
TEST_ROOT  = DATA_DIR / "Testing"

# ── Checkpoint & model output paths ──────────────────────────────────────────
CKPT_S1    = MODEL_DIR / "best_stage1.pth"
CKPT_S2    = MODEL_DIR / "best_stage2.pth"
FINAL_PATH = MODEL_DIR / "brain_tumor_efficientnetb0_final.pth"


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
IMAGE_SIZE = IMG_SIZE   # backward-compatible alias — prefer IMG_SIZE
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
