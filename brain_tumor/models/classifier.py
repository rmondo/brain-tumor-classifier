"""
models/classifier.py
────────────────────
EfficientNetB0-based classifier with a custom head and two-phase
transfer-learning helpers (freeze / partial unfreeze).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from brain_tumor.config import DROPOUT, NUM_CLASSES


class BrainTumorClassifier(nn.Module):
    """
    EfficientNetB0 backbone with a custom classification head.

    Head architecture
    -----------------
    BN(1280) → Dropout(p) → Linear(1280→256) → ReLU
             → BN(256)   → Dropout(p/2)      → Linear(256→num_classes)

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 4).
    dropout : float
        Dropout probability on the first head layer; the second layer uses
        ``dropout / 2`` (default: 0.4).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            "efficientnet-b0", num_classes=num_classes
        )
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ── Transfer-learning helpers ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the classification head."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "_fc" in name

    def unfreeze_last_n(self, n: int = 30) -> None:
        """
        Unfreeze the last *n* backbone parameters (plus the head).
        Called at the start of Stage 2 fine-tuning.
        """
        all_params = list(self.backbone.named_parameters())
        for _, param in all_params[:-n]:
            param.requires_grad = False
        for _, param in all_params[-n:]:
            param.requires_grad = True
        # Always keep head trainable
        for param in self.backbone._fc.parameters():
            param.requires_grad = True

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_trainable(self) -> int:
        """Number of trainable (gradient-enabled) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_total(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())
