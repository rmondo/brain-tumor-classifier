"""
evaluation/gradcam.py
─────────────────────
Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet.
"""

from __future__ import annotations

import cv2
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from brain_tumor.config import CLASS_NAMES, DEVICE, IMAGE_SIZE


class GradCAM:
    """
    Grad-CAM implementation that hooks a single *target_layer*.

    Usage
    -----
    >>> gcam = GradCAM(model, model.backbone._blocks[-1])
    >>> with torch.enable_grad():
    ...     heatmap, pred_idx = gcam.generate(img_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, "_activations", out.detach())
        )
        self.target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, "_gradients", grad_out[0].detach())
        )

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: int | None = None,
        device: torch.device = DEVICE,
    ) -> tuple[np.ndarray, int]:
        """
        Compute a Grad-CAM heatmap for *img_tensor*.

        Parameters
        ----------
        img_tensor : unbatched (C, H, W) tensor
        class_idx  : target class index; defaults to the argmax prediction

        Returns
        -------
        heatmap  : float32 array in [0, 1], shape (H, W)
        pred_idx : the class index used for backprop
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)

        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Weight activations by spatially-pooled gradients
        pooled = self._gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (self._activations * pooled).sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap).cpu().numpy()

        lo, hi  = heatmap.min(), heatmap.max()
        heatmap = (heatmap - lo) / (hi - lo + 1e-8)
        return heatmap, class_idx


def display_gradcam(
    img_path: str,
    model: nn.Module,
    transform,
    alpha: float = 0.4,
    title_prefix: str = "",
    image_size: int = IMAGE_SIZE,
    class_names: list[str] = CLASS_NAMES,
    device: torch.device = DEVICE,
) -> tuple[str, float]:
    """
    Generate and display a Grad-CAM overlay for a single MRI image.

    Parameters
    ----------
    img_path     : path to the original MRI file
    model        : trained ``BrainTumorClassifier``
    transform    : val/test transform (no augmentation)
    alpha        : heatmap overlay strength (0 = no overlay, 1 = full heatmap)
    title_prefix : optional ``suptitle`` text

    Returns
    -------
    pred_label  : predicted class name
    confidence  : predicted class confidence (%)
    """
    original = Image.open(img_path).convert("RGB")
    img_t    = transform(original)

    target_layer = model.backbone._blocks[-1]
    gcam = GradCAM(model, target_layer)

    with torch.enable_grad():
        heatmap, pred_idx = gcam.generate(img_t, device=device)

    orig_arr   = np.array(original.resize((image_size, image_size)))
    hm_resized = cv2.resize(heatmap, (image_size, image_size))
    hm_color   = mpl_cm.jet(hm_resized)[:, :, :3]
    overlay    = (hm_color * alpha * 255 + orig_arr * (1 - alpha)).astype(np.uint8)

    with torch.no_grad():
        logits     = model(img_t.unsqueeze(0).to(device))
        confidence = torch.softmax(logits, dim=1)[0, pred_idx].item() * 100

    pred_label = class_names[pred_idx]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, im, title in zip(
        axes,
        [orig_arr, hm_resized, overlay],
        [
            "Original MRI",
            "Grad-CAM Heatmap",
            f"Overlay  Pred: {pred_label} ({confidence:.1f}%)",
        ],
    ):
        ax.imshow(im, cmap="jet" if "Heatmap" in title else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return pred_label, confidence
