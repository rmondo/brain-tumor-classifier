"""
training/engine.py
──────────────────
Core training and evaluation loop for the Brain Tumor MRI Classifier.

Public API
----------
train_epoch   — one pass over the training DataLoader
eval_epoch    — one pass over a validation/test DataLoader (no grad)
run_stage     — full stage: epoch loop + early stopping + TensorBoard logging
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_tumor.config import (
    AMP_DTYPE,
    DEVICE,
    PATIENCE,
    USE_AMP,
)

# Type alias for the history dict returned by run_stage
History = dict[str, list[float]]


# ── Device helpers ────────────────────────────────────────────────────────────

def _to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to *device*; use non-blocking transfer on CUDA."""
    return tensor.to(device, non_blocking=(device.type == "cuda"))


def _amp_ctx(device: torch.device):
    """Return the AMP autocast context appropriate for *device*."""
    if USE_AMP and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


# ── Epoch functions ───────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss, accuracy
    """
    model.train()
    total_loss = correct = total = 0

    for imgs, labels, _ in tqdm(loader, leave=False, desc="  train"):
        imgs   = _to(imgs, device)
        labels = _to(labels, device)

        optimizer.zero_grad(set_to_none=True)

        with _amp_ctx(device):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """
    Run one evaluation epoch (no gradient computation).

    Returns
    -------
    avg_loss, accuracy
    """
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels, _ in tqdm(loader, leave=False, desc="  eval"):
        imgs   = _to(imgs, device)
        labels = _to(labels, device)

        with _amp_ctx(device):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Stage runner ──────────────────────────────────────────────────────────────

def run_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    stage_name: str,
    best_path: Path,
    device: torch.device = DEVICE,
    tb_writer=None,
    global_step_offset: int = 0,
    patience: int = PATIENCE,
) -> History:
    """
    Train for one stage with early stopping, checkpoint saving, and optional
    TensorBoard logging.

    Parameters
    ----------
    tb_writer         : ``SummaryWriter`` or ``None``
    global_step_offset: epoch offset so Stage-2 x-axis continues Stage-1
    patience          : early-stopping patience (epochs without val-acc gain)

    Returns
    -------
    history dict with keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``
    """
    history: History = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0
    patience_ctr = 0
    #scaler = torch.cuda.amp.GradScaler( #deprecated command
    #    enabled=(USE_AMP and device.type == "cuda")
    #)
    scaler = torch.amp.GradScaler("cuda",
        enabled=(USE_AMP and device.type == "cuda")
    )
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # ── TensorBoard ───────────────────────────────────────────────────────
        if tb_writer is not None:
            g = global_step_offset + epoch
            tb_writer.add_scalars(
                f"Loss/{stage_name}", {"train": tr_loss, "val": vl_loss}, g
            )
            tb_writer.add_scalars(
                f"Accuracy/{stage_name}", {"train": tr_acc, "val": vl_acc}, g
            )
            tb_writer.add_scalars(
                "Loss/combined",
                {f"{stage_name}_train": tr_loss, f"{stage_name}_val": vl_loss}, g,
            )
            tb_writer.add_scalars(
                "Accuracy/combined",
                {f"{stage_name}_train": tr_acc, f"{stage_name}_val": vl_acc}, g,
            )
            tb_writer.add_scalar(
                f"LearningRate/{stage_name}",
                optimizer.param_groups[0]["lr"], g,
            )

        # ── Checkpoint & early stopping ───────────────────────────────────────
        tag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)
            patience_ctr = 0
            tag = "  ← best"
        else:
            patience_ctr += 1

        print(
            f"{stage_name} | Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}{tag}"
        )

        if patience_ctr >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return history
