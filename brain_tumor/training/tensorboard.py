"""
training/tensorboard.py
───────────────────────
TensorBoard writer setup and local-server launcher.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from brain_tumor.config import DEVICE, IMAGE_SIZE, TB_LOG_DIR


def _find_free_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
    """Return the preferred port if available, otherwise the next free one."""
    for offset in range(max_tries + 1):
        candidate = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return preferred_port


def setup_writer(
    model=None,
    log_dir: Path = TB_LOG_DIR,
    reset: bool = True,
):
    """
    Create a fresh ``SummaryWriter`` and optionally log the model graph.

    Parameters
    ----------
    model   : ``nn.Module`` or ``None`` — if supplied, a dummy forward pass
              is used to log the graph; skipped gracefully on failure.
    log_dir : TensorBoard event log directory (wiped when *reset* is True).
    reset   : Remove existing logs so the new run starts clean.

    Returns
    -------
    writer : ``torch.utils.tensorboard.SummaryWriter``
    """
    from torch.utils.tensorboard import SummaryWriter
    import torch

    if reset and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard writer → {log_dir}")

    if model is not None:
        try:
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
            writer.add_graph(model, dummy)
            print("Model graph logged to TensorBoard.")
        except Exception as exc:
            print(f"Graph logging skipped: {exc}")

    return writer


def launch_tensorboard(
    log_dir: Path = TB_LOG_DIR,
    port: int = 6006,
    host: str = "127.0.0.1",
) -> subprocess.Popen | None:
    """
    Launch a TensorBoard server in a background subprocess.

    Uses the ``tensorboard`` executable co-located with the current Python
    interpreter (avoids PATH picking a wrong conda environment binary).
    Falls back to ``shutil.which`` if the sibling binary doesn't exist.

    Returns the ``Popen`` handle so callers can call ``proc.terminate()``,
    or ``None`` if the binary couldn't be found.

    Usage
    -----
    >>> tb_proc = launch_tensorboard()
    >>> # … do work …
    >>> tb_proc.terminate()
    """
    # Prefer the binary from the kernel's own env directory
    kernel_bin = Path(sys.executable).with_name("tensorboard")
    if kernel_bin.exists():
        tb_bin = str(kernel_bin)
    else:
        tb_bin = shutil.which("tensorboard")

    if tb_bin is None:
        print("tensorboard binary not found for this kernel environment.")
        print(f"Install with:  {sys.executable} -m pip install tensorboard")
        return None

    print(f"tensorboard binary : {tb_bin}")
    print(f"Log dir            : {log_dir}")
    print(f"Exists             : {log_dir.exists()}")
    if log_dir.exists():
        files = list(log_dir.rglob("*"))
        print(f"Files              : {[str(f.relative_to(log_dir)) for f in files] or '(empty)'}")

    chosen_port = _find_free_port(host, port)
    if chosen_port != port:
        print(f"Port {port} busy, using {chosen_port} instead.")

    proc = subprocess.Popen(
        [tb_bin, "--logdir", str(log_dir), "--port", str(chosen_port), "--host", host],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(3)

    if proc.poll() is None:
        print(f"✅ TensorBoard running → http://{host}:{chosen_port}")
        print("   (allow a few seconds, then refresh if graphs are empty)")
        print("   Stop with: tb_proc.terminate()")
    else:
        out = proc.stdout.read() if proc.stdout else "(no output)"
        print(f"❌ TensorBoard exited with code {proc.poll()}")
        print("Output:", out[:1200])
        print("\nManual fallback — run in your terminal:")
        print(f'  {sys.executable} -m tensorboard.main --logdir "{log_dir}" --port {port} --host {host}')

    return proc
