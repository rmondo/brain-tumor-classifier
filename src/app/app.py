"""
src/app/app.py
──────────────
Flask inference server for the Brain Tumor MRI Classifier.

Endpoints
---------
GET  /          → HTML upload UI
POST /predict   → multipart/form-data { file: <image> } → JSON prediction
GET  /health    → JSON { status, device }

Run
---
    python src/app/app.py
    # → http://127.0.0.1:5001
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
from werkzeug.exceptions import HTTPException

# Allow `from brain_tumor import …` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brain_tumor.config import FLASK_HOST, FLASK_PORT  # noqa: E402

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
CORS(app)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Path is resolved relative to this file so it works regardless of cwd
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "brain_tumor_efficientnetb0_final.pth"
MODEL_PATH = Path(torch.hub.get_dir() if False else _DEFAULT_MODEL)  # override via env if needed
MODEL_PATH = _DEFAULT_MODEL

_model: nn.Module | None = None
_meta:  dict | None      = None

print(f"[INIT] Device      : {DEVICE}", flush=True)
print(f"[INIT] Model path  : {MODEL_PATH}", flush=True)
print(f"[INIT] Model exists: {MODEL_PATH.exists()}", flush=True)


# ── Error handler ─────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def _handle_error(error):
    if isinstance(error, HTTPException):
        return jsonify({
            "error": error.description,
            "type": type(error).__name__,
        }), error.code

    traceback.print_exc()
    return jsonify({
        "error"    : str(error),
        "type"     : type(error).__name__,
        "traceback": traceback.format_exc(),
    }), 500


# ── Model loading (lazy, cached) ──────────────────────────────────────────────

def _load_model() -> tuple[nn.Module, dict]:
    global _model, _meta
    if _model is not None:
        return _model, _meta

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    from efficientnet_pytorch import EfficientNet

    print(f"[MODEL] Loading from {MODEL_PATH} …", flush=True)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    _meta = ckpt

    n_cls, drop = ckpt["num_classes"], ckpt["dropout"]
    backbone    = EfficientNet.from_pretrained("efficientnet-b0", num_classes=n_cls)
    in_f        = backbone._fc.in_features
    backbone._fc = nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(drop),
        nn.Linear(in_f, 256), nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),  nn.Dropout(drop / 2),
        nn.Linear(256, n_cls),
    )

    # Strip the 'backbone.' prefix that the training wrapper adds
    state = {
        (k[len("backbone."):] if k.startswith("backbone.") else k): v
        for k, v in ckpt["model_state_dict"].items()
    }
    backbone.load_state_dict(state)
    backbone.eval().to(DEVICE)
    _model = backbone
    print(f"[MODEL] Loaded successfully on {DEVICE}", flush=True)
    return _model, _meta


def _preprocess(img: Image.Image, meta: dict) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((meta["image_size"], meta["image_size"])),
        T.ToTensor(),
        T.Normalize(meta["mean"], meta["std"]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "message": "Use POST /predict with multipart/form-data field 'file' containing an MRI image.",
            "example": "curl -X POST http://localhost:5050/predict -F \"file=@path/to/image.jpg\"",
        }), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send 'file' in multipart/form-data"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file received"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as exc:
        return jsonify({"error": f"Failed to read image: {exc}"}), 400

    try:
        model, meta = _load_model()
        x           = _preprocess(img, meta)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Model loading/preprocessing failed: {exc}"}), 500

    try:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        return jsonify({
            "predicted_class"  : meta["class_names"][pred_idx],
            "confidence"       : float(probs[pred_idx]),
            "all_probabilities": {
                cls: float(p)
                for cls, p in zip(meta["class_names"], probs)
            },
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {exc}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[MAIN] Starting Flask on http://{FLASK_HOST}:{FLASK_PORT}", flush=True)
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
