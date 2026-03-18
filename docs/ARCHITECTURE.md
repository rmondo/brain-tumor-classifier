# Brain Tumor Classifier — Architecture

> **Model:** EfficientNetB3 · **Classes:** glioma · meningioma · pituitary · no\_tumor  
> **Stack:** TensorFlow / Keras · Flask · scikit-learn · TensorBoard

---

## Repository Structure

```
brain-tumor-classifier/
│
├── data/                            # Data acquisition & preparation
│   ├── raw/                         # Original Kaggle download (gitignored)
│   ├── split_data/                  # Stratified train / val split
│   │   ├── train/
│   │   │   ├── glioma_tumor/
│   │   │   ├── meningioma_tumor/
│   │   │   ├── no_tumor/
│   │   │   └── pituitary_tumor/
│   │   └── val/
│   │       └── (same structure)
│   ├── kaggle_download.sh           # Pulls dataset via Kaggle API
│   ├── prepare_splits.py            # Stratified split with sklearn
│   └── class_distribution.png      # EDA bar chart (auto-generated)
│
├── src/                             # Core ML library
│   ├── config.py                    # Single source of truth — all hyperparams
│   ├── dataset.py                   # tf.data pipelines + augmentation layer
│   ├── model.py                     # EfficientNetB3 builder + unfreeze helper
│   ├── train.py                     # Two-phase training orchestrator
│   ├── evaluate.py                  # Confusion matrix, ROC curves, metrics JSON
│   ├── gradcam.py                   # Grad-CAM implementation (GradientTape)
│   ├── predict.py                   # Single-image inference + heatmap export
│   └── utils.py                     # Shared helpers (seeding, plotting, logging)
│
├── app/                             # Flask web application
│   ├── __init__.py                  # App factory
│   ├── routes.py                    # GET / and POST /predict endpoints
│   ├── inference.py                 # Model singleton (loaded once at startup)
│   ├── wsgi.py                      # Gunicorn entry point
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js                # Drag-and-drop upload + Chart.js bars
│   ├── templates/
│   │   ├── index.html               # Upload UI
│   │   └── result.html              # MRI + Grad-CAM overlay + probability bars
│   └── uploads/                     # Temp storage for uploaded MRIs (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Class distribution, sample grid, statistics
│   ├── 02_training.ipynb            # Interactive training run with inline plots
│   └── 03_evaluation.ipynb          # Full evaluation suite
│
├── models/
│   ├── brain_tumor_model.keras      # Final saved model
│   └── checkpoints/                 # Per-epoch best weights (gitignored)
│
├── reports/                         # Auto-generated evaluation artefacts
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── gradcam_samples.png
│   └── metrics_summary.json
│
├── tests/
│   ├── test_model.py                # Forward-pass shape checks, build smoke test
│   └── test_api.py                  # Flask route tests (upload, JSON response)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/
│   ├── workflows/ci.yml             # Lint → test → build on every PR
│   └── PULL_REQUEST_TEMPLATE.md
│
├── logs/                            # TensorBoard event files (gitignored)
│   └── fit/<run-timestamp>/
│
├── .env.example                     # KAGGLE_USERNAME, KAGGLE_KEY, FLASK_SECRET
├── .gitignore
├── requirements.txt
├── Makefile                         # make download · make train · make serve
└── README.md
```

---

## ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ① DATA                                                                  │
│                                                                          │
│  Kaggle API  ──►  raw/Training/  ──►  prepare_splits.py                 │
│                       4 classes         stratify=labels, 80/20           │
│                                         ▼                                │
│                                    split_data/{train,val}/<class>/       │
│                                         │                                │
│                                    dataset.py  (tf.data)                 │
│                                    ┌────┴──────────────────────────┐     │
│                                    │  Augmentation (train only)    │     │
│                                    │  RandomFlip · RandomRotation  │     │
│                                    │  RandomZoom · RandomContrast  │     │
│                                    │  RandomBrightness · Translate │     │
│                                    └────────────────┬──────────────┘     │
│                                                     │ .prefetch(AUTOTUNE)│
└─────────────────────────────────────────────────────┼───────────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼───────────────────┐
│  ② MODEL  (model.py)                                                     │
│                                                                          │
│  Input [224×224×3]                                                       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────┐                                     │
│  │  EfficientNetB3 backbone        │  ImageNet weights                   │
│  │  (frozen in phase 1)            │  include_top=False                  │
│  └──────────────┬──────────────────┘                                     │
│                 │ feature maps [7×7×1536]                                │
│                 ▼                                                        │
│  GlobalAveragePooling2D  ──►  BatchNorm  ──►  Dropout(0.4)              │
│                 │                                                        │
│                 ▼                                                        │
│           Dense(512, relu)  ──►  BatchNorm  ──►  Dropout(0.2)           │
│                 │                                                        │
│                 ▼                                                        │
│           Dense(4, softmax)  →  [glioma, meningioma, no_tumor, pituitary]│
└─────────────────────────────────────────────────────────────────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼───────────────────┐
│  ③ TRAINING  (train.py)                                                  │
│                                                                          │
│  Class weights  ←  sklearn compute_class_weight('balanced')              │
│                                                                          │
│  ┌─ Phase 1: frozen backbone ──────────────────────────────────────┐     │
│  │  Optimizer : Adam(lr=1e-3)                                       │     │
│  │  Epochs    : up to 15                                            │     │
│  │  Loss      : categorical_crossentropy + class_weight            │     │
│  └──────────────────────────────────────────────────────────────────┘    │
│                         │                                                │
│                         ▼  (unfreeze last 30 backbone layers)            │
│  ┌─ Phase 2: fine-tune ────────────────────────────────────────────┐     │
│  │  Optimizer : Adam(lr=5e-5)                                       │     │
│  │  Epochs    : up to 25 more                                       │     │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Callbacks (both phases):                                                │
│    • EarlyStopping(patience=8, restore_best_weights=True)                │
│    • ReduceLROnPlateau(factor=0.4, patience=4)                           │
│    • ModelCheckpoint(monitor=val_accuracy, save_best_only=True)          │
│    • TensorBoard(log_dir=logs/fit/<timestamp>, histogram_freq=1)         │
│                                                                          │
│  Output → models/brain_tumor_model.keras                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼───────────────────┐
│  ④ EVALUATION  (evaluate.py · gradcam.py)                                │
│                                                                          │
│  Load .keras  ──►  val_ds  ──►  collect y_true, y_prob                  │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ Confusion matrix │  │ ROC curves       │  │ Grad-CAM              │  │
│  │ raw + normalised │  │ per-class OvR    │  │ GradientTape on       │  │
│  │ seaborn heatmap  │  │ macro-avg AUC    │  │ "top_conv" layer      │  │
│  └──────────────────┘  └──────────────────┘  │ heatmap blended       │  │
│                                               │ over input image      │  │
│  classification_report (precision/recall/F1) │                       │  │
│  metrics_summary.json  (accuracy, F1, AUC)   └───────────────────────┘  │
│                                                                          │
│  All figures → reports/                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼───────────────────┐
│  ⑤ FLASK APP  (app/)                                                     │
│                                                                          │
│  Browser                                                                 │
│    │  GET /            → index.html  (drag-and-drop upload)              │
│    │  POST /predict    → multipart/form-data (MRI image)                 │
│    │                                                                     │
│    ▼                                                                     │
│  app/inference.py                                                        │
│    • Model singleton loaded once at startup (lru_cache)                  │
│    • Preprocesses image → runs model → Grad-CAM heatmap                 │
│    • Returns JSON { class, confidence, probabilities, heatmap_b64 }      │
│    │                                                                     │
│    ▼                                                                     │
│  result.html                                                             │
│    • Side-by-side: original MRI  |  Grad-CAM overlay                    │
│    • Horizontal confidence bars  (Chart.js)                              │
│    • Full softmax probability table for all 4 classes                   │
│                                                                          │
│  Deployment: gunicorn -w 1 app.wsgi:app  (single worker — TF not fork   │
│  safe; use TF Serving or ONNX Runtime for multi-worker production)       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Kaggle MRI images
       │
       ▼
prepare_splits.py  ──  stratified 80/20 split
       │
       ├──► train/   ──►  augmentation  ──►  EfficientNetB3  ──►  softmax(4)
       │                                            │
       └──► val/     ──────────────────────────────►│
                                                    │
                                              class_weight
                                            + two-phase LR
                                                    │
                                            ┌───────▼────────┐
                                            │ .keras model   │
                                            └───────┬────────┘
                                                    │
                           ┌────────────────────────┤
                           │                        │
                    evaluate.py               Flask /predict
                           │                        │
                  confusion matrix           Grad-CAM heatmap
                  ROC curves                 confidence bars
                  metrics JSON               result.html
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `config.py` as single import | Notebook and script usage stay identical; no argparse juggling |
| Hard-copy splits (not symlinks) | Compatible with Windows, Docker volumes, and Colab |
| Grad-CAM targets `"top_conv"` by name | Survives layer count changes during fine-tuning |
| Flask single worker (`-w 1`) | `tf.keras` models are not fork-safe across workers |
| Model loaded once at startup | 48 MB EfficientNetB3 load takes ~1.5 s — must not run per-request |
| `class_weight='balanced'` | Meningioma is underrepresented (~30% fewer samples) |
| `initial_epoch` in phase 2 | TensorBoard and callbacks see a single continuous training run |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (set KAGGLE_USERNAME + KAGGLE_KEY in .env)
make download

# 3. Prepare stratified splits
python data/prepare_splits.py

# 4. Train (both phases)
python -m src.train

# 5. Evaluate + generate reports
python -m src.evaluate

# 6. Launch Flask app
make serve                         # → http://localhost:5000

# TensorBoard (separate terminal)
tensorboard --logdir logs/fit
```

---

> **Disclaimer** — For research and educational purposes only.  
> Model outputs must not substitute for clinical diagnosis.
