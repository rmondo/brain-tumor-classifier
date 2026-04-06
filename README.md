# 🧠 Brain Tumor MRI Classifier

A four-class brain tumor classifier built on **EfficientNetB0** transfer learning, with Grad-CAM explainability and a Flask upload interface. Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

> ⚠️ **Medical Disclaimer** — This project is for **research and educational purposes only**. Model outputs are not a substitute for professional clinical diagnosis. Do not use predictions to inform medical decisions. Always consult a qualified healthcare provider.

---

## Classes

| Label | Description |
|---|---|
| `glioma` | Tumors arising from glial cells |
| `meningioma` | Tumors of the meninges (brain lining) |
| `pituitary` | Tumors of the pituitary gland |
| `notumor` | Healthy brain scan — no tumor present |

---

## Features

- **EfficientNetB0** backbone (ImageNet pretrained, via `efficientnet-pytorch`)
- Two-stage transfer learning: frozen backbone → selective fine-tune of last 30 layers
- Stratified 85/15 train/val split with balanced class-weight loss
- Image augmentation: horizontal/vertical flip, ±20° rotation, random crop, colour jitter
- **Grad-CAM** heatmaps hooking `backbone._blocks[-1]` to highlight prediction-driving regions
- Misclassification review panel with per-sample confidence scores
- Confusion matrix (counts + row-normalised), per-class ROC curves, macro-average AUC
- **TensorBoard** — loss, accuracy, and LR logged per stage with a continuous x-axis
- **Flask** web app — drag-and-drop MRI upload, JSON response, inline confidence bars
- Notebook designed as a **pure orchestration layer** — all logic in the `brain_tumor/` package
- Supports Google Colab (CUDA), Apple Silicon (MPS), and CPU fallback

> **Kaggle credentials:** place `kaggle.json` at `~/.kaggle/kaggle.json` (chmod 600),  
> or use the notebook's interactive upload widget.  
> Format: `{"username": "<username>", "key": "KGAT_<key>"}`

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | _fill after training_ |
| Macro-average AUC | _fill after training_ |
| Macro F1 score | _fill after training_ |

Evaluation artefacts are written to `reports/` automatically when the notebook is run end-to-end.

---

## Project Structure

```
brain-tumor-classifier/                 ← pip install -e . runs here
├── pyproject.toml                      ← NEW (preferred by pip >= 21.3)
├── setup.py                            ← UPDATED (legacy fallback)
├── brain_tumor/                         # Importable package — all logic lives here
│   ├── __init__.py
│   ├── config.py                        # Every constant, path & flag
│   ├── data/
│   │   └── dataset.py                   # BrainTumorDataset · transforms · build_dataloaders · download_dataset
│   ├── models/
│   │   ├── classifier.py                # BrainTumorClassifier (EfficientNetB0 + custom head)
│   │   └── checkpoint.py                # save_model · load_model · save_metrics
│   ├── training/
│   │   ├── engine.py                    # train_epoch · eval_epoch · run_stage (AMP + early stop)
│   │   └── tensorboard.py               # setup_writer · launch_tensorboard
│   └── evaluation/
│       ├── metrics.py                   # compute_class_weights · get_predictions · build_error_dataframe
│       ├── plots.py                     # All matplotlib / seaborn visualisations
│       └── gradcam.py                   # GradCAM class · display_gradcam
│
├── src/
│   └── app/                             # Flask inference server
│       ├── app.py                       # GET / · POST /predict · GET /health
│       └── templates/
│           └── index.html               # Drag-and-drop upload UI + confidence bars
│
├── brain_tumor_classifier_ref_2.ipynb   # Orchestration notebook — imports only, no inline logic
│
├── models/                              # Saved weights (gitignored except final)
│   ├── best_stage1.pth
│   ├── best_stage2.pth
│   └── brain_tumor_efficientnetb0_final.pth
│
├── reports/                             # Auto-generated evaluation artefacts
│   ├── sample_augmented.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── misclassified.csv
│   ├── misclassified_panel.png
│   └── metrics_summary.json
│
├── runs/brain_tumor/                    # TensorBoard event files (gitignored)
├── logs/flask_server.log                # Flask server log (gitignored)
├── data/brain_tumor_mri/                # Kaggle dataset (gitignored)
│
├── setup.py                             # pip install -e . makes brain_tumor importable
├── run.py                               # Entry point for Flask application — bridge between model & Flask web service 
├── requirements.txt
├── fix_install.sh                       # purge project artifacts for clean restart bash script
├── tb.sh                                # tensorBoard execution launch bash script
└── README.md
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full pipeline diagrams, package responsibility tables, and design decision rationale.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Kaggle API token](https://www.kaggle.com/settings) (`kaggle.json`)
- GPU recommended — CUDA or Apple Silicon MPS (CPU fallback available)

### 1 — Clone & install

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
###python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m venv .venv && source .venv/bin/activate  # macOS: .venv/bin/activate
###execute next commands for incompatible tensorflow
###conda create -n tf_m2 python=3.10
###conda activate tf_m2
###pip install tensorflow-macos
###pip install tensorflow-metal
###pip install jupyter matplotlib scikit-learn

pip install -r requirements.txt
# Make the brain_tumor package importable:
pip install -e .                  # core deps, editable

### execute next three lines for misplaced setup.py
##pip install -e ".[dev]"           # + pytest + pytest-cov. 
##pip install -e ".[notebook]"      # + jupyterlab, kaggle, ipywidgets.
##pip install -e ".[dev,notebook]"  # full dev environment.
pip install -e ".[dev]"
pip install -e ".[notebook]"
pip install -e ".[dev,notebook]"
```

> **PyTorch + CUDA:** the default `requirements.txt` installs the CPU/MPS build.  
> For CUDA replace the torch line with the platform wheel from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2 — Kaggle credentials

```bash
# Option A: place your token file directly
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Option B: let the notebook handle it
# The notebook detects missing credentials and shows a file-upload widget.
```

### 3 — Run the notebook

```bash
jupyter notebook brain_tumor_classifier_ref.ipynb
```

Run cells **0 through 18** top-to-bottom. The notebook orchestrates the full pipeline:

| Cell range | What happens |
|---|---|
| 0 | Dependencies installed |
| 1 | Config loaded, directories created, seeds set |
| 2 | Kaggle credentials authenticated |
| 3 | Dataset downloaded and inspected |
| 4 | DataLoaders built, augmented sample preview |
| 5 | Balanced class weights computed |
| 6 | EfficientNetB0 model instantiated |
| 7 | TensorBoard writer created, server launched → http://localhost:6006 |
| 8 | **Stage 1** training (frozen backbone) |
| 9 | **Stage 2** fine-tuning (last 30 layers unfrozen) |
| 10 | Training curves plotted |
| 11–14 | Evaluation: classification report, confusion matrix, ROC curves, misclassification panel |
| 15 | Grad-CAM visualisations |
| 16 | Model + metrics saved |
| 17 | Flask server started → http://127.0.0.1:5001 |
| 18 | Output artefact inventory |

### 4 — TensorBoard

TensorBoard is launched automatically by cell 7. To start it manually:

```bash
tensorboard --logdir runs/brain_tumor --port 6006
# Open http://localhost:6006
```

### 5 — Flask inference server

The Flask server is started automatically by cell 17. To start it manually:

```bash
python src/app/app.py
# → http://127.0.0.1:5001
```

Upload any brain MRI image (JPEG or PNG). The server returns the predicted class, confidence score, and a full softmax probability breakdown for all four classes.

> **Note on opening `index.html` directly:** the upload UI must be served through Flask — opening it as a local file (`file://...`) will produce a network error because the `/predict` endpoint has no server behind it. See the comment at the top of `src/app/templates/index.html` for how to configure a direct-file fallback URL.

---

## Training Schedule

| Stage | Backbone | Optimizer | LR | Max epochs | Early stop |
|---|---|---|---|---|---|
| 1 — frozen | All layers frozen | Adam | `1e-3` | 15 | patience = 8 |
| 2 — fine-tune | Last 30 layers unfrozen | Adam | `5e-5` | 25 | patience = 8 |

`ReduceLROnPlateau(factor=0.4, patience=4)` applies to both stages. The best checkpoint (by `val_acc`) is saved at the end of each stage and reloaded before Stage 2 begins.

---

## Model Architecture

```
Input [B × 3 × 224 × 224]
        │
        ▼
EfficientNetB0 backbone  (ImageNet weights, efficientnet-pytorch)
        │  frozen in Stage 1 · last 30 layers unfrozen in Stage 2
        ▼
BatchNorm1d(1280)  →  Dropout(0.4)  →  Linear(1280 → 256)
        │
        ▼
ReLU  →  BatchNorm1d(256)  →  Dropout(0.2)  →  Linear(256 → 4)
        │
        ▼
logits [B × 4]  →  softmax
        │
        ▼
[glioma · meningioma · pituitary · notumor]
```

---

## Grad-CAM

Grad-CAM is computed by registering forward and backward hooks on `model.backbone._blocks[-1]` — the last convolutional block of EfficientNetB0. Gradients are spatially pooled to weight the activation maps, producing a heatmap that is resized to 224 × 224 and blended over the original scan using OpenCV's `COLORMAP_JET` palette.

The notebook generates Grad-CAM overlays for both a correctly classified and a misclassified example after evaluation.

---

## Configuration

All hyperparameters and paths are centralised in `brain_tumor/config.py`. No other file needs to change when adjusting training settings.

```python
# brain_tumor/config.py (key values)
IMG_SIZE    = 224
BATCH_SIZE  = 32
DROPOUT     = 0.40
UNFREEZE_N  = 30        # backbone layers unfrozen in Stage 2
PATIENCE    = 8         # early-stopping patience

EPOCHS_S1   = 15        # Stage 1 max epochs
LR_S1       = 1e-3
EPOCHS_S2   = 25        # Stage 2 max epochs
LR_S2       = 5e-5
```

---

## Running Tests

```bash
pytest tests/ -v
```

`tests/test_model.py` — forward-pass shape checks, freeze/unfreeze, checkpoint round-trip  
`tests/test_engine.py` — train/eval epoch behaviour, run_stage early stopping, TensorBoard mock  
`tests/test_dataset.py` — transform shapes, dataset labels, DataLoader construction  
`tests/test_evaluation.py` — inference shapes, error DataFrame, plot smoke tests  
`tests/test_checkpoint.py` — save/load round-trip, metadata keys, metrics JSON

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
efficientnet-pytorch>=0.7.1
tensorboard>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
ipywidgets>=8.0.0
jupyterlab>=4.0.0
kaggle>=1.6.0
pyyaml>=6.0
flask>=3.0.0
flask-cors>=4.0.0
gunicorn>=21.2.0
```

Full pinned versions in `requirements.txt`.

---

## Contributing

1. Fork the repo and create a feature branch
2. Run `pytest tests/ -v` before opening a pull request
3. Use the pull request template in `.github/PULL_REQUEST_TEMPLATE.md`
4. CI runs lint (flake8) → tests → Docker build on every PR

---

## License

MIT — see `LICENSE` for details.

---

## Acknowledgements

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar on Kaggle
- Backbone: [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- Explainability: [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017

---

> ⚠️ **Medical Disclaimer** — This software is provided for **research and educational purposes only**. It has not been validated for clinical use and must not be used to diagnose, treat, or inform medical decisions. Brain tumor diagnosis requires qualified radiologists and appropriate clinical evaluation. The authors accept no liability for any use of this software beyond its intended research scope.
