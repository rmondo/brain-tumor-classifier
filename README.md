# 🧠 Brain Tumor MRI Classifier

A four-class brain tumor classifier built on **EfficientNetB3** transfer learning, with Grad-CAM explainability and a Flask upload interface. Trained on the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle.

> ⚠️ **Medical Disclaimer** — This project is for **research and educational purposes only**. Model outputs are not a substitute for professional clinical diagnosis. Do not use predictions to inform medical decisions. Always consult a qualified healthcare provider.

---

## Classes

| Label | Description |
|---|---|
| `glioma_tumor` | Tumors arising from glial cells |
| `meningioma_tumor` | Tumors of the meninges (brain lining) |
| `pituitary_tumor` | Tumors of the pituitary gland |
| `no_tumor` | Healthy brain scan — no tumor present |

---

## Features

- **EfficientNetB3** backbone with ImageNet pre-training
- Two-phase training: frozen backbone → selective fine-tuning
- Stratified 80/20 train/val split with class-weight balancing
- Heavy image augmentation (flip, rotate, zoom, contrast, brightness, translate)
- **Grad-CAM** heatmaps highlighting the regions driving each prediction
- Confusion matrix, per-class ROC curves, and macro-average AUC
- **TensorBoard** integration for live training monitoring
- **Flask** web app — drag-and-drop MRI upload with visual results
- Docker support for reproducible deployment

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | _fill after training_ |
| Macro-average AUC | _fill after training_ |
| Macro F1 score | _fill after training_ |

Evaluation artefacts (confusion matrix, ROC curves, Grad-CAM sample grid) are saved to `reports/` after running `src/evaluate.py`.

---

## Project Structure

```
brain-tumor-classifier/
├── data/
│   ├── kaggle_download.sh      # Dataset acquisition
│   ├── prepare_splits.py       # Stratified train/val split
│   └── split_data/             # Generated split (gitignored)
├── src/
│   ├── config.py               # All hyperparameters — edit here
│   ├── dataset.py              # tf.data pipelines + augmentation
│   ├── model.py                # EfficientNetB3 builder
│   ├── train.py                # Two-phase training loop
│   ├── evaluate.py             # Metrics, plots, reports
│   ├── gradcam.py              # Grad-CAM (GradientTape)
│   ├── predict.py              # Single-image inference
│   └── utils.py                # Shared helpers
├── app/                        # Flask web application
│   ├── routes.py
│   ├── inference.py            # Model singleton
│   ├── templates/
│   │   ├── index.html          # Upload UI
│   │   └── result.html         # Grad-CAM + confidence bars
│   └── wsgi.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── models/                     # Saved model (gitignored)
├── reports/                    # Auto-generated figures + JSON
├── tests/
├── docker/
├── .github/workflows/ci.yml
├── ARCHITECTURE.md
├── Makefile
└── requirements.txt
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full directory listings, pipeline diagrams, and design decision rationale.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Kaggle API token](https://www.kaggle.com/settings) (`kaggle.json`)
- GPU recommended (CPU training is slow for EfficientNetB3)

### 1 — Clone & install

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure credentials

```bash
cp .env.example .env
# Edit .env and set:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_api_key
#   FLASK_SECRET_KEY=any_random_string
```

### 3 — Download & prepare data

```bash
make download        # runs data/kaggle_download.sh
make split           # runs data/prepare_splits.py  (stratified 80/20)
```

Or manually:

```bash
bash data/kaggle_download.sh
python data/prepare_splits.py
```

### 4 — Train

```bash
make train
# or
python -m src.train
```

Training runs in two phases automatically:

| Phase | Backbone | Learning rate | Max epochs |
|---|---|---|---|
| 1 — frozen | All layers frozen | `1e-3` | 15 |
| 2 — fine-tune | Last 30 layers unfrozen | `5e-5` | 25 |

`EarlyStopping(patience=8)` and `ReduceLROnPlateau(patience=4)` apply to both phases. The best checkpoint is restored automatically.

### 5 — Monitor with TensorBoard

```bash
tensorboard --logdir logs/fit
# Open http://localhost:6006
```

### 6 — Evaluate

```bash
make evaluate
# or
python -m src.evaluate
```

Outputs saved to `reports/`:
- `confusion_matrix.png`
- `roc_curves.png`
- `gradcam_samples.png`
- `metrics_summary.json`

### 7 — Run the Flask app

```bash
make serve
# or
gunicorn -w 1 app.wsgi:app --bind 0.0.0.0:5000
# Open http://localhost:5000
```

Upload any brain MRI image (JPEG or PNG). The app returns the predicted class, confidence score, full probability breakdown, and a Grad-CAM heatmap overlay.

> **Note on workers:** `-w 1` is intentional. TensorFlow Keras models are not fork-safe across worker processes. For multi-worker production serving, use TensorFlow Serving or export to ONNX Runtime.

---

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
# App available at http://localhost:5000
```

---

## Configuration

All hyperparameters live in `src/config.py`. No need to touch any other file to change training settings.

```python
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
VAL_SPLIT       = 0.20
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 25
UNFREEZE_FROM   = -30        # unfreeze last N backbone layers
LR_FROZEN       = 1e-3
LR_FINETUNE     = 5e-5
DROPOUT         = 0.40
```

---

## Model Architecture

```
Input [224 × 224 × 3]
        │
        ▼
EfficientNetB3 backbone  (ImageNet weights, include_top=False)
        │  feature maps [7 × 7 × 1536]
        ▼
GlobalAveragePooling2D
        │
        ▼
BatchNormalization  →  Dropout(0.4)
        │
        ▼
Dense(512, relu)
        │
        ▼
BatchNormalization  →  Dropout(0.2)
        │
        ▼
Dense(4, softmax)
        │
        ▼
[glioma · meningioma · no_tumor · pituitary]
```

---

## Grad-CAM

Grad-CAM is computed using `tf.GradientTape` over the final convolutional layer (`top_conv`) of EfficientNetB3. The resulting heatmap is resized to the input image dimensions and blended over the original scan using OpenCV's `COLORMAP_JET`. This highlights the spatial regions that most influenced the model's prediction.

Batch Grad-CAM visualizations for validation samples are saved to `reports/gradcam_samples.png`. The Flask app generates per-image heatmaps inline on the result page.

---

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Class distribution, sample grids, pixel statistics |
| `02_training.ipynb` | Interactive training with inline loss/accuracy/AUC curves |
| `03_evaluation.ipynb` | Full evaluation suite — confusion matrix, ROC, Grad-CAM |

---

## Running Tests

```bash
pytest tests/ -v
```

`tests/test_model.py` — forward-pass shape checks and model build smoke test  
`tests/test_api.py` — Flask route tests including multipart image upload

---

## Makefile Reference

```
make download     Download and unzip Kaggle dataset
make split        Prepare stratified train/val split
make train        Run two-phase training
make evaluate     Generate reports/ artefacts
make serve        Start Flask app via gunicorn
make test         Run pytest suite
make clean        Remove split_data/, models/, reports/, logs/
```

---

## Requirements

```
tensorflow>=2.15
flask>=3.0
gunicorn
scikit-learn
opencv-python-headless
matplotlib
seaborn
kaggle
python-dotenv
pytest
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

- Dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) by Sartaj Bhuvaji et al. on Kaggle
- Backbone: [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- Explainability: [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017

---

> ⚠️ **Medical Disclaimer** — This software is provided for **research and educational purposes only**. It has not been validated for clinical use and must not be used to diagnose, treat, or inform medical decisions. Brain tumor diagnosis requires qualified radiologists and appropriate clinical evaluation. The authors accept no liability for any use of this software beyond its intended research scope.
