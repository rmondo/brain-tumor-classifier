# Brain Tumor Classifier — Architecture

> **Model:** EfficientNetB0 · **Classes:** glioma · meningioma · pituitary · notumor  
> **Stack:** PyTorch · efficientnet-pytorch · Flask · scikit-learn · TensorBoard

---

## Repository Structure

```
brain-tumor-classifier/                  # pip install -e . runs here
├── pyproject.toml                       # NEW (preferred by pip >= 21.3)
├── setup.py                             # UPDATED (legacy fallback)
├── notebooks
│   ├── brain_tumor_classifier_ref.ipynb # Orchestration (refactored) notebook — imports only, no inline logic
│   ├── brain_tumor_classifier.ipynb     # VS Code version
│   ├── brain_tumor_classifier_tut.ipynb # VS Code/TensorBoard tutorial version 
│   └── brain_tumor/                     # Importable Python package (all logic lives here)
│       ├── __init__.py
│       ├── config.py                    # Single source of truth — every constant, path & flag
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py               # BrainTumorDataset · transforms · build_dataloaders · download_dataset
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── classifier.py            # BrainTumorClassifier (EfficientNetB0 + custom head)
│       │   └── checkpoint.py            # save_model · load_model · save_metrics
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py                # train_epoch · eval_epoch · run_stage (AMP + early stop)
│       │   └── tensorboard.py           # setup_writer · launch_tensorboard
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py               # compute_class_weights · get_predictions · build_error_dataframe
│           ├── plots.py                 # plot_history · plot_confusion_matrix · plot_roc_curves · plot_misclassified
│           └── gradcam.py               # GradCAM class · display_gradcam
│
├── src/
│   └── app/                             # Flask inference server
│       ├── app.py                       # Routes: GET / · POST /predict · GET /health
│       └── templates/
│           └── index.html               # Drag-and-drop upload UI + confidence bars
│
├── brain_tumor_classifier_ref.ipynb     # Orchestration (refactored) notebook — imports only, no inline logic
│
├── models/                              # Saved weights (gitignored except final)
│   ├── best_stage1.pth                  # Best checkpoint from Stage 1
│   ├── best_stage2.pth                  # Best checkpoint from Stage 2
│   └── brain_tumor_efficientnetb0_final.pth   # Final export with full metadata
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
├── runs/                                # TensorBoard event files (gitignored)
│   └── brain_tumor/
│
├── logs/                                # Flask server log (gitignored)
│   └── flask_server.log
│
├── data/                                # Kaggle dataset download (gitignored)
│   └── brain_tumor_mri/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── pituitary/
│       │   └── notumor/
│       └── Testing/
│           └── (same structure)
│
├── requirements.txt
├── setup.py                             # `pip install -e .` makes brain_tumor importable
└── README.md
```

---

## Package Responsibilities

| Module | Exports | Responsibility |
|--------|---------|----------------|
| `config.py` | `DEVICE`, `CLASS_NAMES`, `IMG_SIZE`, `BATCH_SIZE`, `DROPOUT`, `SEED`, `EPOCHS_S1/S2`, `LR_S1/S2`, `UNFREEZE_N`, `PATIENCE`, all `Path` constants, `seed_everything()`, `make_dirs()` | Every constant used across the codebase — import once, change once |
| `data/dataset.py` | `BrainTumorDataset`, `get_train_transform()`, `get_val_transform()`, `build_dataloaders()`, `download_dataset()` | Torch `Dataset`, augmentation transforms, 85/15 stratified split, Kaggle API download |
| `models/classifier.py` | `BrainTumorClassifier` | EfficientNetB0 backbone + custom head; `freeze_backbone()`, `unfreeze_last_n()`, `n_trainable` |
| `models/checkpoint.py` | `save_model()`, `load_model()`, `save_metrics()` | Checkpoint serialisation with full inference metadata; JSON metrics persistence |
| `training/engine.py` | `train_epoch()`, `eval_epoch()`, `run_stage()` | AMP-aware forward/backward pass; stage-level loop with early stopping, checkpoint saving, and TensorBoard logging |
| `training/tensorboard.py` | `setup_writer()`, `launch_tensorboard()` | `SummaryWriter` initialisation (with model graph); background subprocess launcher using the kernel-local binary |
| `evaluation/metrics.py` | `compute_class_weights()`, `get_predictions()`, `print_classification_report()`, `build_error_dataframe()` | Balanced class weights; batched inference; misclassification CSV |
| `evaluation/plots.py` | `plot_augmented_samples()`, `plot_history()`, `plot_confusion_matrix()`, `plot_roc_curves()`, `plot_misclassified()` | All matplotlib/seaborn figures; auto-saved to `reports/` |
| `evaluation/gradcam.py` | `GradCAM`, `display_gradcam()` | Forward/backward hook on `backbone._blocks[-1]`; heatmap → colour overlay |
| `src/app/app.py` | Flask app | Lazy model load; `backbone.` prefix stripping; CUDA/MPS/CPU auto-select |

---

## ML Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ① DATA  (brain_tumor/data/dataset.py)                                   │
│                                                                          │
│  Kaggle API  ──►  data/brain_tumor_mri/Training/  ──►  BrainTumorDataset │
│                       4 class sub-folders                LABEL_MAP       │
│                       (with folder-name aliases)     normalises variants │
│                                                           │              │
│                       train_test_split(test_size=0.15, stratify=labels)  │
│                              │                    │                      │
│                           train_ds             val_ds                    │
│                         (augmented)          (val_transform)             │
│                              │                    │                      │
│                    ┌─────────┴──────────-┐         │                      │
│                    │ Training Transforms │         │                     │
│                    │ Resize(224×224)     │         │                     │
│                    │ RandomHorizontalFlip│         │                       │
│                    │ RandomVerticalFlip  │         │                       │
│                    │ RandomRotation(20°) │         │                       │
│                    │ RandomResizedCrop   │         │                       │
│                    │ ColorJitter         │         │                       │
│                    │ ToTensor + Normalize│         │                       │
│                    └─────────────────────┘         │                       │
│                    DataLoader (pin_memory, AMP)    │                       │
└────────────────────────────────────────────────────┼───────────────────────┘
                                                     │
┌────────────────────────────────────────────────────▼───────────────────────┐
│  ② MODEL  (brain_tumor/models/classifier.py)                               │
│                                                                            │
│  Input [B × 3 × 224 × 224]                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌────────────────────────────────────┐                                     │
│  │  EfficientNetB0 backbone           │  ImageNet pretrained weights        │
│  │  (backbone.trainable=False, S1)    │  from efficientnet-pytorch          │
│  └─────────────────┬──────────────────┘                                     │
│                    │ feature maps                                            │
│                    ▼                                                         │
│  BatchNorm1d(1280)  ──►  Dropout(0.4)  ──►  Linear(1280→256)               │
│                    │                                                         │
│                    ▼                                                         │
│  ReLU  ──►  BatchNorm1d(256)  ──►  Dropout(0.2)  ──►  Linear(256→4)        │
│                    │                                                         │
│                    ▼                                                         │
│           logits [B × 4]  →  [glioma, meningioma, pituitary, notumor]      │
└─────────────────────────────────────────────────────────────────────────────┘
                                                     │
┌────────────────────────────────────────────────────▼───────────────────────┐
│  ③ TRAINING  (brain_tumor/training/engine.py)                              │
│                                                                             │
│  Class weights  ←  sklearn compute_class_weight('balanced')                │
│  CrossEntropyLoss(weight=class_weights)                                     │
│                                                                             │
│  ┌─ Stage 1: frozen backbone ──────────────────────────────────────────┐   │
│  │  model.freeze_backbone()                                             │   │
│  │  Optimizer : Adam(lr=1e-3)                                           │   │
│  │  Scheduler : ReduceLROnPlateau(factor=0.4, patience=4)              │   │
│  │  Epochs    : up to 15  (early stop patience=8)                      │   │
│  │  Checkpoint: models/best_stage1.pth  (best val_acc)                 │   │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼  load best_stage1.pth                          │
│                           │  model.unfreeze_last_n(30)                     │
│                           │                                                 │
│  ┌─ Stage 2: fine-tune last 30 backbone layers ────────────────────────┐   │
│  │  Optimizer : Adam(lr=5e-5)                                           │   │
│  │  Scheduler : ReduceLROnPlateau(factor=0.4, patience=4)              │   │
│  │  Epochs    : up to 25  (early stop patience=8)                      │   │
│  │  Checkpoint: models/best_stage2.pth  (best val_acc)                 │   │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  TensorBoard (per epoch, both stages):                                      │
│    Loss/{stage}      Accuracy/{stage}   LearningRate/{stage}               │
│    Loss/combined     Accuracy/combined  (continuous x-axis across stages)  │
│    Model graph logged at writer initialisation                              │
│                                                                             │
│  AMP: torch.autocast + GradScaler on CUDA; no-op on MPS / CPU             │
└─────────────────────────────────────────────────────────────────────────────┘
                                                     │
┌────────────────────────────────────────────────────▼───────────────────────┐
│  ④ EVALUATION  (brain_tumor/evaluation/)                                   │
│                                                                             │
│  Load best_stage2.pth  ──►  test_loader  ──►  get_predictions()            │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│  │ Confusion matrix │  │  ROC curves      │  │  Grad-CAM              │   │
│  │ raw counts +     │  │  per-class OvR   │  │  Hooks backbone        │   │
│  │ row-normalised   │  │  + macro AUC     │  │  ._blocks[-1]          │   │
│  │ seaborn heatmap  │  │  + fill_between  │  │  heatmap → jet overlay │   │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘   │
│                                                                             │
│  classification_report  (precision / recall / F1 / support, digits=4)     │
│  misclassified.csv      (path, true/pred label, confidence scores)         │
│  metrics_summary.json   (macro_auc, roc_per_class, total_test, n_errors)  │
│                                                                             │
│  All figures → reports/                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                                     │
┌────────────────────────────────────────────────────▼───────────────────────┐
│  ⑤ FLASK APP  (src/app/app.py)                                             │
│                                                                             │
│  Browser                                                                    │
│    │  GET  /         → templates/index.html  (click-to-upload UI)          │
│    │  POST /predict  → multipart/form-data { file: MRI image }             │
│    │  GET  /health   → JSON { status, device }                             │
│    │                                                                        │
│    ▼                                                                        │
│  Lazy model load (_load_model, cached in module globals)                   │
│    • torch.load checkpoint  →  strip "backbone." key prefix                │
│    • backbone.eval().to(DEVICE)                                             │
│    • Device: CUDA → MPS → CPU                                              │
│    │                                                                        │
│    ▼                                                                        │
│  Preprocess: Resize(img_size) → ToTensor → Normalize(mean, std)           │
│    │                                                                        │
│    ▼                                                                        │
│  torch.no_grad() forward → softmax → JSON response                        │
│    { predicted_class, confidence, all_probabilities }                      │
│    │                                                                        │
│    ▼                                                                        │
│  index.html renders probability bars inline (vanilla JS)                   │
│                                                                             │
│  Server launched via subprocess.Popen from notebook cell 17               │
│  Health-checked with urllib (polls /health up to 15 s before reporting)   │
│  Logs → logs/flask_server.log                                              │
│  Default port: 5001  (http://127.0.0.1:5001)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Notebook as Orchestration Layer

The notebook (`brain_tumor_classifier.ipynb`) contains **no function definitions**.  
Every cell is 5–15 lines of imports and calls into the `brain_tumor` package.

| Section | Package call(s) |
|---------|----------------|
| 0 · Install | `subprocess.check_call([sys.executable, '-m', 'pip', 'install', …])` |
| 1 · Config | `from brain_tumor.config import …` · `make_dirs()` · `seed_everything()` |
| 2 · Kaggle Auth | Inline credential helper (widget / Colab / existing file) |
| 3 · Dataset | `download_dataset(DATA_DIR, DATASET_SLUG)` |
| 4 · DataLoaders | `build_dataloaders(TRAIN_ROOT, TEST_ROOT)` · `plot_augmented_samples(train_loader)` |
| 5 · Class Weights | `compute_class_weights(full_train_aug, train_loader.dataset.indices)` |
| 6 · Model | `BrainTumorClassifier(NUM_CLASSES, DROPOUT).to(DEVICE)` |
| 7 · TensorBoard | `setup_writer(model)` · `launch_tensorboard()` |
| 8 · Stage 1 | `model.freeze_backbone()` · `run_stage(…, 'S1-frozen', CKPT_S1, tb_writer=writer)` |
| 9 · Stage 2 | `model.unfreeze_last_n(30)` · `run_stage(…, 'S2-finetune', CKPT_S2, step_offset=…)` |
| 10 · Curves | `plot_history(history_s1, history_s2)` |
| 11 · Evaluation | `get_predictions(best_model, test_loader)` · `print_classification_report(…)` |
| 12 · Confusion Matrix | `plot_confusion_matrix(y_true, y_pred)` |
| 13 · ROC Curves | `plot_roc_curves(y_true, y_prob)` |
| 14 · Misclassification | `build_error_dataframe(…)` · `plot_misclassified(errors_df)` |
| 15 · Grad-CAM | `display_gradcam(img_path, best_model, val_transform, …)` |
| 16 · Save | `save_model(best_model, FINAL_PATH)` · `save_metrics({…})` |
| 17 · Flask | `subprocess.Popen([sys.executable, flask_script])` + health-check loop |
| 18 · Inventory | DataFrame of all files under `models/` and `reports/` |

---

## Data Flow Summary

```
Kaggle MRI images
       │
       ▼
download_dataset()  ──  data/brain_tumor_mri/Training/
       │
       ▼
BrainTumorDataset  ──  LABEL_MAP normalisation
       │
       ▼
train_test_split(test_size=0.15, stratify=labels)
       │                           │
    train_ds                    val_ds           test_ds
  (augmented)               (val_transform)  (val_transform)
       │                                          │
       ▼                                          │
 run_stage() × 2                                  │
  Stage 1: head only                              │
  Stage 2: + last 30 backbone layers              │
       │                                          │
  best_stage2.pth  ◄────────────────────────────►│
       │                                          │
       ▼                                          ▼
  save_model()                           get_predictions()
       │                                          │
  FINAL_PATH.pth                    ┌─────────────┤
                                    │             │
                             evaluate.py     Flask /predict
                                    │
                    ┌───────────────┼───────────────────┐
                    │               │                   │
             confusion_matrix   roc_curves          gradcam
             + row-normalised   macro AUC          overlay panel
             metrics_summary.json                misclassified.csv
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `config.py` as single import | Notebook and any standalone scripts share identical constants; no argparse juggling |
| `BrainTumorDataset` returns `(img, label, path)` | Downstream evaluation (Grad-CAM, misclassification panel) needs the file path without re-querying the dataset |
| Two `Subset` views of the same training root | `full_train_aug` uses augmentation transforms; `full_train_val` uses val transforms — same indices, different pipelines for class-weight extraction vs. validation scoring |
| `num_workers=0` on macOS | PyTorch `spawn` start method on macOS cannot pickle notebook-defined classes; forced to 0 to avoid `AttributeError` on `BrainTumorDataset` |
| AMP only on CUDA | `torch.autocast` + `GradScaler` are stable on CUDA; MPS AMP support is incomplete and omitted |
| Stage offset in `run_stage` | `global_step_offset=len(history_s1['train_loss'])` keeps TensorBoard x-axis continuous across both stages |
| `shutil.which` + kernel-local binary for TensorBoard | Avoids conda/venv PATH collisions that caused `No module named tensorboard.__main__` when using `python -m tensorboard` |
| `backbone.` key prefix stripping in Flask | Training wrapper saves state dict with `backbone.*` prefixed keys; Flask loads a bare `EfficientNet` instance, requiring the prefix to be stripped |
| `FINAL_PATH` embeds full metadata | Checkpoint includes `class_names`, `image_size`, `mean`, `std`, `dropout` so Flask inference needs no external config |
| Lazy model load in Flask | 47 MB EfficientNetB0 load takes ~1 s — cached in module globals, loaded on first `/predict` request |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or make the package importable in development mode
pip install -e .

# 2. Place kaggle.json in ~/.kaggle/ (chmod 600)
#    or let the notebook widget handle it interactively

# 3. Open the orchestration notebook
jupyter notebook brain_tumor_classifier.ipynb
# Run cells 0–18 top to bottom

# 4. TensorBoard (launched automatically by cell 7, or manually)
tensorboard --logdir runs/brain_tumor --port 6006

# 5. Flask inference server (launched automatically by cell 17, or manually)
python src/app/app.py
# → http://127.0.0.1:5001
```

---

> **Disclaimer** — For research and educational purposes only.  
> Model outputs must not substitute for clinical diagnosis.
