#!/usr/bin/env bash
# fix_install.sh
# ──────────────
# Purges every stale artefact from previous `pip install -e .` runs
# and reinstalls the brain_tumor package cleanly from the project root.
#
# Run from the project root:
#   bash fix_install.sh
#
# What "(unknown location)" means
# ────────────────────────────────
# Python found brain_tumor via a stale .egg-link or direct_url.json that
# points at an old path instead of the current project. Importing any name
# from that ghost entry fails with "ImportError … (unknown location)".
# The fix is to uninstall every trace of the old install and reinstall
# from the correct directory.

set -euo pipefail

echo "── Step 1: uninstall all existing brain-tumor-classifier installs ──"
pip uninstall brain-tumor-classifier -y 2>/dev/null && echo "  Uninstalled." || echo "  Not installed — skipping."

echo ""
echo "── Step 2: delete stale build / cache artefacts ──"

# .egg-info directories (created by setup.py develop / pip install -e .)
find . -maxdepth 3 -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null && echo "  Removed *.egg-info" || true

# __pycache__ dirs (may hold .pyc files compiled from old source paths)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null && echo "  Removed __pycache__" || true

# Compiled bytecode
find . -name "*.pyc" -delete 2>/dev/null && echo "  Removed *.pyc" || true

echo ""
echo "── Step 3: verify setup.py / pyproject.toml exist at project root ──"
if [[ ! -f "setup.py" && ! -f "pyproject.toml" ]]; then
    echo "  ERROR: neither setup.py nor pyproject.toml found in $(pwd)"
    echo "  Make sure you are running this script from the project root:"
    echo "    cd /path/to/brain-tumor-classifier"
    echo "    bash fix_install.sh"
    exit 1
fi
echo "  Found: $(ls setup.py pyproject.toml 2>/dev/null | tr '\n' ' ')"

echo ""
echo "── Step 4: verify brain_tumor/__init__.py exists ──"
if [[ ! -f "brain_tumor/__init__.py" ]]; then
    echo "  ERROR: brain_tumor/__init__.py not found."
    echo "  Expected layout:"
    echo "    brain-tumor-classifier/"
    echo "    ├── setup.py  (or pyproject.toml)"
    echo "    └── brain_tumor/"
    echo "        ├── __init__.py"
    echo "        ├── config.py"
    echo "        └── data/"
    echo "            ├── __init__.py"
    echo "            └── dataset.py"
    exit 1
fi
echo "  Found: brain_tumor/__init__.py"

echo ""
echo "── Step 5: reinstall in editable mode ──"
pip install -e .

echo ""
echo "── Step 6: smoke-test the import ──"
python - <<'PYEOF'
try:
    from brain_tumor.data import (
        download_dataset,
        build_dataloaders,
        BrainTumorDataset,
        get_train_transform,
        get_val_transform,
    )
    from brain_tumor.config import IMG_SIZE, CKPT_S1, CKPT_S2, FINAL_PATH
    import brain_tumor
    print(f"  brain_tumor location : {brain_tumor.__file__}")
    print(f"  download_dataset     : {download_dataset}")
    print(f"  IMG_SIZE             : {IMG_SIZE}")
    print(f"  CKPT_S1              : {CKPT_S1}")
    print("  All imports OK ✔")
except ImportError as e:
    print(f"  IMPORT FAILED: {e}")
    raise
PYEOF

echo ""
echo "Done. You can now run: jupyter notebook brain_tumor_classifier.ipynb"
