"""
setup.py
────────
Project root install script for the Brain Tumor MRI Classifier.

FILE PLACEMENT
──────────────
Per ARCHITECTURE.md, the brain_tumor/ package lives inside notebooks/:

    brain-tumor-classifier/          ← run `pip install -e .` HERE (project root)
    ├── setup.py                     ← this file
    ├── pyproject.toml               ← preferred by pip >= 21.3
    ├── notebooks/
    │   ├── brain_tumor/             ← importable package (not at root!)
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   ├── data/
    │   │   ├── models/
    │   │   ├── training/
    │   │   └── evaluation/
    │   └── brain_tumor_classifier_ref.ipynb
    ├── src/
    │   └── app/
    ├── data/
    ├── models/
    ├── reports/
    └── requirements.txt

Because the package is NOT at the project root, find_packages() alone won't
find it. package_dir tells setuptools to look inside notebooks/ for packages.

USAGE
─────
    pip install -e .                   # editable, core deps only
    pip install -e ".[dev]"            # + pytest + pytest-cov
    pip install -e ".[notebook]"       # + jupyter, kaggle, ipywidgets
    pip install -e ".[dev,notebook]"   # full development environment
"""

from setuptools import find_packages, setup

setup(
    name="brain-tumor-classifier",
    version="1.0.0",
    description="EfficientNetB0-based Brain Tumor MRI Classifier — PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",

    # Package lives at repository root in brain_tumor/.
    package_dir={"": "."},
    packages=find_packages(
        where=".",
        include=["brain_tumor", "brain_tumor.*"],
        exclude=["tests", "tests.*"],
    ),

    install_requires=[
        # ── Deep learning ──────────────────────────────────────────────────
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "efficientnet-pytorch>=0.7.1",
        # ── Experiment tracking ────────────────────────────────────────────
        "tensorboard>=2.14.0",
        # ── Data / compute ─────────────────────────────────────────────────
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        # ── Visualisation ──────────────────────────────────────────────────
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # ── Progress bars ──────────────────────────────────────────────────
        "tqdm>=4.66.0",
        # ── Flask inference server ─────────────────────────────────────────
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
    ],

    extras_require={
        # pip install -e ".[dev]"
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        # pip install -e ".[notebook]"
        "notebook": [
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
            "kaggle>=1.6.0",
            "pyyaml>=6.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "brain-tumor-train=brain_tumor.train:main",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
