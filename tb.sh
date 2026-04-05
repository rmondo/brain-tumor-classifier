# purge cache
conda activate tf_m2

python -m pip uninstall -y tensorflow tensorflow-macos tensorflow-metal tensorboard tensorboard-data-server keras numpy
python -m pip cache purge

python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install "numpy<2"
python -m pip install tensorflow tensorboard

# test
python -c "import numpy as np; print('numpy', np.__version__)"
python -c "import tensorboard; print('tensorboard ok', tensorboard.__version__)"
python -c "from tensorboard.plugins.hparams import backend_context; print('hparams ok')"
python -c "import tensorflow as tf; print('tf ok', tf.__version__)"

# launch tensorboard
python -m tensorboard.main --logdir "./runs/brain_tumor" --port 6006 --host 127.0.0.1
