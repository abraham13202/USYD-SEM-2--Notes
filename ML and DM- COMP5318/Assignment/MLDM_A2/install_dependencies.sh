#!/bin/bash

# Installation script for MacBook Pro M4 with GPU support
# Run this with: bash install_dependencies.sh

echo "======================================"
echo "Installing Dependencies for Assignment 2"
echo "MacBook Pro M4 Pro - GPU Accelerated"
echo "======================================"
echo ""

# Upgrade pip
echo "1. Upgrading pip..."
pip3 install --upgrade pip

# Install core scientific packages
echo ""
echo "2. Installing NumPy, Pandas, Matplotlib, Seaborn..."
pip3 install numpy pandas matplotlib seaborn

# Install scikit-learn
echo ""
echo "3. Installing scikit-learn..."
pip3 install scikit-learn

# Install TensorFlow for Mac (with Metal GPU support)
echo ""
echo "4. Installing TensorFlow with GPU support for Apple Silicon..."
pip3 install tensorflow-macos tensorflow-metal

# Install Keras Tuner
echo ""
echo "5. Installing Keras Tuner..."
pip3 install keras-tuner

# Install Jupyter
echo ""
echo "6. Installing Jupyter..."
pip3 install jupyter ipykernel

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""

# Verify installation
echo "Verifying installation..."
python3 << END
import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not installed")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except ImportError:
    print("✗ Pandas not installed")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib not installed")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn not installed")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU Available: {gpus}")
        print("  🚀 Your M4 Pro GPU is ready for training!")
    else:
        print("⚠ GPU not detected (but TensorFlow is installed)")
except ImportError:
    print("✗ TensorFlow not installed")

try:
    import keras_tuner
    print(f"✓ Keras Tuner installed")
except ImportError:
    print("✗ Keras Tuner not installed")

END

echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo "1. Open VSCode"
echo "2. Install Jupyter extension in VSCode"
echo "3. Open: COMP5318-assignment2-template-notebook (1).ipynb"
echo "4. Click 'Run All' or run cells individually"
echo ""
echo "Estimated training time with M4 Pro GPU: 30-60 minutes"
echo "======================================"
