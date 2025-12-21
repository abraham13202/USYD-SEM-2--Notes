# Setup Guide - Running on MacBook Pro M4 Pro with GPU

## Prerequisites

Your M4 Pro chip supports Metal GPU acceleration for TensorFlow! This will significantly speed up training.

## Step 1: Install Python Packages

Open Terminal and run:

```bash
cd "/Users/ABRAHAM/Documents/USYD/Sem 2/ML and DM- COMP5318/Assignment/MLDM_A2"

# Install required packages
pip3 install --upgrade pip
pip3 install numpy pandas matplotlib seaborn scikit-learn
pip3 install tensorflow-macos tensorflow-metal
pip3 install keras-tuner
pip3 install jupyter ipykernel
```

**Note:** `tensorflow-macos` and `tensorflow-metal` are optimized for Apple Silicon and will automatically use your M4 Pro GPU!

## Step 2: Verify GPU is Available

Run this to check GPU detection:

```bash
python3 -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
```

You should see output like: `GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

## Step 3: Open in VSCode

1. Open VSCode
2. Install the "Jupyter" extension (if not already installed)
3. Open the notebook: `COMP5318-assignment2-template-notebook (1).ipynb`
4. Select Python interpreter (top right) - choose the one with tensorflow installed

## Step 4: Run the Notebook

### Option A: Run All Cells (Full Training)
- Click "Run All" at the top
- **Estimated time with GPU: 30-60 minutes**
- Let it run completely

### Option B: Run Selectively (Faster Testing)
Run sections in this order:

1. **Setup and Data Loading** (cells 1-20) - 30 seconds
2. **Skip Hyperparameter Tuning** (optional, takes 1-2 hours)
3. **Final Models** - Just run these with pre-set hyperparameters - 15-20 minutes

## Expected Training Times (with M4 Pro GPU)

| Section | Time |
|---------|------|
| Data Loading & Exploration | 1-2 min |
| Random Forest Training | 5-10 min |
| MLP Hyperparameter Tuning | 10-15 min |
| CNN Hyperparameter Tuning | 20-30 min |
| Final Models Training | 15-20 min |
| **Total (if running all)** | **50-80 min** |

## Tips for Faster Execution

If you want to speed things up for testing:

1. **Reduce hyperparameter search space:**
   - In MLP tuning cell: Change `max_trials=20` to `max_trials=5`
   - In CNN tuning cell: Change `max_trials=20` to `max_trials=5`

2. **Reduce epochs:**
   - In training cells: Change `epochs=100` to `epochs=30`

3. **Use subset of data (for testing only):**
   - After loading data, add: `X_train = X_train[:10000]` and `y_train = y_train[:10000]`

## Monitoring GPU Usage

While training, open another terminal and run:
```bash
# Monitor GPU usage
python3 -c "
import tensorflow as tf
import time
while True:
    print(tf.config.experimental.get_memory_info('GPU:0'))
    time.sleep(5)
"
```

Or use Activity Monitor:
- Open Activity Monitor
- Go to "GPU" tab
- Watch GPU usage during training

## Troubleshooting

### Issue: "No GPU devices found"
```bash
# Reinstall TensorFlow for Mac
pip3 uninstall tensorflow tensorflow-macos tensorflow-metal
pip3 install tensorflow-macos tensorflow-metal
```

### Issue: "Out of Memory"
- Reduce batch size: Change `batch_size=128` to `batch_size=64` or `batch_size=32`
- Close other applications

### Issue: Kernel crashes
- Restart VSCode
- Reduce model complexity or batch size

## Expected Results

With the improved CNN, you should see:
- **Random Forest:** ~46% accuracy
- **MLP:** ~50-52% accuracy
- **CNN (Improved):** ~82-87% accuracy ⭐

## Saving Your Work

The notebook will automatically save outputs. Make sure to:
1. Save the notebook after running (Cmd+S)
2. Keep the output cells intact for submission
3. The best models are saved as `best_cnn_model.keras`

## Ready to Submit

After running:
1. ✅ All cells have output
2. ✅ Visualizations are displayed
3. ✅ Summary tables show results
4. ✅ Save notebook as `.ipynb` for submission

---

**Your M4 Pro will make this much faster than a regular CPU! Enjoy the GPU acceleration! 🚀**
