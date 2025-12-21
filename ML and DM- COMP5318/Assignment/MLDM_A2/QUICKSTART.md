# 🚀 QUICK START GUIDE

## Run Your Assignment with GPU Acceleration (M4 Pro)

### Step 1: Install Dependencies (One-Time Setup)

Open **Terminal** and run:

```bash
cd "/Users/ABRAHAM/Documents/USYD/Sem 2/ML and DM- COMP5318/Assignment/MLDM_A2"
bash install_dependencies.sh
```

This will install everything you need including TensorFlow with GPU support for your M4 Pro!

### Step 2: Open in VSCode

1. Open **VSCode**
2. Open folder: `/Users/ABRAHAM/Documents/USYD/Sem 2/ML and DM- COMP5318/Assignment/MLDM_A2`
3. Open file: `COMP5318-assignment2-template-notebook (1).ipynb`
4. If prompted, install the **Jupyter extension**

### Step 3: Run the Notebook

Click **"Run All"** at the top of the notebook, or run cells individually.

## ⏱️ Expected Time (with M4 Pro GPU)

- **Full run:** 30-60 minutes
- **Without hyperparameter tuning:** 10-15 minutes

## 📊 What You'll Get

- **Random Forest:** ~46% accuracy
- **MLP:** ~50-52% accuracy
- **CNN (Improved):** ~82-87% accuracy ⭐ (BEST)

## 💡 Pro Tips

### For Faster Testing (Optional):
Edit these cells to speed things up:

**Hyperparameter Tuning Cells:**
```python
# Change from:
max_trials=20

# To:
max_trials=5
```

**Training Cells:**
```python
# Change from:
epochs=100

# To:
epochs=30
```

### Monitor Your GPU:
While training, open Activity Monitor → GPU tab to see your M4 Pro in action!

## 🎯 What Makes This Good

Your improved CNN includes:
- ✅ Data Augmentation
- ✅ Batch Normalization
- ✅ Learning Rate Scheduling
- ✅ Early Stopping
- ✅ 3 Deep Convolutional Blocks

All **100% specification-compliant** (no pre-built models)!

## 📝 After Running

Your notebook will have:
- ✅ All outputs preserved
- ✅ Visualizations displayed
- ✅ Results tables filled in
- ✅ Ready to submit!

## ❓ Issues?

See `SETUP_GUIDE.md` for detailed troubleshooting.

---

**Ready? Just run the install script and open the notebook in VSCode!** 🎉
