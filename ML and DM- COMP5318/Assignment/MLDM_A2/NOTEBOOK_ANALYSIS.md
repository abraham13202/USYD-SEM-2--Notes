# COMPREHENSIVE NOTEBOOK ANALYSIS
## COMP5318-assignment2-FINAL-IMPROVED-FIXED.ipynb

Generated: 2025-10-21

---

## TABLE OF CONTENTS
1. [Data Flow Overview](#data-flow-overview)
2. [Cell-by-Cell Breakdown](#cell-by-cell-breakdown)
3. [Training Pipeline Analysis](#training-pipeline-analysis)
4. [Critical Issues Identified](#critical-issues-identified)
5. [Variable Reference Guide](#variable-reference-guide)

---

## DATA FLOW OVERVIEW

### Stage 1: Data Loading (Cell 7)
```
CIFAR-10 Dataset (from .npy files)
├── X_train: (50000, 32, 32, 3) - RGB images, uint8, range [0, 255]
├── y_train: (50000,) - Labels 0-9
├── X_test:  (10000, 32, 32, 3) - RGB images, uint8, range [0, 255]
└── y_test:  (10000,) - Labels 0-9
```

### Stage 2: Normalization (Cell 20)
```
X_train (50000, 32, 32, 3) [0-255]
    ↓ / 255.0
X_train_normalized (50000, 32, 32, 3) [0.0-1.0]

X_test (10000, 32, 32, 3) [0-255]
    ↓ / 255.0
X_test_normalized (10000, 32, 32, 3) [0.0-1.0]
```

### Stage 3: Label Encoding (Cell 21)
```
y_train (50000,) [0-9]
    ↓ to_categorical()
y_train_categorical (50000, 10) - One-hot encoded
```

### Stage 4: Train/Validation Split (Cell 22)
```
X_train_normalized (50000, 32, 32, 3)
    ↓ train_test_split(test_size=0.2, random_state=42)
    ├── X_train_final (40000, 32, 32, 3) - 80% for training
    └── X_val (10000, 32, 32, 3) - 20% for validation

y_train (50000,)
    ↓ train_test_split(test_size=0.2, random_state=42)
    ├── y_train_final (40000,)
    └── y_val (10000,)

y_train_categorical (50000, 10)
    ↓ train_test_split(test_size=0.2, random_state=42)
    ├── y_train_final_cat (40000, 10)
    └── y_val_cat (10000, 10)
```

### Stage 5: Data Transformations for Different Models

#### For Random Forest (Cell 28):
```
X_train_final (40000, 32, 32, 3) [0-1]
    ↓ extract_enhanced_features() - HOG, color histograms, stats
X_train_enhanced (40000, 1124) - Enhanced features

X_val (10000, 32, 32, 3) [0-1]
    ↓ extract_enhanced_features()
X_val_enhanced (10000, 1124)

X_test_normalized (10000, 32, 32, 3) [0-1]
    ↓ extract_enhanced_features()
X_test_enhanced (10000, 1124)
```

#### For MLP (Cell 28, 30):
```
X_train_final (40000, 32, 32, 3) [0-1]
    ↓ reshape(-1)
X_train_flat (40000, 3072)

X_val (10000, 32, 32, 3) [0-1]
    ↓ reshape(-1)
X_val_flat (10000, 3072)

X_test_normalized (10000, 32, 32, 3) [0-1]
    ↓ reshape(-1)
X_test_flat (10000, 3072)
```

#### For CNN (Cell 25, 38):
```
X_train_final (40000, 32, 32, 3) [0-1]
    ↓ ImageDataGenerator (rotation, shift, flip, zoom, shear)
datagen.flow() → Augmented batches

X_val (10000, 32, 32, 3) [0-1]
    ↓ ImageDataGenerator (NO augmentation)
val_datagen.flow() → Non-augmented batches
```

### Stage 6: FINAL MODEL TRAINING (Cell 53 - Preparation)
```
ORIGINAL DATA REUSED:
X_train (50000, 32, 32, 3) [0-255]
    ↓ / 255.0
X_train_full_normalized (50000, 32, 32, 3) [0.0-1.0]

TRANSFORMATIONS:
├── For RF: extract_enhanced_features()
│   → X_train_full_enhanced (50000, 1124)
│
├── For MLP: reshape(-1)
│   → X_train_full_flat (50000, 3072)
│
└── For CNN: keep as-is + augmentation
    → X_train_full_images (50000, 32, 32, 3)
```

---

## CELL-BY-CELL BREAKDOWN

### SECTION 1: Setup and Data Loading (Cells 0-7)
- **Cell 0-3**: Markdown headers and descriptions
- **Cell 4**: Import all libraries (numpy, sklearn, tensorflow, etc.)
- **Cell 7**: **[DATA INPUT]** Load CIFAR-10 from .npy files
  - Creates: `X_train, y_train, X_test, y_test`

### SECTION 2: Data Exploration (Cells 8-18)
- **Cell 8-13**: Basic data inspection (shapes, types, statistics)
- **Cell 14-18**: Visualizations (sample images, class distribution, t-SNE)

### SECTION 3: Data Preprocessing (Cells 19-23)
- **Cell 20**: **[DATA TRANSFORM]** Normalize to [0, 1]
  - Creates: `X_train_normalized, X_test_normalized`
- **Cell 21**: **[DATA TRANSFORM]** One-hot encode labels
  - Creates: `y_train_categorical`
- **Cell 22**: **[DATA SPLIT]** Create train/val split (80/20)
  - Creates: `X_train_final, X_val, y_train_final, y_val, y_train_final_cat, y_val_cat`
- **Cell 23**: Visualization of preprocessed data

### SECTION 4: Data Augmentation Setup (Cells 24-26)
- **Cell 25**: **[AUGMENTATION]** Create ImageDataGenerator
  - Creates: `datagen` (with rotation, shift, flip, zoom, shear)
  - **FIXED**: No brightness_range or channel_shift_range
  - Fits on: `X_train_final`
- **Cell 26**: Visualize augmented images

### SECTION 5: Initial Model Training (Cells 27-40)

#### Random Forest (Cells 29-31)
- **Cell 28**: **[FEATURE ENGINEERING]** Extract enhanced features
  - Creates: `X_train_enhanced, X_val_enhanced, X_test_enhanced`
  - Features: HOG descriptors, color histograms, statistics
- **Cell 31**: **[TRAINING #1]** Train simple Random Forest
  - Model: `rf_model` (100 estimators)
  - Trains on: `X_train_flat, y_train_final`
  - Validates on: `X_val_flat, y_val`
  - **OUTPUT**: Validation accuracy

#### MLP (Cells 32-35)
- **Cell 33**: Define `create_mlp()` function
- **Cell 34**: **[TRAINING #2]** Train MLP with early stopping
  - Model: `mlp_model`
  - Trains on: `X_train_flat, y_train_final_cat`
  - **HAS VALIDATION**: `validation_data=(X_val_flat, y_val_cat)`
  - Callbacks: EarlyStopping(monitor='val_loss', patience=5)
  - **OUTPUT**: `history_mlp`
- **Cell 35**: Plot training history

#### CNN (Cells 36-39)
- **Cell 37**: Define `create_cnn_improved()` function
  - Architecture: 4 Conv blocks [64→128→256→512]
  - BatchNorm, Dropout, L2 regularization
- **Cell 38**: **[TRAINING #3]** Train CNN with augmentation
  - Model: `cnn_improved`
  - Trains on: `datagen.flow(X_train_final, y_train_final_cat)`
  - **FIXED**: Uses `val_generator` for validation
  - **HAS VALIDATION**: `validation_data=val_generator, validation_steps=...`
  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  - **OUTPUT**: `history_cnn_improved`
- **Cell 39**: Plot training history

### SECTION 6: Hyperparameter Tuning (Cells 41-51)

#### Random Forest Tuning (Cells 42-44)
- **Cell 43**: **[TRAINING #4]** GridSearchCV for Random Forest
  - Tunes: n_estimators, max_depth, min_samples_split, min_samples_leaf
  - Uses: `X_train_flat, y_train_final`
  - Cross-validation: 3-fold CV
  - **OUTPUT**: Best params, best CV score
- **Cell 44**: Visualize tuning results

#### MLP Tuning (Cells 45-47)
- **Cell 46**: **[TRAINING #5]** Keras Tuner for MLP
  - Tunes: hidden layer sizes, dropout, learning rate
  - Uses: `X_train_flat, y_train_final_cat`
  - **HAS VALIDATION**: Uses validation data for tuning
  - **OUTPUT**: Best MLP hyperparameters
- **Cell 47**: Visualize tuning results

#### CNN Tuning (Cells 48-50)
- **Cell 49**: **[TRAINING #6]** Keras Tuner for CNN
  - Tunes: filters, kernel size, dense units, dropout, learning rate
  - Uses: `X_train_final, y_train_final_cat` with augmentation
  - **HAS VALIDATION**: Uses validation data for tuning
  - **OUTPUT**: Best CNN hyperparameters
- **Cell 50**: Visualize tuning results

### SECTION 7: FINAL MODEL TRAINING (Cells 52-61)

#### Data Preparation (Cell 53)
- **Cell 53**: **[DATA TRANSFORM]** Prepare FULL training data
  - Creates:
    - `X_train_full_normalized` (50000, 32, 32, 3) [0-1]
    - `X_train_full_enhanced` (50000, 1124) - for RF
    - `X_train_full_flat` (50000, 3072) - for MLP
    - `X_train_full_images` (50000, 32, 32, 3) - for CNN
    - `y_train_full` (50000,) - original labels
    - `y_train_full_cat` (50000, 10) - categorical labels
    - `y_test_cat` (10000, 10) - categorical test labels

#### Random Forest Final (Cells 55-56)
- **Cell 56**: **[TRAINING #7]** Train final Random Forest
  - Model: `rf_final`
  - Trains on: `X_train_full_enhanced, y_train_full` (50000 samples)
  - **NO VALIDATION DATA** ❌
  - Hyperparameters: Best from GridSearch
  - **OUTPUT**: `rf_test_accuracy` (evaluated on test set only)

#### MLP Final (Cells 57-58)
- **Cell 58**: **[TRAINING #8]** Train final MLP
  - Model: `mlp_final`
  - Trains on: `X_train_full_flat, y_train_full_cat` (50000 samples)
  - **NO VALIDATION DATA** ❌
  - Callbacks: EarlyStopping(monitor='**loss**', patience=10) ⚠️
    - **ISSUE**: Monitors 'loss', not 'val_loss' (no validation data!)
  - Hyperparameters: Best from Tuner
  - **OUTPUT**: `history_mlp_final`, `mlp_test_accuracy`

#### CNN Final (Cells 59-61)
- **Cell 60**: **[TRAINING #9]** Train final CNN
  - Model: `cnn_final`
  - Trains on: `datagen_full.flow(X_train_full_images, y_train_full_cat)` (50000 samples)
  - **NO VALIDATION DATA** ❌
  - Callbacks: EarlyStopping(monitor='**loss**', patience=15) ⚠️
    - **ISSUE**: Monitors 'loss', not 'val_loss' (no validation data!)
  - Data augmentation: rotation, shift, flip, zoom, shear
  - **FIXED**: No brightness/channel shift
  - **OUTPUT**: `history_cnn_final`
  - **SPECIAL**: Test-Time Augmentation (TTA) for predictions
    - Creates: `y_test_pred_cnn_tta` (averaged over 5 augmentations)
- **Cell 61**: Compare TTA vs non-TTA predictions

### SECTION 8: Results and Analysis (Cells 62-67)
- **Cell 62**: Final results summary table
- **Cell 63**: Confusion matrices for all 3 models
- **Cell 64**: Training history comparison
- **Cell 65**: Per-class performance analysis
- **Cell 67**: Comprehensive analysis and recommendations

---

## TRAINING PIPELINE ANALYSIS

### Training Phase 1: Initial Models (WITH Validation)
**Purpose**: Quick baseline, understand data

1. **Random Forest** (Cell 31)
   - Data: X_train_flat (40000 samples)
   - Validation: X_val_flat (10000 samples) ✓
   - Monitoring: Manual accuracy calculation

2. **MLP** (Cell 34)
   - Data: X_train_flat (40000 samples)
   - Validation: (X_val_flat, y_val_cat) ✓
   - Monitoring: EarlyStopping on val_loss ✓
   - Epochs: Up to 50 (early stopped)

3. **CNN** (Cell 38)
   - Data: datagen.flow(X_train_final) (40000 samples)
   - Validation: val_generator (10000 samples) ✓
   - Monitoring: EarlyStopping on val_loss ✓
   - Epochs: Up to 100 (early stopped)

### Training Phase 2: Hyperparameter Tuning (WITH Validation)
**Purpose**: Find optimal hyperparameters

1. **Random Forest GridSearch** (Cell 43)
   - Uses 3-fold cross-validation ✓
   - Data: X_train_flat (40000 samples)

2. **MLP Keras Tuner** (Cell 46)
   - Uses validation split ✓
   - Data: X_train_flat (40000 samples)
   - Trials: Multiple configurations

3. **CNN Keras Tuner** (Cell 49)
   - Uses validation split ✓
   - Data: X_train_final with augmentation (40000 samples)
   - Trials: Multiple configurations

### Training Phase 3: FINAL Models (WITHOUT Validation)
**Purpose**: Train best model on maximum data for submission

1. **Random Forest Final** (Cell 56)
   - Data: X_train_full_enhanced (50000 samples)
   - **NO validation data** ❌
   - **NO cross-validation** ❌
   - Hyperparameters: From GridSearch (best_params)
   - Evaluation: **Only on test set**

2. **MLP Final** (Cell 58)
   - Data: X_train_full_flat (50000 samples)
   - **NO validation_data parameter** ❌
   - Callbacks: EarlyStopping(monitor='loss', patience=10)
     - **ISSUE**: Monitors training loss, not validation loss
     - Will stop if training loss doesn't improve (overfitting indicator)
   - Hyperparameters: From Tuner
   - Evaluation: **Only on test set**

3. **CNN Final** (Cell 60)
   - Data: datagen_full.flow(X_train_full_images) (50000 samples)
   - **NO validation_data parameter** ❌
   - Callbacks:
     - EarlyStopping(monitor='loss', patience=15) ⚠️
     - ReduceLROnPlateau(monitor='loss') ⚠️
     - Cosine learning rate schedule
   - Hyperparameters: From Tuner
   - Evaluation: **Only on test set** (with TTA)

---

## CRITICAL ISSUES IDENTIFIED

### Issue 1: Final Models Train WITHOUT Validation Set ❌
**Location**: Cells 56, 58, 60

**Problem**:
- All final models train on full 50,000 samples
- No validation data is held out
- Cannot monitor generalization during training

**Impact**:
- **MLP & CNN**: Early stopping monitors 'loss' (training loss) instead of 'val_loss'
  - May stop early if overfitting begins (training loss stops improving)
  - OR may train too long (if training loss keeps decreasing)
- **Random Forest**: No issue (doesn't use validation)
- **Risk of overfitting**: No way to detect if models overfit during training

**Evidence**:
```python
# Cell 58 - MLP
callbacks_mlp = [
    EarlyStopping(monitor='loss', patience=10, ...)  # ← monitoring 'loss' not 'val_loss'
]
history_mlp_final = mlp_final.fit(
    X_train_full_flat, y_train_full_cat,  # ← No validation_data
    epochs=60,
    callbacks=callbacks_mlp
)

# Cell 60 - CNN
callbacks_final = [
    EarlyStopping(monitor='loss', patience=15, ...),  # ← monitoring 'loss'
    ReduceLROnPlateau(monitor='loss', ...),  # ← monitoring 'loss'
    ...
]
history_cnn_final = cnn_final.fit(
    datagen_full.flow(X_train_full_images, y_train_full_cat, ...),  # ← No validation_data
    epochs=100,
    callbacks=callbacks_final
)
```

### Issue 2: Data Augmentation Was Producing Black Images (FIXED) ✓
**Location**: Cell 25

**Problem** (BEFORE FIX):
- brightness_range=[0.8, 1.2] and channel_shift_range=0.1
- These parameters assume data in [0, 255] range
- Data is normalized to [0, 1]
- Results in pixel values < 0 or > 1, appearing black

**Solution** (APPLIED):
- Removed brightness_range and channel_shift_range
- Kept geometric augmentations (rotation, shift, flip, zoom, shear)
- Works correctly with [0, 1] normalized data

**Status**: ✓ FIXED in current notebook

### Issue 3: Low Validation Accuracy in Initial CNN (FIXED) ✓
**Location**: Cell 38

**Problem** (BEFORE FIX):
- Used tuple for validation_data while using generator for training
- Inconsistent batch handling

**Solution** (APPLIED):
- Created val_datagen and val_generator
- Uses generator for both training and validation
- Consistent data handling

**Status**: ✓ FIXED in current notebook

---

## VARIABLE REFERENCE GUIDE

### Original Data Variables
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `X_train` | (50000, 32, 32, 3) | uint8 | [0, 255] | Cell 7 | Original training images |
| `y_train` | (50000,) | int64 | [0, 9] | Cell 7 | Original training labels |
| `X_test` | (10000, 32, 32, 3) | uint8 | [0, 255] | Cell 7 | Original test images |
| `y_test` | (10000,) | int64 | [0, 9] | Cell 7 | Original test labels |

### Normalized Data Variables
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `X_train_normalized` | (50000, 32, 32, 3) | float32 | [0.0, 1.0] | Cell 20 | Normalized training images |
| `X_test_normalized` | (10000, 32, 32, 3) | float32 | [0.0, 1.0] | Cell 20 | Normalized test images |
| `X_train_full_normalized` | (50000, 32, 32, 3) | float32 | [0.0, 1.0] | Cell 53 | Full training data (for final models) |

### Split Data Variables (80/20 split)
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `X_train_final` | (40000, 32, 32, 3) | float32 | [0.0, 1.0] | Cell 22 | Training images (80%) |
| `X_val` | (10000, 32, 32, 3) | float32 | [0.0, 1.0] | Cell 22 | Validation images (20%) |
| `y_train_final` | (40000,) | int64 | [0, 9] | Cell 22 | Training labels (80%) |
| `y_val` | (10000,) | int64 | [0, 9] | Cell 22 | Validation labels (20%) |

### Categorical Label Variables
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `y_train_categorical` | (50000, 10) | float32 | [0.0, 1.0] | Cell 21 | One-hot encoded full training labels |
| `y_train_final_cat` | (40000, 10) | float32 | [0.0, 1.0] | Cell 22 | One-hot training labels (80%) |
| `y_val_cat` | (10000, 10) | float32 | [0.0, 1.0] | Cell 22 | One-hot validation labels (20%) |
| `y_train_full_cat` | (50000, 10) | float32 | [0.0, 1.0] | Cell 53 | One-hot full training labels (final) |
| `y_test_cat` | (10000, 10) | float32 | [0.0, 1.0] | Cell 53 | One-hot test labels |

### Flattened Data Variables (for MLP/RF)
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `X_train_flat` | (40000, 3072) | float32 | [0.0, 1.0] | Cell 28, 30 | Flattened training images |
| `X_val_flat` | (10000, 3072) | float32 | [0.0, 1.0] | Cell 28, 30 | Flattened validation images |
| `X_test_flat` | (10000, 3072) | float32 | [0.0, 1.0] | Cell 28, 30 | Flattened test images |
| `X_train_full_flat` | (50000, 3072) | float32 | [0.0, 1.0] | Cell 53 | Flattened full training (final MLP) |

### Enhanced Feature Variables (for Random Forest)
| Variable | Shape | Type | Range | Created In | Purpose |
|----------|-------|------|-------|------------|---------|
| `X_train_enhanced` | (40000, 1124) | float64 | varies | Cell 28 | Enhanced features (HOG, color, stats) |
| `X_val_enhanced` | (10000, 1124) | float64 | varies | Cell 28 | Enhanced validation features |
| `X_test_enhanced` | (10000, 1124) | float64 | varies | Cell 28 | Enhanced test features |
| `X_train_full_enhanced` | (50000, 1124) | float64 | varies | Cell 53 | Enhanced full training (final RF) |

### Data Augmentation Variables
| Variable | Type | Created In | Purpose |
|----------|------|------------|---------|
| `datagen` | ImageDataGenerator | Cell 25 | Augmentation for initial CNN training |
| `val_datagen` | ImageDataGenerator | Cell 38 | No-augmentation generator for validation |
| `val_generator` | Generator | Cell 38 | Validation data generator |
| `datagen_full` | ImageDataGenerator | Cell 60 | Augmentation for final CNN training |

### Model Variables
| Variable | Type | Created In | Purpose |
|----------|------|------------|---------|
| `rf_model` | RandomForestClassifier | Cell 31 | Initial Random Forest |
| `mlp_model` | Sequential | Cell 34 | Initial MLP |
| `cnn_improved` | Sequential | Cell 38 | Initial CNN |
| `rf_final` | RandomForestClassifier | Cell 56 | Final Random Forest |
| `mlp_final` | Sequential | Cell 58 | Final MLP |
| `cnn_final` | Sequential | Cell 60 | Final CNN |

### Training History Variables
| Variable | Type | Created In | Purpose |
|----------|------|------------|---------|
| `history_mlp` | History | Cell 34 | MLP initial training history |
| `history_cnn_improved` | History | Cell 38 | CNN initial training history |
| `history_mlp_final` | History | Cell 58 | MLP final training history |
| `history_cnn_final` | History | Cell 60 | CNN final training history |

### Prediction/Accuracy Variables
| Variable | Type | Created In | Purpose |
|----------|------|------------|---------|
| `y_test_pred_rf` | ndarray | Cell 56 | Random Forest test predictions |
| `rf_test_accuracy` | float | Cell 56 | Random Forest test accuracy |
| `mlp_test_accuracy` | float | Cell 58 | MLP test accuracy |
| `y_test_pred_cnn_tta` | ndarray | Cell 60 | CNN test predictions (with TTA) |
| `cnn_test_accuracy_tta` | float | Cell 60 | CNN test accuracy (with TTA) |

---

## DATA SIZE SUMMARY

### Initial Training (WITH Validation Split)
- **Training**: 40,000 samples (80%)
- **Validation**: 10,000 samples (20%)
- **Test**: 10,000 samples (held out)

### Final Training (WITHOUT Validation Split)
- **Training**: 50,000 samples (100% of original training data)
- **Validation**: 0 samples ❌
- **Test**: 10,000 samples (held out)

### Data Usage Breakdown
| Phase | RF | MLP | CNN |
|-------|----|----|-----|
| **Initial Training** | 40K (flat) | 40K (flat) | 40K (images + aug) |
| **Hyperparameter Tuning** | 40K (flat + 3-fold CV) | 40K (flat + val) | 40K (images + aug + val) |
| **Final Training** | 50K (enhanced features) | 50K (flat) | 50K (images + aug) |
| **Validation in Final** | ❌ None | ❌ None | ❌ None |
| **Test Evaluation** | 10K (enhanced) | 10K (flat) | 10K (images + TTA) |

---

## SUMMARY

### What Works Well ✓
1. Clear data pipeline from raw images to model-specific formats
2. Proper normalization to [0, 1] range
3. Train/val split for hyperparameter tuning
4. Data augmentation fixed for normalized data
5. Enhanced features for Random Forest (HOG, color, stats)
6. Test-Time Augmentation for CNN
7. Comprehensive evaluation and visualization

### What Needs Attention ⚠️
1. **Final models train without validation data**
   - Cannot monitor generalization during training
   - Early stopping monitors 'loss' instead of 'val_loss'
   - Risk of overfitting

2. **Callbacks configuration for final models**
   - Should either:
     - Keep validation split (recommended)
     - OR remove EarlyStopping/ReduceLR callbacks
     - OR use different stopping criteria

### Recommendations
1. **Option A**: Keep validation split even for final training
   - More reliable generalization monitoring
   - Better overfitting prevention
   - Slight reduction in training data (40K vs 50K)

2. **Option B**: Train on full data with modified approach
   - Remove validation-dependent callbacks
   - Use fixed epoch count or LR schedule
   - Accept higher overfitting risk
   - Maximum training data utilization

---

END OF ANALYSIS
