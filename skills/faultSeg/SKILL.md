---
name: faultSeg
description: "Use when you need a repository-specific analysis of faultSeg (FaultSeg3D): summarize its purpose, explain each important script/module/function, describe execution flow, and show how to run it. faultSeg trains and predicts 3D seismic fault segmentation using a simplified 3D U-Net CNN trained entirely on synthetic seismic/fault volume pairs in raw binary (.dat) format."
---

# FaultSeg3D Repository Analysis

## Purpose

**FaultSeg3D** is a Keras/TensorFlow implementation of a 3D convolutional neural network for **automatic fault segmentation in 3D seismic volumes**. Given a 3D seismic amplitude cube, the model outputs a corresponding fault-probability volume in which high values indicate likely fault surfaces.

The key innovation is that the CNN is trained entirely on **synthetic** seismic/fault pairs (200 pairs), yet generalises well to real field datasets (Netherlands F3, Clyde, Costa Rica, Campos, Kerry-3D, Opunake-3D). This eliminates the need for hand-labelled field data.

Published as:
> Wu et al. (2019). *FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation*. **GEOPHYSICS**, 84(3), IM35–IM45.

---

## Verified Repository Summary

| Item | Detail |
|---|---|
| Language | Python 3 |
| Framework | Keras (TensorFlow 1.x backend) |
| Input format | Raw binary `.dat` files, `float32`, C order |
| Training volume size | 128 × 128 × 128 voxels |
| Network | Simplified 3D U-Net (4 levels, 16–128 filters) |
| Loss | `binary_crossentropy` (Adam, lr=1e-4) |
| Training epochs | 100 |
| Training samples | 200 synthetic pairs |
| Validation samples | 20 synthetic pairs |
| License | CC BY-NC 4.0 (non-commercial research only) |

---

## Repository Walkthrough

### Top-level structure

```
faultSeg/
├── train.py          # Training entrypoint
├── apply.py          # Inference/evaluation script
├── unet3.py          # 3D U-Net model definition + custom loss
├── utils.py          # Keras DataGenerator for .dat files
├── predNew.ipynb     # Newer interactive prediction notebook
├── prediction.ipynb  # Original prediction notebook
├── cpurun            # Shell wrapper: forces CPU (CUDA_VISIBLE_DEVICES='')
├── model/
│   └── model3.json   # Exported JSON of a trained model architecture
├── data/
│   ├── train/        # (downloaded separately) seis/ fault/
│   ├── validation/   # (downloaded separately) seis/ fault/
│   └── prediction/   # Field data for inference
├── log/              # TensorBoard log directory
└── results/          # Result images for the README
```

---

### `unet3.py` — 3D U-Net model

**Role:** Defines the CNN architecture and the optional balanced cross-entropy loss.

#### `unet(pretrained_weights=None, input_size=(None, None, None, 1))`

Builds and returns a simplified 3D U-Net Keras model. The architecture:

**Encoder (contracting path):**
| Block | Layers | Output filters |
|---|---|---|
| Block 1 | Conv3D(16) × 2, ReLU, same padding | 16 |
| Pool 1 | MaxPooling3D(2,2,2) | — |
| Block 2 | Conv3D(32) × 2, ReLU, same padding | 32 |
| Pool 2 | MaxPooling3D(2,2,2) | — |
| Block 3 | Conv3D(64) × 2, ReLU, same padding | 64 |
| Pool 3 | MaxPooling3D(2,2,2) | — |

**Bottleneck:**
| Block 4 | Conv3D(128) × 2, ReLU | 128 |

**Decoder (expanding path) — skip connections via concatenate:**
| Block 5 | UpSampling3D(2) + skip(conv3) → Conv3D(64) × 2 | 64 |
| Block 6 | UpSampling3D(2) + skip(conv2) → Conv3D(32) × 2 | 32 |
| Block 7 | UpSampling3D(2) + skip(conv1) → Conv3D(16) × 2 | 16 |

**Output:**
| Block 8 | Conv3D(1, kernel=1×1×1, sigmoid) | 1 (fault probability) |

Accepts volumes of arbitrary spatial size (all-`None` input shape), enabling inference on field datasets larger than 128³. Returns a `keras.Model`.

#### `cross_entropy_balanced(y_true, y_pred)`

A class-balanced binary cross-entropy loss that dynamically weights positive (fault) and negative (non-fault) voxels inversely proportional to their counts:

```
beta    = count_neg / (count_neg + count_pos)
weight  = beta / (1 - beta)   # up-weights scarce fault voxels
cost    = weighted_cross_entropy(logits, targets, pos_weight=weight) * (1 - beta)
```

Returns 0 if no fault voxels are present (avoids division-by-zero). This loss is defined here but **not used in `train.py`** (which uses plain `'binary_crossentropy'`). It is required when loading checkpoints saved with this custom loss.

#### `_to_tensor(x, dtype)`

Utility: converts a numpy array or Python scalar to a TensorFlow tensor of the specified dtype.

---

### `utils.py` — DataGenerator

**Role:** A `keras.utils.Sequence` subclass that streams training/validation batches from `.dat` files on disk. Supports shuffling and on-the-fly data augmentation.

#### `DataGenerator.__init__(dpath, fpath, data_IDs, batch_size, dim, n_channels, shuffle)`

Stores paths, IDs, and hyper-parameters, then calls `on_epoch_end()` to initialise the index array.

#### `DataGenerator.__len__()`

Returns `floor(len(data_IDs) / batch_size)` — the number of batches per epoch.

#### `DataGenerator.__getitem__(index)`

Returns one batch `(X, Y)` by:
1. Slicing `self.indexes` to get the batch's sample IDs.
2. Calling `__data_generation`.

#### `DataGenerator.on_epoch_end()`

Resets and optionally shuffles `self.indexes` after every epoch.

#### `DataGenerator.__data_generation(data_IDs_temp)`

Loads one seismic volume and its fault label from disk:

1. Reads `<dpath>/<id>.dat` and `<fpath>/<id>.dat` as raw `float32` arrays.
2. Reshapes to `(n1, n2, n3)` = `(128, 128, 128)`.
3. **Normalises seismic**: zero-mean, unit-variance (`gx = (gx - mean) / std`).
4. **Transposes** both arrays: seismic data in `.dat` files is stored as `[n3][n2][n1]` (depth-last); transpose converts to `[n1][n2][n3]` (depth-first) as expected by the network.
5. **Data augmentation**: doubles the effective dataset by appending a vertically flipped copy (`np.flipud`). This means each call returns a batch of size 2, even if `batch_size=1`.
6. Returns `X` (seismic) and `Y` (fault label) each shaped `(2, 128, 128, 128, 1)`.

---

### `train.py` — Training entrypoint

**Role:** Full training pipeline. Call `python train.py` to train from scratch.

#### `main()`

Simply calls `goTrain()`.

#### `goTrain()`

Full training workflow:

1. **Sets random seeds** for reproducibility (numpy seed 12345, TF seed 1234).
2. **Configures DataGenerators**:
   - Training: `./data/train/seis/` + `./data/train/fault/`, IDs 0–199 (200 samples).
   - Validation: `./data/validation/seis/` + `./data/validation/fault/`, IDs 0–19 (20 samples).
   - `batch_size=1`, `dim=(128,128,128)`, `shuffle=True`.
3. **Builds model**: `unet(input_size=(None,None,None,1))` — accepts arbitrary spatial sizes at inference.
4. **Compiles** with `Adam(lr=1e-4)` and `binary_crossentropy` loss.
5. **Callbacks**:
   - `ModelCheckpoint`: saves `check1/fseg-{epoch:02d}.hdf5` every epoch (not best-only).
   - `TrainValTensorBoard`: custom callback that splits train and validation metrics into separate TensorBoard subdirectories (`log1/training/`, `log1/validation/`).
6. **Trains** with `model.fit_generator` for 100 epochs.
7. Saves final model as `check1/fseg.hdf5`.
8. Calls `showHistory(history)` to plot accuracy and loss curves.

#### `showHistory(history)`

Produces two matplotlib plots: model accuracy and model loss (train vs. validation) over epochs.

#### `TrainValTensorBoard` (class, extends `TensorBoard`)

Custom TensorBoard callback that routes training metrics and validation metrics to separate log directories, so they can be overlaid on the same TensorBoard chart. Also logs the current learning rate at each epoch end.

---

### `apply.py` — Inference script

**Role:** Loads a trained checkpoint and runs inference on synthetic or field data. Called after training.

**At module import time**, the script immediately loads a model:
```python
model = load_model('check/fseg-70.hdf5',
                   custom_objects={'cross_entropy_balanced': cross_entropy_balanced})
```
This means the model is loaded as a side effect of importing `apply.py`.

#### `main()`

Calls `goValidTest()` and `goF3Test()` in sequence.

#### `goTrainTest()`

Loads training sample #100, runs inference, and plots a 2D slice comparison (seismic | true fault | predicted fault). Not called by default in `main()`.

#### `goValidTest()`

Loads validation sample #2 (128³), normalises, runs inference, and plots three 2D slice orientations (time, inline, xline).

#### `goF3Test()`

Loads the Netherlands F3 field dataset (`data/prediction/f3d/gxl.dat`, shape 128×384×512). Normalises, runs inference, plots slices, and saves the predicted fault volume back to disk as `data/prediction/f3d/fp.dat`.

#### `plot2d(gx, fx, fp, at=1, png=None)`

Renders a three-panel matplotlib figure: seismic | true/reference fault | predicted fault. Optionally saves to `./png/<png>.png`.

---

### `predNew.ipynb` — Interactive prediction notebook

**Role:** Newer, self-contained Jupyter notebook for inference (updated 2020-05-07). Uses models from `model/fseg-<epoch>.hdf5`.

**Workflow:**
1. Disables GPU (`CUDA_VISIBLE_DEVICES=""`).
2. Loads a pre-trained model at epoch 60 using `cross_entropy_balanced` as a custom object.
3. **Validation demo**: runs inference on validation sample #2, displays 6 panels (3 orientations × seismic + predicted fault).
4. **F3 field data**: runs inference on the full F3 volume, saves the fault probability cube to `data/prediction/f3d/fp.dat`, and visualises 3 slice orientations.

---

### `cpurun` — CPU execution wrapper

```sh
CUDA_VISIBLE_DEVICES='' python $*
```

Forces all TensorFlow operations onto CPU by hiding GPUs. Usage: `./cpurun train.py` or `./cpurun apply.py`.

---

### `model/model3.json`

JSON export of a trained model architecture (a `Model` with input shape `[null, 128, 128, 128, 1]`). Useful for inspecting the network structure without loading HDF5 weights. The actual weights must be downloaded separately.

---

## Execution Flow

### Training flow

```
python train.py
  └─ main()
       └─ goTrain()
            ├─ DataGenerator(train)   ← streams 200 × (128³) .dat pairs
            ├─ DataGenerator(valid)   ← streams 20  × (128³) .dat pairs
            ├─ unet(input_size=(None,None,None,1))
            │    └─ unet3.py: 3D U-Net (encoder/bottleneck/decoder)
            ├─ model.fit_generator(100 epochs)
            │    ├─ DataGenerator.__getitem__()
            │    │    └─ __data_generation(): load → normalise → transpose → augment (flipud)
            │    ├─ ModelCheckpoint → check1/fseg-{epoch}.hdf5
            │    └─ TrainValTensorBoard → log1/{training,validation}/
            └─ showHistory() → matplotlib plots
```

### Inference flow

```
python apply.py   (or predNew.ipynb)
  └─ load_model('check/fseg-70.hdf5')
       └─ main()
            ├─ goValidTest()
            │    ├─ load .dat seismic + fault from data/validation/
            │    ├─ normalise (zero-mean, unit-variance)
            │    ├─ transpose → (128,128,128,1)
            │    ├─ model.predict(reshape to (1,128,128,128,1))
            │    └─ plot2d() → 3 orientations
            └─ goF3Test()
                 ├─ load gxl.dat (128×384×512)
                 ├─ normalise + transpose
                 ├─ model.predict
                 ├─ fp.tofile("data/prediction/f3d/fp.dat")
                 └─ plot2d() → 3 orientations
```

---

## How To Run The Repository

### Prerequisites

- Python 3.6–3.7 (TensorFlow 1.x / Keras 2.x ecosystem)
- Key packages: `tensorflow==1.x`, `keras`, `numpy`, `scikit-image`, `matplotlib`
- GPU optional; use `cpurun` wrapper for CPU execution

> **Note:** `train.py` uses `from tensorflow import set_random_seed` and `model.fit_generator`, both TF1/Keras2 APIs. TF2 requires compatibility shims or minor edits.

### 1. Download data and pre-trained models

| Asset | URL |
|---|---|
| Training + validation data | [Google Drive](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8) |
| Pre-trained model weights | [Google Drive](https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd) |

**Download pre-trained models using `gdown`:**

```bash
# Install gdown if not already available
uv add gdown   # or: pip install gdown

# Create the target directory first
mkdir -p /data/faultSeg_model

# Download all model files
uv run gdown --folder 'https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd' -O /data/faultSeg_model/
```

Files will be placed under `/data/faultSeg_model/model/` (gdown creates a `model/` subdirectory matching the folder name on Drive).

gdown will create a `model/` subdirectory inside the target and download these files:

| File | Size | Description |
|---|---|---|
| `fseg-60.hdf5` | ~17 MB | Checkpoint at epoch 60 |
| `fseg-65.hdf5` | ~17 MB | Checkpoint at epoch 65 |
| `fseg-70.hdf5` | ~17 MB | Checkpoint at epoch 70 |
| `pretrained_model.hdf5` | ~108 MB | Full pretrained model |
| `model3.json` | ~13 KB | Model architecture (no weights) |

Place `.hdf5` model files under `model/` or `check/` relative to the repo root, or update the load path in the notebook/script accordingly. Place data under `data/train/`, `data/validation/`, and field data under `data/prediction/`.

### 2. Train a new model (evidence-based, unverified here)

```bash
python train.py
# or, to force CPU:
./cpurun train.py
```

Checkpoints are written to `check1/fseg-{epoch:02d}.hdf5`. Logs go to `log1/`.

Monitor with TensorBoard:
```bash
tensorboard --logdir log1/
```

### 3. Run inference on synthetic validation data

Edit `apply.py` to point to the correct checkpoint path, then:

```bash
python apply.py
```

Output PNGs are saved to `./png/`.

### 4. Interactive prediction (recommended)

Open `predNew.ipynb` in Jupyter, set the model epoch (`md = 60`), and run all cells:

```bash
jupyter notebook predNew.ipynb
```

This loads the pre-trained model, runs inference on validation and F3 field data, and displays results inline. The F3 fault probability cube is also saved to disk as `fp.dat`.

### 5. Apply to custom field data

Adapt `goF3Test()` in `apply.py` or add a new cell in `predNew.ipynb`:
1. Load your seismic cube as `float32`, shape `(n3, n2, n1)`.
2. Normalise: `gx = (gx - mean) / std`.
3. Transpose: `gx = np.transpose(gx)` → shape `(n1, n2, n3)`.
4. Predict: `fp = model.predict(gx.reshape(1, n1, n2, n3, 1))[0,:,:,:,0]`.
5. Transpose result back and save: `np.transpose(fp).tofile('output.dat')`.

---

## Important Constraints or Gaps

- **TensorFlow 1.x only** as written: uses deprecated APIs (`set_random_seed`, `fit_generator`, `tf.Summary`). Migration to TF2 requires minor but non-trivial edits.
- **Data not included in the repo**: all `.dat` training/validation files and `.hdf5` model weights must be downloaded from Google Drive separately.
- **`apply.py` loads a model at import time** (side effect), which will fail if the checkpoint file doesn't exist at the hardcoded path.
- **`check1/` directory** must exist before training or `ModelCheckpoint` will error. Create it manually: `mkdir -p check1`.
- **Batch size quirk**: `DataGenerator.__data_generation` always returns 2 samples (original + flipped) regardless of `batch_size`. With `batch_size=1`, each "batch" is actually size 2.
- **`cross_entropy_balanced`** is defined but not actually used in the default training script (plain `binary_crossentropy` is used instead). It is however required as a `custom_object` when loading older checkpoints.
- The `model/model3.json` file stores only the architecture (no weights). It requires the corresponding `.hdf5` weight file to perform inference.

---

## Bottom Line

FaultSeg3D is a focused, minimal codebase (~300 lines across 4 Python files) that implements a complete 3D seismic fault segmentation pipeline:

- **`unet3.py`** defines the 3D U-Net and custom balanced loss.
- **`utils.py`** provides a streaming data generator for raw binary seismic volumes.
- **`train.py`** orchestrates training with checkpointing and TensorBoard logging.
- **`apply.py`** and **`predNew.ipynb`** demonstrate inference on both synthetic and real field data.

The entire workflow centres on raw `float32` binary arrays (`.dat` files) in `[n3][n2][n1]` order that are transposed to `[n1][n2][n3]` at load time. The network accepts variable-size volumes, making it straightforward to apply to arbitrarily sized field datasets.
