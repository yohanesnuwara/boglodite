---
name: malenov
description: "Use when you need a repository-specific analysis of MalenoV (MAchine LEarNing Of Voxels): summarize its purpose, explain each important function, describe execution flow, and show how to run it. MalenoV trains and classifies 3D/nD seismic facies using 3D convolutional neural networks on SEGY input data."
---

# MalenoV Repository Analysis

## Purpose

MalenoV (**MA**chine **LE**ar**N**ing **O**f **V**oxels) is a Python library and runnable script for seismic facies classification using deep 3D convolutional neural networks.

Given one or more standard 3D SEGY seismic cubes and a set of user-annotated training-point files (one file per facies class), MalenoV:

1. Reads the SEGY volumes into memory as numpy arrays.
2. Extracts 3D mini-cubes ("voxelets") centred on each annotated training location.
3. Trains a 5-layer 3D CNN (Keras/TensorFlow) to distinguish the provided facies classes.
4. Classifies the entire seismic cube (or a user-defined sub-section) voxel-by-voxel.
5. Writes the resulting facies prediction back out as a SEGY cube (plus `.npy` and `.ixz` CSV files).

The tool was created by Charles Rutherford Ildstad at ConocoPhillips (2017) and supports an unlimited number of simultaneous input seismic volumes (offset stacks, 4D data, attributes, etc.), with memory and time scaling linearly with the number of channels.

---

## Verified Repository Summary

| Item | Detail |
|---|---|
| Language | Python (no `.py` extension on the source file) |
| Single source file | `MalenoV Code 5 layer CNN 65x65x65 voxels` |
| License | GNU LGPL v3.0 |
| Key dependencies | `segyio`, `keras` (TF1-era API), `numpy`, `matplotlib` |
| No packaging files | No `requirements.txt`, `setup.py`, or `pyproject.toml` present |
| CNN architecture | 5 × `Conv3D` → 2 × `Dense` → softmax |
| Default voxelet size | 65 × 65 × 65 (controlled by `cube_incr=32`) |
| Training annotation format | Plain-text `.pts` files: one row per point, columns = inline, xline, time |

---

## Repository Walkthrough

The entire codebase lives in one script file. It is organised into five conceptual sections, each introduced by a block comment:

1. **Input data (SEGY) formatting and reading** — `segy_decomp`, `segy_adder`, `csv_struct`
2. **Data augmentation** — `randomRotationXY`, `randomRotationZ`, `randomStretch`, `randomFlip` (stub, not wired into training)
3. **Training** — `convert`, `ex_create`, `adaptive_lr`, `train_model`
4. **Prediction** — `cube_parse`, `makeIntermediate`, `predicting`
5. **Visualisation** — `plotNNpred`, `visualization`, `show_details`
6. **Master / main** — `master`, and a concrete run example at the bottom of the file

---

### `segy_decomp(segy_file, plot_data, read_direc, inp_res)` — SEGY → numpy

**Role:** Opens a SEGY file with `segyio`, memory-maps it, reads the full cube line by line, normalises amplitude values to the range −127…+127, and returns a `segyio.spec` object whose `.data` attribute is the 3-D numpy array `(xlines, inlines, samples)`.

**Key parameters:**
- `read_direc`: `'inline'` | `'xline'` | `'full'` — controls the fast read direction.
- `inp_res`: numpy dtype for the output array (e.g. `np.float32`, `np.int8`).

**Side effects:** Prints progress messages. Optionally plots xline 100 with matplotlib.

**Returns:** A `segyio.spec` object augmented with `.inl_start/end/step`, `.xl_start/end/step`, `.t_start/end/step`, and `.data`.

---

### `segy_adder(segy_file, inp_cube, read_direc, inp_res)` — add a channel

**Role:** Reads a second (or Nth) SEGY cube, normalises it the same way as `segy_decomp`, and returns the plain numpy array (not a spec object). Called by `master` when the input is a list of SEGY filenames.

---

### `csv_struct(inp_numpy, spec_obj, section, inp_res, save, savename)` — numpy → IXZ CSV

**Role:** Iterates through the inline × xline × time dimensions of a prediction numpy array and builds a 4-column array of `[inline, xline, time, value]` rows. Optionally saves to disk as `.ixz` (a space-delimited text file). Used by `predicting` to export results in a format suitable for loading into Petrel.

---

### Data augmentation stubs (`randomRotationXY`, `randomRotationZ`, `randomStretch`, `randomFlip`)

These four functions accept TensorFlow tensors and apply random rotations, stretches, or flips. They reference `tf` directly (not imported) and are **not connected to the training loop** — the `data_augmentation=True` branch in `train_model` uses Keras `ImageDataGenerator` instead, which is also noted as broken. These functions are present for future development.

---

### `convert(file_list, save, savename, ex_adjust)` — class address list builder

**Role:** Reads one or more `.pts` annotation files (inline, xline, time rows), appends a class-index column to each, concatenates them into a single `(N, 4)` integer array, and optionally equalises class sizes by oversampling minority classes.

**Input:** List of file paths, one per facies class.  
**Output:** A numpy array `(N, 4)` — columns: inline, xline, time, class_index.

---

### `ex_create(adr_arr, seis_arr, seis_spec, num_examp, cube_incr, ...)` — mini-cube sampler

**Role:** Randomly draws `num_examp` rows from the address array, converts each (inline, xline, time) coordinate to array indices, and slices a `(cube_size, cube_size, cube_size, num_channels)` voxelet from the seismic data. Illegal addresses (too close to cube edge) are replaced with fresh random draws up to 50 attempts.

**Output:** Tuple `(examples, labels)` where `examples` has shape `(N, cube_size, cube_size, cube_size, num_channels)` and `labels` has shape `(N,)`.

---

### `adaptive_lr(input_int)` — learning rate schedule

**Role:** Returns `0.1 ** epoch`. Passed to Keras `LearningRateScheduler` callback.

---

### `train_model(segy_obj, class_array, num_classes, cube_incr, ...)` — CNN builder and trainer

**Role:** The core training function. Builds or accepts a Keras Sequential model, then runs a double loop: an outer loop over `num_bunch` batches (each batch draws a fresh `num_examples` training voxelets) and an inner Keras `model.fit` over `num_epochs` epochs.

**CNN architecture (when building fresh):**

```
Input: (cube_size, cube_size, cube_size, num_channels)
  Conv3D(50, 5×5×5, stride 4, padding=same)  ← conv_layer1
  BatchNorm → ReLU
  Conv3D(50, 3×3×3, stride 2, padding=same)  ← conv_layer2
  Dropout(0.2) → BatchNorm → ReLU
  Conv3D(50, 3×3×3, stride 2)                ← conv_layer3
  Dropout(0.2) → BatchNorm → ReLU
  Conv3D(50, 3×3×3, stride 2)                ← conv_layer4
  Dropout(0.2) → BatchNorm → ReLU
  Conv3D(50, 3×3×3, stride 2)                ← conv_layer5
  Flatten
  Dense(50)  ← dense_layer1
  BatchNorm → ReLU
  Dense(10)  ← attribute_layer   (feature embedding)
  BatchNorm → ReLU
  Dense(num_classes)  ← pre-softmax_layer
  BatchNorm → Softmax
```

**Optimizer:** Adam, initial lr=0.001, then decayed by `adaptive_lr`.  
**Loss:** `categorical_crossentropy`.  
**Callbacks:** `EarlyStopping(monitor='acc', patience=opt_patience)` + `LearningRateScheduler`.  
**Output:** Trained Keras model. Optionally saved as `.h5`.

---

### `cube_parse(seis_arr, cube_incr, inp_res, mode, padding, conc, ...)` — bulk voxelet extractor

**Role:** Converts a region of the seismic numpy array into a densely packed array of overlapping 3D mini-cubes, ready for batch model prediction.

**`mode` options:** `'full'` | `'inline'` | `'xline'` | `'trace'` | `'point'` — control which spatial extent to extract.  
**`conc=True`**: returns a flat `(N, cube_size, cube_size, cube_size, channels)` array.  
**`conc=False`**: returns a spatially-indexed `(inls, xls, zls, cube_size, cube_size, cube_size, channels)` array.

Note: there is a typo bug at line 699 (`cube_size1` instead of `cube_size`) in the `xline`/non-concatenated branch.

---

### `makeIntermediate(keras_model, layer_name)` — intermediate layer extractor

**Role:** Creates a new Keras `Model` that outputs the activations of a named intermediate layer (default: `'attribute_layer'`). Used by `predicting` when `show_features=True`.

---

### `predicting(filename, inp_seis, seis_obj, keras_model, cube_incr, ...)` — batch predictor

**Role:** The main prediction engine. Iterates over all (inline, xline) positions in the requested section in batches of `pred_batch` traces, calls `cube_parse(mode='point')` for each point, collects `pred_batch × z_samples` mini-cubes, runs `keras_model.predict` or `predict_classes`, and accumulates results into a `(inl, xl, z, num_classes)` prediction array.

**Output modes:**
- `show_features=False, show_prob=False`: integer class per voxel.
- `show_features=False, show_prob=True`: per-class probability per voxel.
- `show_features=True`: 10-dimensional feature vector (from `attribute_layer`) per voxel.

**File output (when `print_segy=True`):**
- `.npy` — raw numpy prediction array.
- `.sgy` — SEGY copy of the input, with prediction values written into the section.
- `.ixz` — CSV with inline, xline, time, predicted value columns.

---

### `plotNNpred(pred, im_per_line, line_num, section)` — feature map plotter

**Role:** Creates a matplotlib subplot grid and renders each of the 10 feature channels from the `attribute_layer` as a 2D colour image along a chosen xline.

---

### `visualization(filename, inp_seis, seis_obj, keras_model, ...)` — prediction + plot orchestrator

**Role:** Converts `section_edge` from SEGY coordinates to array indices if `sect_form='segy'`, calls `predicting`, then displays a two-panel matplotlib figure: the reference xline amplitude section on the left, the facies classification/probability on the right. Returns the prediction array.

---

### `show_details(filename, cube_incr, predic, ...)` — QC detail plotter

**Role:** Reads the reference SEGY file fresh (via `segy_decomp`), then produces three matplotlib figures showing inline sections, xline sections, and depth slices of both the seismic amplitude and the prediction, centred on a user-specified (inline, xline, depth) location.

---

### `master(segy_filename, inp_format, cube_incr, train_dict, pred_dict, mode)` — top-level orchestrator

**Role:** Entry point for a complete train-and-predict run. Accepts one or multiple SEGY filenames, assembles them into a 4-D numpy array `(inl, xl, z, channels)`, then branches on `mode`:

| `mode` | Action |
|---|---|
| `'train'` | Calls `convert` + `train_model`, times training, optionally saves model |
| `'predict'` | Loads pre-trained model from `pred_dict['keras_model']`, calls `visualization` |
| `'full'` | Trains then immediately predicts |

**Returns:** `{'model': trained_model, 'pred': prediction_array}`.

---

## Execution Flow

```
master()
├─ segy_decomp()           ← load first SEGY cube
├─ segy_adder()            ← (for each extra SEGY) append channel
│
├─ [if mode = 'train' or 'full']
│   ├─ convert()           ← read .pts files → (N,4) address array
│   └─ train_model()
│       └─ for each bunch:
│           ├─ ex_create() ← draw random 3D voxelets from seismic
│           └─ model.fit() ← Keras training with callbacks
│
└─ [if mode = 'predict' or 'full']
    └─ visualization()
        └─ predicting()
            └─ for each (inline, xline):
                ├─ cube_parse(mode='point') ← extract voxelet
                └─ model.predict()          ← classify voxelet
        └─ matplotlib plots
        └─ [optional] write .sgy / .npy / .ixz
```

---

## How To Run The Repository

> **Note:** These instructions are evidence-based from the bottom-of-file run example. The script targets the **TensorFlow 1 / Keras 2.x** era (`keras.optimizers.adam`, `BatchNormalization` from `keras.layers.normalization`). It will not run as-is with modern TF2/Keras without API changes.

### Prerequisites

```bash
pip install segyio keras tensorflow numpy matplotlib
```

Recommended versions (to match the original API calls):
- `tensorflow==1.x` or `tensorflow==2.x` with `tf.compat.v1`
- `keras==2.2.x`
- `segyio>=1.8`
- `numpy>=1.14`

### Prepare inputs

1. One or more 3D SEGY files (e.g. `near_stack.segy`, `far_stack.segy`).
2. One `.pts` file per facies class — plain text, three columns: **inline xline time**, one annotated location per row.

### Configure and run

Open `MalenoV Code 5 layer CNN 65x65x65 voxels` in a Python interpreter or Jupyter notebook, or rename it to a `.py` file and run it. Edit the bottom section to match your data:

```python
filenames = ['near_stack.segy', 'mid_stack.segy', 'far_stack.segy']
inp_res = np.float32
cube_incr = 32   # → 65×65×65 voxelets

train_dict = {
    'files': ['class1.pts', 'class2.pts', 'class3.pts'],
    'num_tot_iterations': 25,
    'epochs': 12,
    'num_train_ex': 18000,
    'batch_size': 32,
    'opt_patience': 10,
    'data_augmentation': False,
    'save_model': True,
    'save_location': 'my_model'
}

pred_dict = {
    'keras_model': keras.models.load_model('my_model.h5'),
    'section_edge': np.asarray([inl_min, inl_max, xl_min, xl_max, z_min, z_max]),
    'show_feature': False,
    'xline': xl_ref,
    'num_class': len(train_dict['files']),
    'cord_syst': 'segy',
    'save_pred': True,
    'save_location': 'output_prediction',
    'pred_batch': 25,
    'pred_prob': False
}

output_dict = master(
    segy_filename=filenames,
    inp_format=inp_res,
    cube_incr=cube_incr,
    train_dict=train_dict,
    pred_dict=pred_dict,
    mode='full'   # 'train', 'predict', or 'full'
)
```

### Load/save helpers (commented out at file bottom)

```python
# Reload a saved prediction
prediction = np.load('output_prediction.npy')

# Reload a saved model
loaded_model = keras.models.load_model('my_model.h5')
```

### Test data

The authors provide the **Poseidon** seismic dataset (Near/Mid/Far stacks, ~100 GB per SEGY file) and the smaller **Dutch F3** dataset (~1 GB) at:  
https://drive.google.com/drive/folders/0B7brcf-eGK8CRUhfRW9rSG91bW8

Pre-annotated training `.pts` files and pre-trained `.h5` models for steep dip and multi-facies classification are also provided there.

---

## Important Constraints or Gaps

| Issue | Detail |
|---|---|
| **Old Keras API** | Uses `keras.optimizers.adam(lr=...)` and `from keras.layers.normalization import BatchNormalization` — both deprecated/moved in TF2/Keras 3. |
| **Data augmentation not functional** | The `data_augmentation=True` branch uses `ImageDataGenerator` (2D, not 3D) and references `x_test` which is never defined. The separate TF-based augmentation functions reference an unimported `tf`. |
| **No packaging** | No `requirements.txt`, `setup.py`, `pyproject.toml`, or `conda` environment file. Dependencies must be installed manually. |
| **No `.py` extension** | The source file has a space-separated descriptive name; it must be renamed or imported explicitly. |
| **`cube_size1` typo** | Line 699 in `cube_parse` references `cube_size1` (undefined) in the `xline`/non-concatenated branch. |
| **Memory requirements** | A 65×65×65 voxelet with float32 ≈ 1 MB. Training 18,000 examples at once ≈ 18 GB RAM. A 100 GB SEGY cube fully loaded in float32 requires ~100 GB RAM. |
| **Single-file architecture** | Everything (I/O, model, training, prediction, visualisation) lives in one script with no module separation. |
| **No tests** | No test files, test framework, or CI configuration present. |

---

## Bottom Line

MalenoV is a **self-contained Python script** that wires together `segyio` SEGY I/O, numpy voxelet extraction, a 5-layer 3D CNN in Keras, and matplotlib visualisation into a single end-to-end seismic facies classification pipeline. The `master()` function is the single entry point; all other functions are helpers it calls. The codebase targets the TF1/Keras 2.2 era and will require minor API updates to run on modern environments. The data augmentation subsystem is present but non-functional. The tool is well-commented and readable but monolithic.
