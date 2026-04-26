---
name: facies_net
description: "Use when you need a repository-specific analysis of facies_net: summarize its purpose, explain each important script/module/function, describe execution flow, and show how to run it. facies_net trains and classifies 3D seismic facies using 3D convolutional neural networks on SEGY input data, with full support for data augmentation, TensorBoard logging, and pre-trained models."
---

# facies_net Repository Analysis

## Purpose

`facies_net` is a Python library and runnable script for **seismic facies classification** using deep 3D convolutional neural networks (CNNs). It is a direct evolution of the MalenoV project by the same author (Charles Rutherford Ildstad, ConocoPhillips / University of Trondheim, 2017).

Given one or more 3D SEGY seismic cubes and either a set of annotated training-point `.pts` files (one per class) or a labelled SEGY facies cube, facies_net:

1. Reads the SEGY volumes into memory as normalised numpy arrays.
2. Generates 3D mini-cubes ("voxelets") on-the-fly using a Keras `Sequence` generator.
3. Trains a 4-layer 3D CNN (Keras/TensorFlow) to distinguish the provided facies classes.
4. Logs training to TensorBoard.
5. Classifies a user-defined sub-section of the seismic cube voxel-by-voxel.
6. Writes the resulting facies prediction back out as a SEGY, `.npy`, and `.ixz` (CSV) file.

Compared to MalenoV, key improvements include:
- **Modular, multi-file architecture** instead of a single script.
- **Keras `Sequence` generator** (proper generator-based training instead of batch loops).
- **True data augmentation** (Mirror1/2/3, Transpose, and combinations) that is wired and functional.
- **Dual training input modes**: `.pts` annotation files OR a labelled SEGY facies cube.
- **TensorBoard integration** for training monitoring.
- **Multiple pre-trained models** included in the `F3/` folder for the Dutch F3 dataset.
- **Attribution analysis** (Integrated Gradients) and **feature visualisation** scripts for model interpretability.

---

## Verified Repository Summary

| Item | Detail |
|---|---|
| Language | Python 3 |
| Entry point | `facies_net.py` |
| Module folder | `facies_net_func/` (9 Python modules) |
| Pre-trained models | `F3/` (21 × `.h5` files for Dutch F3 dataset) |
| Training annotation formats | `.pts` text files (inline, xline, time) OR labelled SEGY cube |
| License | GNU LGPL v3.0 |
| Key dependencies | `segyio`, `keras`, `tensorflow`, `numpy`, `matplotlib`, `scipy` |
| CNN architecture | 4 × `Conv3D` → `Dense(10)` → `Dense(num_classes)` → softmax |
| Default voxelet size | 61 × 61 × 61 (controlled by `cube_incr=30`) |
| Optimised for | TensorFlow 1.5.0 |

---

## Repository Walkthrough

### Top-level structure

```
facies_net.py               ← Main entry point (run this)
facies_net_func/            ← All supporting modules
    masterf.py              ← Top-level orchestrator (master function)
    segy_files.py           ← SEGY I/O (segy_reader, segy_decomp, segy_adder)
    training.py             ← Model training (train_model, adaptive_lr)
    data_cond.py            ← Data conditioning (convert, convert_segy, ex_create generator)
    modelling.py            ← CNN architecture (make_model, SpatialDropout3D)
    prediction.py           ← Batch prediction (cube_parse, predicting, makeIntermediate)
    visualize.py            ← Plotting and prediction orchestration (visualization, show_details)
    attribution.py          ← Integrated Gradients (integrated_gradients class)
    feature_vis.py          ← Feature maximisation visualisation (features, save_image)
F3/                         ← Pre-trained Keras .h5 models for Dutch F3 dataset (21 variants)
class_addresses/            ← Pre-annotated .pts files for Dutch F3 multi-facies classification
    multi_*.pts             ← 9 facies class address files
    address_maker.py        ← Utility to generate address lists from a SEGY
    pixels_to_points.py     ← Convert pixel coordinates to seismic points
    points_to_pixels.py     ← Convert seismic points to pixel coordinates
attribution_analysis.py     ← Standalone script: run Integrated Gradients on a trained model
feature_analysis.py         ← Standalone script: visualise CNN filter responses
size_check.py               ← Utility: print layer input/output shapes for a trained model
logs/                       ← TensorBoard log output directory
predictions/                ← Output directory for saved prediction files
images/                     ← Output directory for attribution/feature visualisation images
```

---

### `facies_net.py` — Main entry point

**Role:** Configure and launch training runs. Not a library — it is a run script.

The script sets up `filenames`, `cube_incr`, `train_dict`, and `pred_dict`, then calls `master()` once per augmentation variant. The default run trains on `F3_entire.segy` using `F3_facies.segy` as the labelled training cube, iterating through all 15 data augmentation combinations (baseline + 7 single + 6 double + 1 triple + 1 full).

Key configuration:
- `cube_incr = 30` → voxelet size 61×61×61
- `num_train_ex = 80000` per epoch
- `epochs = 10`, `batch_size = 32`, `val_split = 0.3`
- `data_augmentation`: list of strings from `['Mirror1', 'Mirror2', 'Mirror3', 'Transpose']`
- Training input: `'files': ['F3_facies.segy']` (single SEGY label cube mode)

---

### `facies_net_func/masterf.py` — `master()`

**Role:** Top-level orchestrator. Single public function.

```python
def master(segy_filename, cube_incr, train_dict={}, pred_dict={}, mode='full')
```

| `mode` | Action |
|---|---|
| `'train'` | Calls `segy_reader` → `train_model`, times and prints duration |
| `'predict'` | Loads pre-trained model from `pred_dict['keras_model']`, calls `visualization` |
| `'full'` | Trains then immediately predicts |

**Returns:** `{'model': trained_model, 'pred': prediction_array}`

Notable: if `'keras_model'` is present in `pred_dict`, training will **continue from that model** rather than building a new one (fine-tuning).

---

### `facies_net_func/segy_files.py` — SEGY I/O

#### `segy_reader(segy_filenames)`

**Role:** Entry point for SEGY loading. Accepts a single filename string or a list of filenames.

- Single filename: calls `segy_decomp`, adds a channel dimension → shape `(xlines, inlines, samples, 1)`.
- List of filenames: stacks multiple SEGY cubes as channels → shape `(xlines, inlines, samples, N)`.

**Returns:** A `segyio.spec` object augmented with `.inl_start/end/step`, `.xl_start/end/step`, `.t_start/end/step`, `.data`, `.cube_num`.

#### `segy_decomp(segy_file)`

**Role:** Opens a SEGY with `segyio`, reads the entire 3D volume via `segyio.tools.cube()`, normalises amplitudes to −127…+127.

**Returns:** Augmented `segyio.spec` object.

#### `segy_adder(segy_file, inp_cube)`

**Role:** Reads a second SEGY cube and returns it as a plain normalised numpy array for stacking as an additional channel. Called by `segy_reader` for multi-cube inputs.

---

### `facies_net_func/modelling.py` — CNN Architecture

#### `SpatialDropout3D`

A custom Keras `Dropout` subclass implementing spatial dropout for 5D tensors (drops entire 3D feature maps). More effective than element-wise dropout for convolutional layers with correlated activations.

#### `make_model(cube_size=65, num_channels=1, num_classes=2, opt=adam(lr=0.001))`

**Role:** Builds the 3D CNN from scratch.

**Architecture:**

```
Input: (cube_size, cube_size, cube_size, num_channels)
  Conv3D(50, 5×5×5, stride=4, padding='valid')   ← conv_layer1
  BatchNorm → SpatialDropout3D(0.2) → ReLU

  Conv3D(50, 3×3×3, stride=2, padding='valid')   ← conv_layer2
  BatchNorm → SpatialDropout3D(0.2) → ReLU

  Conv3D(50, 3×3×3, stride=2, padding='valid')   ← conv_layer3
  BatchNorm → SpatialDropout3D(0.2) → ReLU

  Conv3D(50, 3×3×3, stride=1, padding='valid')   ← conv_layer4
  BatchNorm → SpatialDropout3D(0.2) → ReLU

  Dense(10)                                        ← attribute_layer (embedding)
  BatchNorm → Dropout(0.2) → ReLU

  Dense(num_classes)                               ← pre-softmax_layer
  BatchNorm → Softmax → Flatten
```

**Compiled with:** `categorical_crossentropy`, Adam, metric `accuracy`.

Note: `dense_layer1` (Dense(50)) is present but commented out.

---

### `facies_net_func/data_cond.py` — Data Conditioning

#### `convert(file_list, save, savename, ex_adjust, val_split)`

**Role:** Reads `.pts` annotation files (inline, xline, time columns), appends class-index column, splits into training/validation, optionally oversamples minority classes.

**Input:** List of `.pts` file paths, one per facies class.
**Output:** `(tr_list, val_list)` — each `(N, 4)` numpy arrays: inline, xline, time, class_index.

#### `convert_segy(segy_name, save, savename, ex_adjust, val_split, mode)`

**Role:** Alternative to `convert` when training data is a **labelled SEGY facies cube** rather than `.pts` files. Extracts one central slice (inline or xline) from the label cube and converts it to an address array.

- `mode='xline'`: uses the middle xline of the label cube for training.
- `mode='iline'`: uses the middle inline for validation.

This is the mode used in the default `facies_net.py` configuration.

#### `class ex_create(keras.utils.Sequence)` — Training Data Generator

**Role:** On-demand 3D voxelet generator implementing the Keras `Sequence` interface (supports multi-worker parallel loading).

Key methods:
- `__init__`: stores seismic array, address list, cube size, augmentation settings, and computes buffer-zone boundaries (addresses too close to cube edges are illegal).
- `__len__`: returns number of batches per epoch (`steps`).
- `__getitem__(index)`: returns batch at position `index` → calls `data_generation`.
- `data_generation(index_start)`: slices `(cube_size, cube_size, cube_size, channels)` voxelets from the seismic array for each address in the batch, applies data augmentation, returns `(examples, one_hot_labels)`.

**Data augmentation operations** (applied stochastically at 50% probability):
| Name | Operation |
|---|---|
| `Mirror1` | Flip axis 0 (inline direction) |
| `Mirror2` | Flip axis 1 (xline direction) |
| `Mirror3` | Flip axis 2 (depth direction) |
| `Transpose` | Swap axes 0 and 1 (inline ↔ xline) |
| `Mirror1T`, `Mirror2T`, `Mirror12T` | Combined mirror + transpose |

Illegal addresses (outside buffer zone) are silently removed from the address list; a warning is printed if more than 10% are illegal.

---

### `facies_net_func/training.py` — Training

#### `adaptive_lr(input_int)`

Returns `0.1 ** epoch` — exponential decay. Used as a Keras `LearningRateScheduler` callback.

#### `train_model(segy_obj, file_list, cube_incr, num_epochs, num_classes, num_examples, batch_size, val_split, opt_patience, data_augmentation, keras_model, write_out, write_location)`

**Role:** Core training function.

1. Calculates `cube_size`, `num_channels`, training/validation step counts.
2. Builds a fresh model via `make_model()` or reuses `keras_model` if provided.
3. Creates address lists via `convert_segy()` (single SEGY label) or `convert()` (`.pts` files).
4. Instantiates `ex_create` generators for training and validation.
5. Defines callbacks: `EarlyStopping(monitor='acc', patience=opt_patience)`, `LearningRateScheduler`, `TensorBoard`.
6. Calls `model.fit_generator()` with class weights `{0:1, 1:5, 2:1}` (hardcoded — gives extra weight to class 1).
7. Saves model as `.h5` if `write_out=True`.

**TensorBoard logs** are saved to `./logs/<write_location>`.

---

### `facies_net_func/prediction.py` — Prediction

#### `cube_parse(seis_arr, cube_incr, inline_num, xline_num, depth)`

**Role:** Extracts a single `(1, cube_size, cube_size, cube_size, channels)` voxelet from the seismic array at a specific (inline, xline, depth) index. Returns the raw numpy array ready for `model.predict`.

#### `makeIntermediate(keras_model, layer_name)`

**Role:** Creates a new Keras `Model` that outputs the activation of a named intermediate layer. Used for feature extraction (e.g. `attribute_layer`).

#### `predicting(filename, seis_obj, keras_model, cube_incr, num_classes, section, print_segy, savename, pred_batch, show_features, layer_name)`

**Role:** The main inference engine.

- Iterates all (inline, xline) positions within `section` bounds in batches of `pred_batch` traces.
- For each position calls `cube_parse` for all depth samples.
- Calls `keras_model.predict_classes()` (class mode) or `intermediate_layer_model.predict()` (feature mode).
- Accumulates results into a `(inl, xl, z, num_classes)` prediction array.
- If `print_segy=True`: saves `.npy`, `.sgy` (SEGY copy with prediction values), and prints completion with time estimates.

**`section` parameter** is a 6-element index array: `[inl_min_idx, inl_max_idx, xl_min_idx, xl_max_idx, z_min_idx, z_max_idx]`.

---

### `facies_net_func/visualize.py` — Visualisation

#### `visualization(filename, seis_obj, keras_model, cube_incr, section_edge, xline_ref, num_classes, sect_form, save_pred, save_file, pred_batch, show_feature)`

**Role:** Converts `section_edge` from SEGY coordinates to array indices (if `sect_form='segy'`), calls `predicting`, then produces a two-panel matplotlib figure: reference xline amplitude on the left, classification/probability on the right.

**Returns:** prediction array.

#### `plotNNpred(pred, im_per_line, line_num, section)`

**Role:** Visualises all 10 feature channels from the `attribute_layer` as individual 2D colour images.

#### `show_details(filename, cube_incr, predic, inline, inl_start, xline, xl_start, slice_number, slice_incr, inp_format, show_prob, num_classes)`

**Role:** QC detail plotter. Produces three matplotlib figures:
1. Inline sections: seismic amplitude ± 3 lines around reference inline.
2. Xline sections: seismic amplitude ± 3 lines around reference xline.
3. Depth slices: seismic amplitude and prediction at ±`slice_incr` around `slice_number`.

---

### `facies_net_func/attribution.py` — `integrated_gradients`

**Role:** Keras-compatible implementation of **Integrated Gradients** (Sundararajan et al., 2017) for explaining CNN predictions via Shapley-like attribution.

- `__init__`: wraps a Keras Sequential or Model; builds gradient computation functions for each output class.
- `explain(inp, reference, ind, num_steps)`: interpolates `num_steps` inputs between `reference` (baseline) and `inp`, averages the gradients — gives per-input-feature importance scores.

Used by `attribution_analysis.py` to produce overlay visualisations of which seismic features drive a given classification.

---

### `facies_net_func/feature_vis.py` — Feature Visualisation

**Role:** Implements **activation maximisation** — finds input images that maximally activate a given CNN filter via gradient ascent.

Key functions:
- `deprocess_image(x)`: converts a gradient-ascent result tensor to a displayable uint8 image.
- `normalize(x)`: L2 normalisation of a Keras tensor.
- `smoothing(im, mode)`: regularisation options for gradient ascent (`L2`, `GaussianBlur`, `Decay`, `Clip_weak`).
- `save_image(kept_filters, keras_model, name)`: saves filter visualisations.
- `save_or(im, name, formatting)`: saves a raw input image (original or normalised).
- `save_overlay(ig, num_classes, test_im, name, steps, mosaic)`: saves attribution overlays.
- `features(keras_model, layer_name, iterations, smoothing_par, inp_im, name)`: main activation maximisation loop — for each filter in the named layer, runs gradient ascent for `iterations` steps, saves resulting images.

---

### `attribution_analysis.py` — Standalone Attribution Script

**Role:** Loads a pre-trained model, extracts one training example, runs Integrated Gradients, and saves overlay images to `images/image<idx>/`.

Configure: `keras_model`, `segy_filename`, `file_list`, `im_idx`.

---

### `feature_analysis.py` — Standalone Feature Visualisation Script

**Role:** Loads a pre-trained model and runs activation maximisation on all 5 named layers (`conv_layer1`–`conv_layer4`, `attribute_layer`, `pre-softmax_layer`), saving filter visualisation images.

---

### `size_check.py` — Layer Shape Utility

**Role:** Prints input and output shapes for each named layer of a loaded model. Useful for debugging architecture changes.

---

### `class_addresses/` — Pre-annotated Training Data

Contains 9 `.pts` files for multi-facies classification on the Dutch F3 dataset:
- `multi_else_ilxl.pts`, `multi_grizzly_ilxl.pts`, `multi_high_amp_continuous_ilxl.pts`
- `multi_high_amplitude_ilxl.pts`, `multi_low_amp_dips_ilxl.pts`, `multi_low_amplitude_ilxl.pts`
- `multi_low_coherency_ilxl.pts`, `multi_salt_ilxl.pts`, `multi_steep_dips_ilxl.pts`

Also contains:
- `address_maker.py`: generate address lists programmatically from a SEGY.
- `pixels_to_points.py` / `points_to_pixels.py`: coordinate conversion utilities.

---

### `F3/` — Pre-trained Models

21 `.h5` Keras model files for the Dutch F3 dataset, representing all combinations of data augmentation:
- `10_epochs_80000_examples.h5` — baseline (no augmentation)
- `10_epochs_80000_examples_mirror1.h5`, `_mirror2.h5`, `_mirror3.h5`, `_T.h5` — single augmentation
- All pairwise and triple combinations
- `10_epochs_80000_examples_mirror1_2_3_T.h5` — full augmentation

These are **ready-to-use for inference** on the Dutch F3 dataset without any training.

---

## Execution Flow

```
facies_net.py
└─ master(mode='train')
    ├─ segy_reader()
    │   ├─ segy_decomp()       ← load & normalise first SEGY cube
    │   └─ segy_adder()        ← (optional) stack additional channel cubes
    │
    └─ train_model()
        ├─ make_model()        ← build 4-layer 3D CNN (or reuse existing)
        ├─ convert_segy()      ← extract address list from label SEGY
        │   OR convert()       ← extract address list from .pts files
        ├─ ex_create()         ← instantiate Keras Sequence generators (tr + val)
        └─ model.fit_generator()
            └─ [each batch]
                └─ ex_create.data_generation()
                    ├─ slice voxelet from seismic array
                    └─ apply data augmentation

master(mode='predict')
└─ visualization()
    └─ predicting()
        └─ [each (inline, xline) in section]
            ├─ cube_parse()    ← extract one voxelet
            └─ model.predict_classes()
    └─ matplotlib plots
    └─ [optional] write .npy / .sgy
```

---

## How To Run The Repository

> **Note:** These instructions are evidence-based from `facies_net.py` and the README. The codebase targets **TensorFlow 1.5.0 / Keras 2.x**. It will not run as-is on TF2/Keras 3 without API changes.

### Prerequisites

```bash
pip install segyio keras tensorflow==1.5.0 numpy matplotlib scipy
```

### Prepare inputs

**Option A — Using a labelled SEGY facies cube (default in facies_net.py):**
1. Place your seismic SEGY as `F3_entire.segy` in the repository root.
2. Place your labelled facies SEGY as `F3_facies.segy` in the repository root.

**Option B — Using `.pts` annotation files:**
1. Place your seismic SEGY(s) in the repository root.
2. Create one `.pts` file per facies class (plain text: `inline xline time`, one row per annotated location).
3. Update `train_dict['files']` in `facies_net.py` to list all `.pts` files.

### Run training

```bash
cd /path/to/facies_net
python facies_net.py
```

Trained models are saved to `F3/<save_location>.h5`. TensorBoard logs go to `logs/<save_location>/`.

### Monitor training with TensorBoard

```bash
tensorboard --logdir=logs/F3
# then open http://localhost:6006
```

### Run inference with a pre-trained model

Edit `facies_net.py` to use `mode='predict'` and load a pre-trained model:

```python
pred_dict = {
    'keras_model': keras.models.load_model('F3/10_epochs_80000_examples.h5'),
    'section_edge': np.asarray([150, 700, 350, 1200, 150, 1700]),
    'show_feature': False,
    'xline': 775,
    'num_class': 2,
    'cord_syst': 'segy',
    'save_pred': True,
    'save_location': 'predictions/F3_pred',
    'pred_batch': 1
}

output_dict = master(
    segy_filename=['F3_entire.segy'],
    cube_incr=30,
    train_dict={},
    pred_dict=pred_dict,
    mode='predict'
)
```

Predictions are saved to `predictions/<save_location>.npy` and `.sgy`.

### Run attribution analysis

```bash
python attribution_analysis.py
# saves overlay images to images/image<idx>/
```

### Run feature visualisation

```bash
python feature_analysis.py
# saves filter maximisation images to images/image<idx>/
```

### Reload saved results

```python
import numpy as np, keras
prediction = np.load('predictions/F3_pred.npy')
model = keras.models.load_model('F3/10_epochs_80000_examples.h5')
```

---

## Important Constraints or Gaps

| Issue | Detail |
|---|---|
| **Old Keras/TF API** | Uses `keras.optimizers.adam(lr=...)`, `from keras.layers.normalization import BatchNormalization`, `model.fit_generator()`, `model.predict_classes()` — all deprecated/removed in TF2/Keras 3. |
| **Hardcoded class weights** | `{0:1, 1:5, 2:1}` is hardcoded in `train_model`. Will be wrong for datasets with a different number of classes or different class importance. |
| **`convert_segy` train/val split asymmetry** | In `train_model`, `convert_segy` is called twice — once with `mode='xline'` for training and once with `mode='iline'` for validation. The `val_split=0` parameter means the split logic inside `convert_segy` is not used; instead the two modes provide different data. This is an unusual design that may cause data leakage if the chosen inline and xline overlap. |
| **`scipy.misc.imsave` deprecated** | `attribution.py` and `feature_vis.py` use `scipy.misc.imsave`, removed in SciPy 1.3. Use `imageio.imwrite` instead. |
| **No `requirements.txt`** | Dependencies must be installed manually. |
| **Memory requirements** | A 61×61×61 float32 voxelet ≈ 0.9 MB. 80,000 examples ≈ 72 GB RAM. In practice the generator streams batches so full dataset need not fit in RAM simultaneously, but the full SEGY cube does need to fit. |
| **No tests** | No test files, framework, or CI configuration present. |
| **`F3_entire.segy` and `F3_facies.segy` not included** | Must be sourced externally (Google Drive link in MalenoV docs). |

---

## Bottom Line

`facies_net` is a **well-structured, modular Python package** for end-to-end seismic facies classification. It improves significantly over MalenoV by splitting functionality into clean modules, using a proper Keras `Sequence` generator, implementing functional data augmentation, and including TensorBoard logging. **Pre-trained `.h5` models are bundled** for the Dutch F3 dataset, enabling immediate inference without training. The `master()` function in `facies_net_func/masterf.py` is the single entry point; `facies_net.py` is the run configuration script. The codebase targets TF1/Keras 2.2 and will need API updates for modern environments.
