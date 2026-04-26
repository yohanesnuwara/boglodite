"""
Multi-class seismic facies training and prediction on Dutch F3 dataset.

Trains a 4-layer 3D CNN on 9 annotated facies classes (.pts files),
then predicts inline 130 across all valid xline/time positions.

Output:
  predictions/F3_multiclass_model.h5    -- saved trained model
  predictions/F3_multi_prob.npy         -- (1, n_xl, n_z, 9) softmax probs
  predictions/F3_multi_class.npy        -- (1, n_xl, n_z) integer class labels
  predictions/F3_multi_inline.png       -- 3-panel plot
"""

import math
import os
import sys

import matplotlib
import numpy as np
import segyio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras
import tf_keras.backend as K
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import (
    Activation, BatchNormalization, Conv3D, Dense, Dropout, Flatten
)
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam

# ── Configuration ─────────────────────────────────────────────────────────────
SEGY_PATH  = "/workspace/boglodite/data/Dutch Government_F3_entire_8bit seismic.segy"
PTS_DIR    = "/workspace/boglodite/tools/facies_net/class_addresses"
OUT_DIR    = "/workspace/boglodite/outputs"
MODEL_SAVE = os.path.join(OUT_DIR, "F3_multiclass_model.h5")

CUBE_INCR   = 30       # voxelet half-size → 61×61×61
BATCH_SIZE  = 64
EPOCHS      = 15

# 9 annotated facies classes for Dutch F3
FACIES_FILES = [
    "multi_else_ilxl.pts",
    "multi_grizzly_ilxl.pts",
    "multi_high_amp_continuous_ilxl.pts",
    "multi_high_amplitude_ilxl.pts",
    "multi_low_amp_dips_ilxl.pts",
    "multi_low_amplitude_ilxl.pts",
    "multi_low_coherency_ilxl.pts",
    "multi_salt_ilxl.pts",
    "multi_steep_dips_ilxl.pts",
]
FACIES_NAMES = [
    "Else", "Grizzly", "High Amp Cont.", "High Amplitude",
    "Low Amp Dips", "Low Amplitude", "Low Coherency", "Salt", "Steep Dips",
]
NUM_CLASSES = len(FACIES_FILES)

# Inline 130 — first inline with full buffer clearance
SECTION_SEGY = np.array([130, 130, 330, 1220, 124, 1728])

os.makedirs(OUT_DIR, exist_ok=True)


# ── Custom layer ──────────────────────────────────────────────────────────────
class SpatialDropout3D(keras.layers.Dropout):
    """Drops entire 3D feature maps (spatial dropout for 5D tensors)."""

    def __init__(self, rate, data_format=None, **kwargs):
        super().__init__(rate, **kwargs)
        self.data_format = data_format or K.image_data_format()

    def _get_noise_shape(self, inputs):
        import tensorflow as tf
        s = tf.shape(inputs)
        if self.data_format == "channels_first":
            return (s[0], s[1], 1, 1, 1)
        return (s[0], 1, 1, 1, s[4])

    def get_config(self):
        cfg = super().get_config()
        cfg["data_format"] = self.data_format
        return cfg


# ── SEGY loading ──────────────────────────────────────────────────────────────
def load_segy(path):
    print(f"Loading SEGY: {path}")
    with segyio.open(path, "r", strict=False) as f:
        f.mmap()
        specs = dict(
            inl_start=int(f.ilines[0]),  inl_end=int(f.ilines[-1]),
            inl_step=int(f.ilines[1] - f.ilines[0]),
            xl_start=int(f.xlines[0]),   xl_end=int(f.xlines[-1]),
            xl_step=int(f.xlines[1] - f.xlines[0]),
            t_start=int(f.samples[0]),   t_end=int(f.samples[-1]),
            t_step=int(f.samples[1] - f.samples[0]),
        )
    print(f"  inlines : {specs['inl_start']}..{specs['inl_end']}  step {specs['inl_step']}")
    print(f"  xlines  : {specs['xl_start']}..{specs['xl_end']}  step {specs['xl_step']}")
    print(f"  time    : {specs['t_start']}..{specs['t_end']}  step {specs['t_step']} ms")

    data = segyio.tools.cube(path).astype(np.float32)
    print(f"  cube shape (inlines, xlines, samples): {data.shape}")
    amax = np.amax(np.abs(data))
    if amax > 0:
        data *= 127.0 / amax
    data = np.expand_dims(data, axis=-1)   # → (n_il, n_xl, n_z, 1)
    return data, specs


# ── .pts loading and index conversion ─────────────────────────────────────────
def load_pts_files(pts_dir, filenames, specs, data_shape):
    """
    Loads all .pts annotation files. Returns array of shape (N, 4):
    columns = [il_idx, xl_idx, t_idx, class_idx]

    Points outside the SEGY grid or buffer zone are discarded.
    """
    ni, nx, nz, _ = data_shape
    ci = CUBE_INCR
    all_addrs = []

    for cls_idx, fname in enumerate(filenames):
        path = os.path.join(pts_dir, fname)
        raw = np.loadtxt(path, usecols=[0, 1, 2])   # inline, xline, time
        n_raw = len(raw)

        # Convert SEGY coordinates to array indices
        il_idx = (raw[:, 0] - specs["inl_start"]) / specs["inl_step"]
        xl_idx = (raw[:, 1] - specs["xl_start"])  / specs["xl_step"]
        t_idx  = (raw[:, 2] - specs["t_start"])   / specs["t_step"]

        # Validate on-grid (coords must align to the SEGY sampling)
        on_grid = (
            (np.abs(il_idx - np.round(il_idx)) < 0.01) &
            (np.abs(xl_idx - np.round(xl_idx)) < 0.01) &
            (np.abs(t_idx  - np.round(t_idx))  < 0.01)
        )
        il_idx = np.round(il_idx).astype(int)
        xl_idx = np.round(xl_idx).astype(int)
        t_idx  = np.round(t_idx).astype(int)

        # Validate within bounds + buffer zone
        in_bounds = (
            (il_idx >= ci) & (il_idx < ni - ci) &
            (xl_idx >= ci) & (xl_idx < nx - ci) &
            (t_idx  >= ci) & (t_idx  < nz - ci)
        )
        valid = on_grid & in_bounds
        n_valid = valid.sum()
        n_dropped = n_raw - n_valid
        print(f"  {fname:45s}: {n_raw:6d} pts → {n_valid:6d} valid "
              f"({n_dropped} dropped)")

        cls_col = np.full(n_valid, cls_idx, dtype=int)
        chunk = np.stack([il_idx[valid], xl_idx[valid], t_idx[valid], cls_col], axis=1)
        all_addrs.append(chunk)

    addrs = np.concatenate(all_addrs, axis=0)
    print(f"\n  Total valid addresses: {len(addrs):,}  ({NUM_CLASSES} classes)")
    return addrs


def spatial_split(addrs, val_frac=0.20, seed=42):
    """
    Stratified split per class using time-axis blocks to reduce voxelet leakage.
    For each class, unique time indices are sorted, and the top val_frac fraction
    (deepest samples) go to val. Falls back to random split if a class spans only
    one time block.
    """
    rng = np.random.default_rng(seed)
    tr_list, va_list = [], []

    for c in range(NUM_CLASSES):
        cls_mask  = addrs[:, 3] == c
        cls_addrs = addrs[cls_mask]

        t_sorted = np.sort(np.unique(cls_addrs[:, 2]))
        if len(t_sorted) > 1:
            threshold  = t_sorted[int(len(t_sorted) * (1 - val_frac))]
            val_mask   = cls_addrs[:, 2] >= threshold
        else:
            # Only one unique time — fall back to random 80/20
            perm     = rng.permutation(len(cls_addrs))
            n_val    = max(1, int(len(cls_addrs) * val_frac))
            val_mask = np.zeros(len(cls_addrs), dtype=bool)
            val_mask[perm[:n_val]] = True

        tr_list.append(cls_addrs[~val_mask])
        va_list.append(cls_addrs[val_mask])

    tr = np.concatenate(tr_list, axis=0)
    va = np.concatenate(va_list, axis=0)

    print(f"\n  Time-stratified split (val = deepest {int(val_frac*100)}% per class):")
    print(f"    Train: {len(tr):,}   Val: {len(va):,}")
    for c in range(NUM_CLASSES):
        n_tr = (tr[:, 3] == c).sum()
        n_va = (va[:, 3] == c).sum()
        print(f"    class {c} ({FACIES_NAMES[c]:20s}): train={n_tr:5d}  val={n_va:5d}")
    return tr, va


def compute_class_weights(addrs):
    counts = np.bincount(addrs[:, 3], minlength=NUM_CLASSES).astype(float)
    total  = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ── Keras Sequence data generator ─────────────────────────────────────────────
class VoxeletGenerator(keras.utils.Sequence):
    """
    Yields batches of (61,61,61,1) voxelets and one-hot class labels.
    Shuffles address order after each epoch.
    """

    def __init__(self, data, addresses, batch_size, num_classes, cube_incr,
                 augment=False):
        self.data       = data
        self.addresses  = addresses.copy()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ci         = cube_incr
        self.augment    = augment
        np.random.shuffle(self.addresses)

    def __len__(self):
        return math.ceil(len(self.addresses) / self.batch_size)

    def __getitem__(self, batch_idx):
        ci   = self.ci
        cs   = 2 * ci + 1
        start = batch_idx * self.batch_size
        batch = self.addresses[start : start + self.batch_size]
        n     = len(batch)

        X = np.empty((n, cs, cs, cs, 1), dtype=np.float32)
        y = np.zeros((n, self.num_classes), dtype=np.float32)

        for i, (il, xl, tz, cls) in enumerate(batch):
            vox = self.data[il-ci:il+ci+1, xl-ci:xl+ci+1, tz-ci:tz+ci+1, :]
            if self.augment:
                if np.random.rand() < 0.5:
                    vox = vox[::-1, :, :, :]   # mirror inline axis
                if np.random.rand() < 0.5:
                    vox = vox[:, ::-1, :, :]   # mirror xline axis
            X[i]      = vox
            y[i, cls] = 1.0

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.addresses)


# ── Model architecture ─────────────────────────────────────────────────────────
def build_model(cube_size, num_channels, num_classes):
    model = Sequential([
        Conv3D(50, (5, 5, 5), padding="valid", strides=(4, 4, 4),
               input_shape=(cube_size,) * 3 + (num_channels,),
               data_format="channels_last", name="conv_layer1"),
        BatchNormalization(), SpatialDropout3D(0.2), Activation("relu"),

        Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding="valid", name="conv_layer2"),
        BatchNormalization(), SpatialDropout3D(0.2), Activation("relu"),

        Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding="valid", name="conv_layer3"),
        BatchNormalization(), SpatialDropout3D(0.2), Activation("relu"),

        Conv3D(50, (3, 3, 3), strides=(1, 1, 1), padding="valid", name="conv_layer4"),
        BatchNormalization(), SpatialDropout3D(0.2), Activation("relu"),

        Dense(10, name="attribute_layer"),
        BatchNormalization(), Dropout(0.2), Activation("relu"),

        Dense(num_classes, name="pre_softmax_layer"),
        BatchNormalization(), Activation("softmax"),
        Flatten(),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Prediction ─────────────────────────────────────────────────────────────────
def segy_to_index(section_segy, specs, data_shape):
    ci = CUBE_INCR
    ni, nx, nz, _ = data_shape

    def to_idx(val, start, step):
        return (val - start) // step

    inl_min = max(to_idx(section_segy[0], specs["inl_start"], specs["inl_step"]), ci)
    inl_max = min(to_idx(section_segy[1], specs["inl_start"], specs["inl_step"]), ni - ci - 1)
    xl_min  = max(to_idx(section_segy[2], specs["xl_start"],  specs["xl_step"]),  ci)
    xl_max  = min(to_idx(section_segy[3], specs["xl_start"],  specs["xl_step"]),  nx - ci - 1)
    t_min   = max(to_idx(section_segy[4], specs["t_start"],   specs["t_step"]),   ci)
    t_max   = min(to_idx(section_segy[5], specs["t_start"],   specs["t_step"]),   nz - ci - 1)

    section_idx = np.array([inl_min, inl_max, xl_min, xl_max, t_min, t_max])
    print(f"  SEGY section  : {section_segy.tolist()}")
    print(f"  Index section : {section_idx.tolist()}")
    return section_idx


def predict_section(data, model, section_idx, num_classes, batch_size=64):
    ci = CUBE_INCR
    cs = 2 * ci + 1
    inl_min, inl_max, xl_min, xl_max, t_min, t_max = section_idx
    n_inl = inl_max - inl_min + 1
    n_xl  = xl_max  - xl_min  + 1
    n_z   = t_max   - t_min   + 1
    total = n_inl * n_xl * n_z

    print(f"\nPredicting {n_inl} × {n_xl} × {n_z} = {total:,} voxels ...")
    probs   = np.empty((total, num_classes), dtype=np.float32)
    buf     = np.empty((batch_size, cs, cs, cs, 1), dtype=np.float32)
    idx     = 0

    for i in range(n_inl):
        il = inl_min + i
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  inline {i+1}/{n_inl} ...", flush=True)
        for x in range(n_xl):
            xl = xl_min + x
            for z in range(n_z):
                tz = t_min + z
                buf[idx % batch_size] = data[
                    il-ci:il+ci+1, xl-ci:xl+ci+1, tz-ci:tz+ci+1, :
                ]
                idx += 1
                if idx % batch_size == 0:
                    probs[idx - batch_size : idx] = model.predict(buf, verbose=0)

    rem = idx % batch_size
    if rem > 0:
        probs[idx - rem : idx] = model.predict(buf[:rem], verbose=0)

    return probs.reshape(n_inl, n_xl, n_z, num_classes)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_prediction(prediction, section_idx, specs, seismic_data):
    pred_slice = prediction[0]                          # (n_xl, n_z, 9)
    class_map  = pred_slice.argmax(axis=-1)             # (n_xl, n_z)

    # SEGY coordinate extents for axis labels
    xl_min_s = specs["xl_start"] + section_idx[2] * specs["xl_step"]
    xl_max_s = specs["xl_start"] + section_idx[3] * specs["xl_step"]
    t_min_s  = specs["t_start"]  + section_idx[4] * specs["t_step"]
    t_max_s  = specs["t_start"]  + section_idx[5] * specs["t_step"]
    extent   = [xl_min_s, xl_max_s, t_max_s, t_min_s]

    # Matching seismic amplitude crop
    il_idx = section_idx[0]
    seis_crop = seismic_data[il_idx, section_idx[2]:section_idx[3]+1,
                             section_idx[4]:section_idx[5]+1, 0]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Inline 130 — Dutch F3  |  9-class facies prediction", fontsize=14)

    # Panel 1: seismic amplitude
    axes[0].imshow(seis_crop.T, aspect="auto", cmap="gray", extent=extent)
    axes[0].set_title("Seismic amplitude")
    axes[0].set_xlabel("Xline"); axes[0].set_ylabel("Time (ms)")

    # Panel 2: predicted class map (9 discrete colours)
    cmap9  = get_cmap("tab10", NUM_CLASSES)
    bounds = np.arange(-0.5, NUM_CLASSES + 0.5, 1)
    norm   = BoundaryNorm(bounds, cmap9.N)
    im2    = axes[1].imshow(class_map.T, aspect="auto",
                            cmap=cmap9, norm=norm, extent=extent)
    axes[1].set_title("Predicted facies class")
    axes[1].set_xlabel("Xline"); axes[1].set_ylabel("Time (ms)")
    cbar = plt.colorbar(im2, ax=axes[1], ticks=range(NUM_CLASSES))
    cbar.ax.set_yticklabels(FACIES_NAMES, fontsize=7)

    # Panel 3: max-class probability (confidence)
    max_prob = pred_slice.max(axis=-1)
    im3 = axes[2].imshow(max_prob.T, aspect="auto", cmap="plasma",
                         vmin=0, vmax=1, extent=extent)
    axes[2].set_title("Prediction confidence (max prob)")
    axes[2].set_xlabel("Xline"); axes[2].set_ylabel("Time (ms)")
    plt.colorbar(im3, ax=axes[2], label="Max class probability")

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "F3_multi_inline.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_fig}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Load seismic data
    data, specs = load_segy(SEGY_PATH)

    # 2. Load and validate .pts annotations
    print("\nLoading .pts annotation files ...")
    addrs = load_pts_files(PTS_DIR, FACIES_FILES, specs, data.shape)

    # 3. Stratified train/val split (by time blocks per class)
    tr_addrs, va_addrs = spatial_split(addrs, val_frac=0.20)

    # 4. Class weights (computed from training set)
    class_weights = compute_class_weights(tr_addrs)
    print(f"\n  Class weights: { {FACIES_NAMES[k]: round(v,2) for k,v in class_weights.items()} }")

    # 5. Generators
    train_gen = VoxeletGenerator(data, tr_addrs, BATCH_SIZE, NUM_CLASSES,
                                 CUBE_INCR, augment=True)
    val_gen   = VoxeletGenerator(data, va_addrs, BATCH_SIZE, NUM_CLASSES,
                                 CUBE_INCR, augment=False)

    # 6. Build model
    cube_size = 2 * CUBE_INCR + 1
    model = build_model(cube_size=cube_size, num_channels=1, num_classes=NUM_CLASSES)
    model.summary()

    # 7. Train
    print(f"\nTraining  ({EPOCHS} epochs max, early stopping on val_loss) ...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(MODEL_SAVE, monitor="val_loss", save_best_only=True,
                        verbose=1),
    ]
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
    )

    # Ensure best model is loaded (ModelCheckpoint already handles it, but reload
    # explicitly so predict uses the same saved weights)
    model = keras.models.load_model(
        MODEL_SAVE,
        custom_objects={"SpatialDropout3D": SpatialDropout3D},
        compile=False,
    )
    print(f"\nBest model loaded from: {MODEL_SAVE}")

    # 8. Predict
    section_idx = segy_to_index(SECTION_SEGY, specs, data.shape)
    prediction  = predict_section(data, model, section_idx, NUM_CLASSES,
                                  batch_size=BATCH_SIZE)

    np.save(os.path.join(OUT_DIR, "F3_multi_prob.npy"),  prediction)
    np.save(os.path.join(OUT_DIR, "F3_multi_class.npy"), prediction.argmax(axis=-1).astype(np.int8))
    print(f"Saved  F3_multi_prob.npy  shape={prediction.shape}")
    print(f"Saved  F3_multi_class.npy shape={prediction.shape[:3]}")

    # 9. Plot
    print("\nPlotting ...")
    plot_prediction(prediction, section_idx, specs, data)

    # 10. Class distribution summary
    cls_map = prediction[0].argmax(axis=-1)
    print("\nPredicted class distribution:")
    for c in range(NUM_CLASSES):
        pct = 100 * (cls_map == c).sum() / cls_map.size
        print(f"  class {c} ({FACIES_NAMES[c]:20s}): {pct:5.1f}%")


if __name__ == "__main__":
    main()
