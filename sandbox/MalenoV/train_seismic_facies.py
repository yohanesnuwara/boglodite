"""
Multi-class seismic facies TRAINING on the Dutch F3 dataset (MalenoV skill).

Trains a 4-layer 3D CNN on 9 annotated facies classes (.pts files) and saves
the best model to models/F3_multiclass_model.h5. Prediction/plotting is handled
separately by the predict_only_facies_*.py scripts (see facies_common.py for the
shared config and helpers).

Run via the GPU wrapper:
    bash sandbox/run_facies_gpu.sh train_seismic_facies.py
"""

import math
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from facies_common import (
    SEGY_PATH, PTS_DIR, MODEL_SAVE, CUBE_INCR, BATCH_SIZE, EPOCHS,
    FACIES_FILES, FACIES_NAMES, NUM_CLASSES,
    SpatialDropout3D, load_segy,
)

import tf_keras as keras
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import (
    Activation, BatchNormalization, Conv3D, Dense, Dropout, Flatten
)
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam


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


# ── Main (training only) ──────────────────────────────────────────────────────
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

    print(f"\nTraining complete. Best model saved to: {MODEL_SAVE}")
    print("Run a predict_only_facies_*.py script to segment an inline.")


if __name__ == "__main__":
    main()
