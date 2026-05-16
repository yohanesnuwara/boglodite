"""
FaultSeg fault extraction on inline 130 of Dutch F3 seismic dataset.

Replicates the structure of train_predict_f3.py but uses the pre-trained
FaultSeg 3D U-Net (Wu et al., 2019) instead of the facies CNN.

Workflow:
  1. Load Dutch F3 SEGY and extract a 3D sub-volume that includes inline 130.
  2. Normalise (zero-mean, unit-variance) and pad time dimension to nearest ×8.
  3. Run FaultSeg 3D U-Net inference on the whole sub-volume.
  4. Extract inline 130 slice from the fault-probability cube.
  5. Save results and produce a 2-panel seismic | fault-probability plot.

Output:
  outputs/F3_fault_inline130.npy   -- fault probability slice (samples × xlines)
  outputs/F3_fault_inline130.png   -- 2-panel plot
"""

import math
import os
import sys

import numpy as np
import segyio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

# ── Configuration ─────────────────────────────────────────────────────────────
SEGY_PATH  = "/workspace/boglodite/data/Dutch Government_F3_entire_8bit seismic.segy"
MODEL_PATH = "/data/faultSeg_model/model/fseg-60.hdf5"
OUT_DIR    = "/workspace/boglodite/outputs"

INLINE_TARGET = 130     # inline number to extract fault prediction for

# Sub-volume window: first N_IL inlines (100..227) cover inline 130 at index 30;
# first N_XL xlines (300..683) give a representative section.
N_IL           = 128
N_XL           = 384
IL_IDX_TARGET  = INLINE_TARGET - 100   # 0-based index of inline 130

os.makedirs(OUT_DIR, exist_ok=True)


# ── Custom balanced loss (required as custom_object when loading checkpoint) ───
def cross_entropy_balanced(y_true, y_pred):
    """Class-balanced binary cross-entropy from Wu et al. (2019)."""
    _eps   = tf.cast(keras.backend.epsilon(), y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, _eps, 1.0 - _eps)
    y_pred = tf.math.log(y_pred / (1.0 - y_pred))     # back to logits
    y_true = tf.cast(y_true, tf.float32)
    count_neg  = tf.reduce_sum(1.0 - y_true)
    count_pos  = tf.reduce_sum(y_true)
    beta       = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1.0 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    cost = tf.reduce_mean(cost * (1.0 - beta))
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


# ── 3D U-Net architecture (tf_keras re-implementation of unet3.py) ─────────────
def build_unet(input_size=(None, None, None, 1)):
    """Simplified 3D U-Net identical to FaultSeg's unet3.py."""
    inputs = keras.Input(input_size)

    # Encoder
    c1 = keras.layers.Conv3D(16, 3, activation="relu", padding="same")(inputs)
    c1 = keras.layers.Conv3D(16, 3, activation="relu", padding="same")(c1)
    p1 = keras.layers.MaxPooling3D(2)(c1)

    c2 = keras.layers.Conv3D(32, 3, activation="relu", padding="same")(p1)
    c2 = keras.layers.Conv3D(32, 3, activation="relu", padding="same")(c2)
    p2 = keras.layers.MaxPooling3D(2)(c2)

    c3 = keras.layers.Conv3D(64, 3, activation="relu", padding="same")(p2)
    c3 = keras.layers.Conv3D(64, 3, activation="relu", padding="same")(c3)
    p3 = keras.layers.MaxPooling3D(2)(c3)

    # Bottleneck
    c4 = keras.layers.Conv3D(128, 3, activation="relu", padding="same")(p3)
    c4 = keras.layers.Conv3D(128, 3, activation="relu", padding="same")(c4)

    # Decoder (skip connections from encoder)
    u5 = keras.layers.UpSampling3D(2)(c4)
    u5 = keras.layers.Concatenate()([u5, c3])
    c5 = keras.layers.Conv3D(64, 3, activation="relu", padding="same")(u5)
    c5 = keras.layers.Conv3D(64, 3, activation="relu", padding="same")(c5)

    u6 = keras.layers.UpSampling3D(2)(c5)
    u6 = keras.layers.Concatenate()([u6, c2])
    c6 = keras.layers.Conv3D(32, 3, activation="relu", padding="same")(u6)
    c6 = keras.layers.Conv3D(32, 3, activation="relu", padding="same")(c6)

    u7 = keras.layers.UpSampling3D(2)(c6)
    u7 = keras.layers.Concatenate()([u7, c1])
    c7 = keras.layers.Conv3D(16, 3, activation="relu", padding="same")(u7)
    c7 = keras.layers.Conv3D(16, 3, activation="relu", padding="same")(c7)

    outputs = keras.layers.Conv3D(1, 1, activation="sigmoid")(c7)
    return keras.Model(inputs, outputs)


# ── SEGY loading ───────────────────────────────────────────────────────────────
def load_subvolume(segy_path, n_il, n_xl):
    """
    Loads the Dutch F3 SEGY and extracts a (n_il, n_xl, n_z) sub-volume
    covering the first n_il inlines and n_xl xlines.
    Returns the sub-volume array and a specs dict for coordinate labels.
    """
    print(f"Loading SEGY: {segy_path}")
    with segyio.open(segy_path, "r", strict=False) as f:
        f.mmap()
        il_start = int(f.ilines[0])
        xl_start = int(f.xlines[0])
        xl_step  = int(f.xlines[1] - f.xlines[0])
        t_start  = float(f.samples[0])
        t_end    = float(f.samples[-1])
        t_step   = float(f.samples[1] - f.samples[0])
        n_t      = len(f.samples)
        print(f"  Full cube: {len(f.ilines)} IL × {len(f.xlines)} XL × {n_t} samples")

    data = segyio.tools.cube(segy_path).astype(np.float32)  # (n_il, n_xl, n_z)
    print(f"  Cube shape (IL, XL, Z): {data.shape}")

    sub = data[:n_il, :n_xl, :].copy()
    del data

    specs = dict(
        il_start=il_start,
        xl_start=xl_start, xl_step=xl_step,
        t_start=t_start, t_end=t_end, t_step=t_step, n_t=n_t,
    )
    print(f"  Sub-volume (IL, XL, Z): {sub.shape}")
    return sub, specs


# ── Normalise + pad + transpose ───────────────────────────────────────────────
def prepare_volume(sub):
    """
    Normalises seismic to zero-mean / unit-variance, pads the time dimension
    to the next multiple of 8 (required for clean U-Net skip connections),
    then transposes to FaultSeg convention: (n_z_pad, n_xl, n_il).

    Returns (gx_transposed, n_z_orig) so the caller can unpad after inference.
    """
    gx = (sub - sub.mean()) / (sub.std() + 1e-8)   # (n_il, n_xl, n_z)

    n_il, n_xl, n_z = gx.shape
    n_z_pad = math.ceil(n_z / 8) * 8
    if n_z_pad != n_z:
        pad_width = n_z_pad - n_z
        gx = np.pad(gx, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
        print(f"  Time dimension padded: {n_z} → {n_z_pad} (added {pad_width} zeros)")

    gx = np.transpose(gx).astype(np.float32)   # (n_z_pad, n_xl, n_il)
    return gx, n_z


# ── Load model ─────────────────────────────────────────────────────────────────
def load_faultseg_model(model_path):
    """
    Loads the pre-trained FaultSeg checkpoint.
    Falls back to rebuilding the architecture and loading weights if the full
    model-save format is not compatible with tf_keras.
    """
    print(f"\nLoading model: {model_path}")
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={"cross_entropy_balanced": cross_entropy_balanced},
            compile=False,
        )
        print("  Model loaded (architecture + weights from HDF5).")
    except Exception as exc:
        print(f"  load_model failed: {exc}")
        print("  Rebuilding architecture and loading weights only ...")
        model = build_unet()
        model.load_weights(model_path, by_name=False)
        print("  Weights loaded.")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_fault(model, gx):
    """
    Runs the FaultSeg U-Net on the full sub-volume.

    gx : float32 (n_z_pad, n_xl, n_il) — normalised, padded, transposed
    Returns fault-probability cube of the same shape.
    """
    n_z, n_xl, n_il = gx.shape
    print(f"\nRunning inference on volume ({n_z}, {n_xl}, {n_il}) ...")
    inp = gx.reshape(1, n_z, n_xl, n_il, 1)
    fp  = model.predict(inp, verbose=1)   # (1, n_z, n_xl, n_il, 1)
    fp  = fp[0, :, :, :, 0]              # (n_z, n_xl, n_il)
    print(f"  Fault probability range: [{fp.min():.4f}, {fp.max():.4f}]")
    return fp


# ── Plot ───────────────────────────────────────────────────────────────────────
def fault_rgba(fault_slice, cmap_name="hot_r"):
    """
    Converts a 2-D fault-probability array (values 0–1) into an RGBA image
    where the alpha channel equals the probability value:
      prob = 0  →  fully transparent   (invisible)
      prob = 1  →  fully opaque        (solid colour)
    """
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(fault_slice)                     # (n_z, n_xl, 4) RGBA float32
    rgba[..., 3] = fault_slice                   # alpha = probability
    return rgba.astype(np.float32)


def plot_inline(seis_slice, fault_slice, specs, inline_no, out_path):
    """
    Single-panel overlay: seismic amplitude in greyscale with fault probability
    blended on top. Alpha varies linearly from transparent (prob=0) to opaque
    (prob=1) so only genuine fault evidence is highlighted.

    seis_slice, fault_slice : (n_z, n_xl) arrays (time × xline)
    """
    n_z, n_xl = seis_slice.shape
    xl_end = specs["xl_start"] + (n_xl - 1) * specs["xl_step"]
    # extent = [left, right, bottom, top] — time increases downward
    extent = [specs["xl_start"], xl_end, specs["t_end"], specs["t_start"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"FaultSeg (Wu et al., 2019) — Dutch F3  |  Inline {inline_no}",
        fontsize=14,
    )

    # Background: seismic amplitude in greyscale
    ax.imshow(seis_slice, aspect="auto", cmap="gray", extent=extent)

    # Overlay: fault probability with alpha = probability value
    rgba = fault_rgba(fault_slice)
    im   = ax.imshow(rgba, aspect="auto", extent=extent)

    # Colourbar: map the hot_r colours to probability values (ignoring alpha)
    sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Fault probability")

    ax.set_title("Seismic amplitude + fault probability overlay")
    ax.set_xlabel("Xline")
    ax.set_ylabel("Time (ms)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== FaultSeg — Inline {INLINE_TARGET} extraction ===\n")

    # 1. Load sub-volume containing inline 130 (inlines 100–227, xlines 300–683)
    sub, specs = load_subvolume(SEGY_PATH, N_IL, N_XL)

    # 2. Normalise, pad, transpose → (n_z_pad, n_xl, n_il)
    gx, n_z_orig = prepare_volume(sub)
    del sub
    print(f"  Prepared volume shape (Z_pad, XL, IL): {gx.shape}")

    # 3. Load pre-trained FaultSeg model
    model = load_faultseg_model(MODEL_PATH)

    # 4. Run inference → fault probability cube (n_z_pad, n_xl, n_il)
    fp = predict_fault(model, gx)

    # 5. Unpad time dimension back to original sample count
    fp = fp[:n_z_orig, :, :]    # (n_z_orig, n_xl, n_il)
    gx = gx[:n_z_orig, :, :]

    # 6. Extract inline 130 slice
    seis_inline  = gx[:, :, IL_IDX_TARGET]     # (n_z, n_xl)
    fault_inline = fp[:, :, IL_IDX_TARGET]     # (n_z, n_xl)

    print(f"\nInline {INLINE_TARGET} (sub-volume index {IL_IDX_TARGET}) extracted:")
    print(f"  Seismic shape   : {seis_inline.shape}")
    print(f"  Fault prob shape: {fault_inline.shape}")
    print(f"  Fault prob stats: mean={fault_inline.mean():.4f}  "
          f"max={fault_inline.max():.4f}")

    # 7. Save and plot
    npy_out = os.path.join(OUT_DIR, "F3_fault_inline130.npy")
    np.save(npy_out, fault_inline)
    print(f"Saved: {npy_out}  shape={fault_inline.shape}")

    plot_inline(
        seis_inline, fault_inline, specs,
        inline_no=INLINE_TARGET,
        out_path=os.path.join(OUT_DIR, "F3_fault_inline130.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
