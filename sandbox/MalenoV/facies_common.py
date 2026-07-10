"""
Shared config and helpers for MalenoV seismic-facies segmentation on Dutch F3.

Imported by both the trainer (train_seismic_facies.py) and the predictors
(predict_only_facies_*.py). Holds:
  * paths / hyper-parameters / class definitions
  * the SpatialDropout3D custom layer
  * SEGY loading and coordinate conversion
  * the reference per-voxel predictor and the 3-panel plot
"""

import os

import matplotlib
import numpy as np
import segyio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
import tf_keras.backend as K

# Enable GPU memory growth to avoid upfront full-VRAM allocation and
# fragmentation-related OOMs during long prediction loops.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# ── Configuration ─────────────────────────────────────────────────────────────
# Repo root resolved relative to this file (sandbox/MalenoV/facies_common.py),
# so the scripts work regardless of where the repo is checked out.
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SEGY_PATH   = os.path.join(REPO_ROOT, "data", "Dutch F3 seismic data", "Dutch Government_F3_entire_8bit seismic.segy")
PTS_DIR     = os.path.join(REPO_ROOT, "tools", "facies_net", "class_addresses")
OUT_DIR     = os.path.join(REPO_ROOT, "outputs")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")
MODEL_SAVE  = os.path.join(MODELS_DIR, "F3_multiclass_model.h5")

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

# Inline to predict: [inl_min, inl_max, xl_min, xl_max, t_min, t_max] (SEGY coords)
SECTION_SEGY = np.array([130, 130, 330, 1220, 124, 1728])

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


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


# ── Coordinate conversion ─────────────────────────────────────────────────────
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


# ── Reference predictor (naive per-voxel loop) ────────────────────────────────
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
                    probs[idx - batch_size : idx] = model.predict_on_batch(buf)

    rem = idx % batch_size
    if rem > 0:
        probs[idx - rem : idx] = model.predict_on_batch(buf[:rem])

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

    inl_segy = specs["inl_start"] + il_idx * specs["inl_step"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Inline {inl_segy} — Dutch F3  |  9-class facies prediction", fontsize=14)

    # Panel 1: seismic amplitude — symmetric 98th-percentile clip to match
    # the Boglodite console viewer (class/probability panels stay unscaled).
    vclip = float(np.percentile(np.abs(seis_crop), 98)) or 1.0
    axes[0].imshow(seis_crop.T, aspect="auto", cmap="gray",
                   vmin=-vclip, vmax=vclip, extent=extent)
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
    out_fig = os.path.join(OUT_DIR, f"F3_multi_inline_{inl_segy}.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_fig}")
