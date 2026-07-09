"""
FaultSeg fault extraction on the Dutch F3 seismic dataset — single source of truth.

Predict-only: runs the pre-trained FaultSeg 3D U-Net (Wu et al., 2019) on a
user-chosen section and saves a seismic + fault-probability overlay.

Command pattern (mirrors MalenoV): edit SECTION_SEGY below, then run

    uv run python sandbox/FaultSeg/predict_only_fault.py

SECTION_SEGY selects the slice, in SEGY coordinates:
    [inl_min, inl_max, xl_min, xl_max, t_min, t_max]
Collapse ONE dimension (min == max) to pick the slice orientation:
    * inline slice    -> inl_min == inl_max      (e.g. [150,150, 300,1250, 4,1848])
    * crossline slice -> xl_min  == xl_max       (e.g. [100,750,  800, 800, 4,1848])
    * time slice      -> t_min   == t_max        (e.g. [100,750, 300,1250, 1000,1000])
The other two dimensions default to the full survey extent if left wide.

Memory-safe: predicts the full section by tiling the free horizontal axes into
overlapping windows and blending them with a cosine taper, so there is no OOM
and no crop. Uses the GPU by default (with memory growth enabled to avoid
upfront full-VRAM allocation); falls back to CPU automatically if no GPU is
visible.

Outputs (written to outputs/):
    F3_fault_<orient>_<num>.npy   -- fault probability slice
    F3_fault_<orient>_<num>.png   -- seismic + fault probability overlay
"""

import math
import os

import sys, os, numpy as np
import segyio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

# Enable GPU memory growth to avoid upfront full-VRAM allocation and
# fragmentation-related OOMs during tiled prediction.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# ── Configuration ─────────────────────────────────────────────────────────────
# Repo root resolved relative to this file (sandbox/FaultSeg/predict_only_fault.py),
# so the script works regardless of where the repo is checked out.
REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEGY_PATH  = os.path.join(REPO_ROOT, "data", "Dutch F3 seismic data", "Dutch Government_F3_entire_8bit seismic.segy")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "faultSeg_model", "model", "fseg-60.hdf5")
OUT_DIR    = os.path.join(REPO_ROOT, "outputs")

# Section to predict, SEGY coords: [inl_min, inl_max, xl_min, xl_max, t_min, t_max]
# Collapse one pair (min == max) to choose inline / crossline / time slice.
SECTION_SEGY = np.array([150, 150, 300, 1250, 4, 1848])

# Optional CLI override: `uv run python predict_only_fault.py <inline_num>`
# selects a full inline slice at that inline number.
if len(sys.argv) > 1:
    SECTION_SEGY = np.array([int(sys.argv[1]), int(sys.argv[1]), 300, 1250, 4, 1848])

CONTEXT = 128    # context window (voxels) along the slice axis for inline/xline
TILE    = 256    # window size along each tiled (free) horizontal axis
OVER    = 64     # overlap between consecutive tiles (cosine-blended)

os.makedirs(OUT_DIR, exist_ok=True)


# ── Custom balanced loss (needed to load the checkpoint) ──────────────────────
def cross_entropy_balanced(y_true, y_pred):
    _eps   = tf.cast(keras.backend.epsilon(), y_pred.dtype)
    y_pred = tf.clip_by_value(y_pred, _eps, 1.0 - _eps)
    y_pred = tf.math.log(y_pred / (1.0 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    count_neg  = tf.reduce_sum(1.0 - y_true)
    count_pos  = tf.reduce_sum(y_true)
    beta       = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1.0 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1.0 - beta))
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def load_model(path):
    print(f"\nLoading model: {path}")
    m = keras.models.load_model(
        path, custom_objects={"cross_entropy_balanced": cross_entropy_balanced},
        compile=False)
    print("  Model loaded.")
    return m


# ── SEGY helpers ──────────────────────────────────────────────────────────────
def read_specs(path):
    with segyio.open(path, "r", strict=False) as f:
        f.mmap()
        specs = dict(
            il_start=int(f.ilines[0]),  il_end=int(f.ilines[-1]),
            il_step=int(f.ilines[1] - f.ilines[0]),
            xl_start=int(f.xlines[0]),  xl_end=int(f.xlines[-1]),
            xl_step=int(f.xlines[1] - f.xlines[0]),
            t_start=float(f.samples[0]), t_end=float(f.samples[-1]),
            t_step=float(f.samples[1] - f.samples[0]),
        )
    return specs


def classify_section(section, specs):
    """Return (orient, target_idx, ranges) from SECTION_SEGY.

    ranges is a dict with index ranges (inclusive) for the two free axes.
    """
    inl_min, inl_max, xl_min, xl_max, t_min, t_max = [float(v) for v in section]

    def il_i(v): return int(round((v - specs["il_start"]) / specs["il_step"]))
    def xl_i(v): return int(round((v - specs["xl_start"]) / specs["xl_step"]))
    def t_i(v):  return int(round((v - specs["t_start"])  / specs["t_step"]))

    if inl_min == inl_max:
        return "inline", il_i(inl_min), dict(
            xl=(xl_i(xl_min), xl_i(xl_max)), t=(t_i(t_min), t_i(t_max)))
    if xl_min == xl_max:
        return "xline", xl_i(xl_min), dict(
            il=(il_i(inl_min), il_i(inl_max)), t=(t_i(t_min), t_i(t_max)))
    if t_min == t_max:
        return "timeslice", t_i(t_min), dict(
            il=(il_i(inl_min), il_i(inl_max)), xl=(xl_i(xl_min), xl_i(xl_max)))
    raise ValueError("SECTION_SEGY must collapse one dimension (min == max) "
                     "to select inline, crossline, or time slice.")


def clamp_window(center, size, n):
    """Window of `size` centred on `center`, clamped to [0, n); returns (s, e, local)."""
    size = min(size, n)
    s = center - size // 2
    s = max(0, min(s, n - size))
    e = s + size
    return s, e, center - s


# ── Tiling ────────────────────────────────────────────────────────────────────
def tile_windows(n, win, over):
    """Overlapping window starts across length n; each trimmed to a multiple of 8."""
    win = max(8, (min(win, n) // 8) * 8)
    stride = max(8, win - over)
    starts = list(range(0, max(1, n - win + 1), stride))
    if not starts or starts[-1] != n - win:
        starts.append(max(0, n - win))
    out = []
    for s in starts:
        e = min(s + win, n)
        w8 = (e - s) // 8 * 8
        if w8 >= 8:
            out.append((s, s + w8))
    return out


def cosine_weight(length, over):
    """1-D blend weight: cosine ramp up/down over `over` samples at each edge."""
    w = np.ones(length, dtype=np.float32)
    if over > 0 and length > 1:
        m = min(over, length)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
        w[:m]  *= ramp
        w[-m:] *= ramp[::-1]
    return w


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_slice(model, gx, orient, tgt_local):
    """
    gx : (Z, XL, IL) normalised + z-padded volume for the loaded sub-region.
    Returns the 2-D fault-probability plane and its axis meaning.

      inline    -> plane (Z, XL), tile over XL, IL is context, extract IL=tgt
      xline     -> plane (Z, IL), tile over IL, XL is context, extract XL=tgt
      timeslice -> plane (XL, IL), tile over XL & IL, Z is context, extract Z=tgt
    """
    n_z, n_xl, n_il = gx.shape

    if orient == "inline":
        acc  = np.zeros((n_z, n_xl), np.float32)
        wsum = np.zeros((n_z, n_xl), np.float32)
        wins = tile_windows(n_xl, TILE, OVER)
        print(f"\nInline: {len(wins)} XL tiles over {n_xl} xlines ...")
        for k, (s, e) in enumerate(wins):
            blk = gx[:, s:e, :].reshape(1, n_z, e - s, n_il, 1)
            fp  = model.predict(blk, verbose=0)[0, :, :, tgt_local, 0]  # (Z, e-s)
            w   = cosine_weight(e - s, OVER)[None, :]
            acc[:, s:e]  += fp * w
            wsum[:, s:e] += w
            print(f"  XL tile {k+1}/{len(wins)}  [{s}:{e}]")
        wsum[wsum == 0] = 1.0
        return acc / wsum, ("Z", "XL")

    if orient == "xline":
        acc  = np.zeros((n_z, n_il), np.float32)
        wsum = np.zeros((n_z, n_il), np.float32)
        wins = tile_windows(n_il, TILE, OVER)
        print(f"\nCrossline: {len(wins)} IL tiles over {n_il} inlines ...")
        for k, (s, e) in enumerate(wins):
            blk = gx[:, :, s:e].reshape(1, n_z, n_xl, e - s, 1)
            fp  = model.predict(blk, verbose=0)[0, :, tgt_local, :, 0]  # (Z, e-s)
            w   = cosine_weight(e - s, OVER)[None, :]
            acc[:, s:e]  += fp * w
            wsum[:, s:e] += w
            print(f"  IL tile {k+1}/{len(wins)}  [{s}:{e}]")
        wsum[wsum == 0] = 1.0
        return acc / wsum, ("Z", "IL")

    # timeslice: 2-D tiling over XL and IL
    acc  = np.zeros((n_xl, n_il), np.float32)
    wsum = np.zeros((n_xl, n_il), np.float32)
    xl_wins = tile_windows(n_xl, TILE, OVER)
    il_wins = tile_windows(n_il, TILE, OVER)
    total = len(xl_wins) * len(il_wins)
    print(f"\nTime slice: {total} tiles ({len(xl_wins)} XL × {len(il_wins)} IL) ...")
    c = 0
    for xs, xe in xl_wins:
        wx = cosine_weight(xe - xs, OVER)
        for is_, ie in il_wins:
            wi  = cosine_weight(ie - is_, OVER)
            blk = gx[:, xs:xe, is_:ie].reshape(1, n_z, xe - xs, ie - is_, 1)
            fp  = model.predict(blk, verbose=0)[0, tgt_local, :, :, 0]  # (xe-xs, ie-is)
            w   = np.outer(wx, wi)
            acc[xs:xe, is_:ie]  += fp * w
            wsum[xs:xe, is_:ie] += w
            c += 1
            print(f"  tile {c}/{total}  XL[{xs}:{xe}] IL[{is_}:{ie}]")
    wsum[wsum == 0] = 1.0
    return acc / wsum, ("XL", "IL")


# ── Plotting ──────────────────────────────────────────────────────────────────
def fault_rgba(f, cmap_name="hot_r"):
    rgba = plt.get_cmap(cmap_name)(f)
    rgba[..., 3] = f
    return rgba.astype(np.float32)


def plot_section(seis, fault, orient, num, axes_meaning, specs,
                 il_lo, xl_lo, out_path):
    (ay, ax_) = axes_meaning
    # Build extents from SEGY coords for whichever axes are shown.
    def il_coord(i): return specs["il_start"] + (il_lo + i) * specs["il_step"]
    def xl_coord(i): return specs["xl_start"] + (xl_lo + i) * specs["xl_step"]

    fig, ax = plt.subplots(figsize=(18, 7))
    label = {"inline": "Inline", "xline": "Crossline",
             "timeslice": "Time slice (ms)"}[orient]
    fig.suptitle(f"FaultSeg (Wu et al., 2019) — Dutch F3  |  {label} {num}",
                 fontsize=14)

    if orient in ("inline", "xline"):
        free = seis.shape[1]
        if ax_ == "XL":
            x0, x1 = xl_coord(0), xl_coord(free - 1); xlabel = "Xline"
        else:
            x0, x1 = il_coord(0), il_coord(free - 1); xlabel = "Inline"
        extent = [x0, x1, specs["t_end"], specs["t_start"]]
        ax.imshow(seis, aspect="auto", cmap="gray", extent=extent)
        ax.imshow(fault_rgba(fault), aspect="auto", extent=extent)
        ax.set_xlabel(xlabel); ax.set_ylabel("Time (ms)")
    else:  # timeslice: plane (XL, IL) -> show IL on x, XL on y
        img = seis.T; fimg = fault.T           # (IL, XL)
        extent = [xl_coord(0), xl_coord(seis.shape[0] - 1),
                  il_coord(seis.shape[1] - 1), il_coord(0)]
        ax.imshow(img, aspect="auto", cmap="gray", extent=extent)
        ax.imshow(fault_rgba(fimg), aspect="auto", extent=extent)
        ax.set_xlabel("Xline"); ax.set_ylabel("Inline")

    sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(0, 1)); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Fault probability")
    ax.set_title("Seismic amplitude + fault probability overlay (tiled/blended)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    specs = read_specs(SEGY_PATH)
    orient, tgt_idx, ranges = classify_section(SECTION_SEGY, specs)
    num = {"inline": int(SECTION_SEGY[0]),
           "xline": int(SECTION_SEGY[2]),
           "timeslice": int(SECTION_SEGY[4])}[orient]
    print(f"=== FaultSeg — {orient} {num} ===")

    print(f"Loading SEGY: {SEGY_PATH}")
    cube = segyio.tools.cube(SEGY_PATH).astype(np.float32)   # (IL, XL, Z)
    n_il, n_xl, n_z = cube.shape
    print(f"  Full cube (IL, XL, Z): {cube.shape}")

    # Decide the loaded sub-region + local target index along the slice axis.
    if orient == "inline":
        s, e, tgt_local = clamp_window(tgt_idx, CONTEXT, n_il)
        (xl0, xl1) = ranges["xl"]; (t0, t1) = ranges["t"]
        xl0 = max(0, xl0); xl1 = min(n_xl - 1, xl1)
        t0 = max(0, t0);   t1 = min(n_z - 1, t1)
        sub = cube[s:e, xl0:xl1 + 1, t0:t1 + 1]
        il_lo, xl_lo = s, xl0
    elif orient == "xline":
        s, e, tgt_local = clamp_window(tgt_idx, CONTEXT, n_xl)
        (il0, il1) = ranges["il"]; (t0, t1) = ranges["t"]
        il0 = max(0, il0); il1 = min(n_il - 1, il1)
        t0 = max(0, t0);   t1 = min(n_z - 1, t1)
        sub = cube[il0:il1 + 1, s:e, t0:t1 + 1]
        il_lo, xl_lo = il0, s
    else:  # timeslice: full Z (context), free IL & XL
        (il0, il1) = ranges["il"]; (xl0, xl1) = ranges["xl"]
        il0 = max(0, il0); il1 = min(n_il - 1, il1)
        xl0 = max(0, xl0); xl1 = min(n_xl - 1, xl1)
        sub = cube[il0:il1 + 1, xl0:xl1 + 1, :]
        tgt_local = tgt_idx
        il_lo, xl_lo = il0, xl0
    del cube
    print(f"  Sub-volume (IL, XL, Z): {sub.shape}  target_local={tgt_local}")

    # Normalise, pad time to a multiple of 8, transpose to (Z, XL, IL).
    gx = (sub - sub.mean()) / (sub.std() + 1e-8)
    del sub
    zi = gx.shape[2]
    z_pad = math.ceil(zi / 8) * 8
    if z_pad != zi:
        gx = np.pad(gx, ((0, 0), (0, 0), (0, z_pad - zi)), mode="constant")
    gx = np.transpose(gx).astype(np.float32)     # (Z_pad, XL, IL)
    if orient == "timeslice" and tgt_local >= zi:
        tgt_local = zi - 1
    print(f"  Prepared (Z_pad, XL, IL): {gx.shape}")

    model = load_model(MODEL_PATH)
    fault, axes_meaning = predict_slice(model, gx, orient, tgt_local)

    # Matching seismic background + unpad time for inline/xline planes.
    if orient == "inline":
        seis = gx[:zi, :, tgt_local]
        fault = fault[:zi, :]
    elif orient == "xline":
        seis = gx[:zi, tgt_local, :]
        fault = fault[:zi, :]
    else:
        seis = gx[tgt_local, :, :]               # (XL, IL)

    print(f"\n{orient} {num}: seismic {seis.shape}  fault {fault.shape}  "
          f"(mean={fault.mean():.4f}, max={fault.max():.4f})")

    tag = {"inline": "inline", "xline": "xline", "timeslice": "timeslice"}[orient]
    npy = os.path.join(OUT_DIR, f"F3_fault_{tag}_{num}.npy")
    np.save(npy, fault)
    print(f"Saved: {npy}  shape={fault.shape}")
    plot_section(seis, fault, orient, num, axes_meaning, specs,
                 il_lo, xl_lo, os.path.join(OUT_DIR, f"F3_fault_{tag}_{num}.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
