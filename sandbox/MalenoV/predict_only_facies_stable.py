"""
Stable prediction script -- coarse-stride + interpolation (recommended).

This is the recommended MalenoV facies-segmentation predictor. It reuses the
trained model at models/F3_multiclass_model.h5 and predicts a single inline
(configured via SECTION_SEGY in facies_common.py).

Background: on an under-provisioned laptop GPU (~3.3 GB usable, compute
capability 12.0a with PTX-JIT kernels) the bottleneck is GPU inference, not
CPU voxel slicing, so larger batches do not help (see predict_only_facies_v2).
The only way to get a real speedup is to run the CNN on FEWER voxels.

Strategy:
  * Predict facies only on a subsampled grid of voxel centres (every STRIDE-th
    xline and time sample) -> STRIDE**2 fewer CNN evaluations.
  * Upsample the resulting probability volume back to full resolution with
    smooth (order-1) interpolation, then take the argmax for the class map.

Because facies are spatially smooth, STRIDE=2 (4x fewer voxels) reproduces the
full-resolution map almost exactly while running ~STRIDE**2 (~4x) faster.

Companion scripts:
  * predict_only_facies_v1.py -- naive per-voxel loop (reference, slowest)
  * predict_only_facies_v2.py -- vectorized batching (no real speedup here)
  * predict_only_facies_stable.py -- this file (recommended)
"""

import argparse
import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import zoom

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import sys
sys.path.insert(0, os.path.dirname(__file__))
from facies_common import (
    SEGY_PATH, OUT_DIR, MODEL_SAVE, NUM_CLASSES, SECTION_SEGY, CUBE_INCR,
    FACIES_NAMES, SpatialDropout3D,
    load_segy, segy_to_index, plot_prediction,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
STRIDE = 2      # subsampling step in xline & time (2 -> 4x fewer voxels)
BATCH  = 256    # moderate batch: avoids OOM thrashing on the ~3.3GB GPU pool

# Measured full-resolution per-voxel baseline (v1/v2, inline 140, 358,182 vox).
BASELINE_SECONDS = 649.52


def parse_args():
    parser = argparse.ArgumentParser(description="Run MalenoV facies prediction on one Dutch F3 inline.")
    parser.add_argument(
        "legacy_inline", nargs="?", type=int,
        help="Backward-compatible positional inline number.",
    )
    parser.add_argument("--inline", dest="inline_num", type=int, help="SEG-Y inline coordinate.")
    args = parser.parse_args()
    if args.legacy_inline is not None and args.inline_num is not None:
        parser.error("positional inline cannot be combined with --inline")
    return args.inline_num if args.inline_num is not None else args.legacy_inline


def predict_section_strided(data, model, section_idx, num_classes,
                            stride=STRIDE, batch_size=BATCH):
    ci = CUBE_INCR
    cs = 2 * ci + 1
    inl_min, inl_max, xl_min, xl_max, t_min, t_max = section_idx
    n_inl = inl_max - inl_min + 1
    n_xl  = xl_max  - xl_min  + 1
    n_z   = t_max   - t_min   + 1

    # Coarse centre grids (always include the last index so the map spans fully)
    xl_centers = np.unique(np.append(np.arange(xl_min, xl_max + 1, stride), xl_max))
    z_centers  = np.unique(np.append(np.arange(t_min,  t_max  + 1, stride), t_max))
    n_xlc, n_zc = len(xl_centers), len(z_centers)

    full_total   = n_inl * n_xl * n_z
    coarse_total = n_inl * n_xlc * n_zc
    print(f"\n[STRIDE={stride}] full grid  : {n_xl} × {n_z} = "
          f"{n_xl * n_z:,} voxels/inline")
    print(f"[STRIDE={stride}] coarse grid: {n_xlc} × {n_zc} = "
          f"{n_xlc * n_zc:,} voxels/inline  "
          f"({full_total / max(coarse_total,1):.2f}x fewer CNN evals)")

    data3d  = data[..., 0]
    windows = sliding_window_view(data3d, (cs, cs, cs))

    # Coarse (xl, z) centre index pairs, xl-major order
    xg, zg = np.meshgrid(xl_centers, z_centers, indexing="ij")
    xg = xg.ravel()
    zg = zg.ravel()

    out_full = np.empty((n_inl, n_xl, n_z, num_classes), dtype=np.float32)

    for i in range(n_inl):
        il = inl_min + i
        coarse = np.empty((len(xg), num_classes), dtype=np.float32)
        for start in range(0, len(xg), batch_size):
            end = min(start + batch_size, len(xg))
            batch = windows[il - ci, xg[start:end] - ci, zg[start:end] - ci]
            batch = batch[..., np.newaxis].astype(np.float32, copy=False)
            coarse[start:end] = model.predict_on_batch(batch)

        coarse = coarse.reshape(n_xlc, n_zc, num_classes)

        # Smoothly upsample the coarse probability grid to full resolution.
        zoom_factors = (n_xl / n_xlc, n_z / n_zc, 1.0)
        up = zoom(coarse, zoom_factors, order=1)          # linear interpolation
        # zoom may be off by a pixel; crop/pad to exact shape
        up = up[:n_xl, :n_z, :]
        if up.shape[0] < n_xl or up.shape[1] < n_z:
            pad = ((0, n_xl - up.shape[0]), (0, n_z - up.shape[1]), (0, 0))
            up = np.pad(up, pad, mode="edge")
        out_full[i] = up
        print(f"  [STRIDE] inline {i+1}/{n_inl} done", flush=True)

    return out_full


def main():
    inline_override = parse_args()
    section_segy = SECTION_SEGY.copy()
    if inline_override is not None:
        section_segy[0] = inline_override
        section_segy[1] = inline_override

    data, specs = load_segy(SEGY_PATH)

    print(f"\nLoading saved model: {MODEL_SAVE}")
    model = keras.models.load_model(
        MODEL_SAVE,
        custom_objects={"SpatialDropout3D": SpatialDropout3D},
        compile=False,
    )

    section_idx = segy_to_index(section_segy, specs, data.shape)

    cs = 2 * CUBE_INCR + 1
    print("\nWarming up GPU ...")
    model.predict_on_batch(np.zeros((8, cs, cs, cs, 1), dtype=np.float32))

    t0 = time.perf_counter()
    pred = predict_section_strided(data, model, section_idx, NUM_CLASSES)
    t_pred = time.perf_counter() - t0

    inl_tag = int(section_segy[0])
    cls_map = pred.argmax(axis=-1)

    print("\n" + "=" * 60)
    print(f"  stable  strided (STRIDE={STRIDE}, batch={BATCH})  inline {inl_tag}")
    print("=" * 60)
    print(f"  prediction time         : {t_pred:8.2f} s")
    print(f"  full-res baseline (v1/v2): {BASELINE_SECONDS:8.2f} s")
    print(f"  speedup vs full-res     : {BASELINE_SECONDS / t_pred:8.2f}x")
    print("=" * 60)

    np.save(os.path.join(OUT_DIR, f"F3_multi_prob_{inl_tag}.npy"), pred)
    np.save(os.path.join(OUT_DIR, f"F3_multi_class_{inl_tag}.npy"),
            cls_map.astype(np.int8))
    print(f"\nSaved  F3_multi_prob_{inl_tag}.npy  shape={pred.shape}")

    print("Plotting ...")
    plot_prediction(pred, section_idx, specs, data)

    print("\nPredicted class distribution:")
    cmap = pred[0].argmax(axis=-1)
    for c in range(NUM_CLASSES):
        pct = 100 * (cmap == c).sum() / cmap.size
        print(f"  class {c} ({FACIES_NAMES[c]:20s}): {pct:5.1f}%")


if __name__ == "__main__":
    main()
