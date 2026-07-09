"""
Prediction-only script v2 -- GPU-optimized facies segmentation.

Difference vs predict_only_facies.py:
  v1 (predict_section) extracts voxels one-at-a-time inside a pure-Python
  triple nested loop (n_xl * n_z iterations), copying each 61x61x61 cube
  individually before it can fill a 64-voxel batch. The CPU slicing is the
  bottleneck and the GPU sits mostly idle (~18% util).

  v2 (predict_section_fast) uses numpy.lib.stride_tricks.sliding_window_view
  to obtain a *zero-copy* view of every 61x61x61 window in the cube, then
  gathers LARGE batches (default 1024) with a single vectorized fancy-index
  operation (executed in C) and hands them to the GPU. This keeps the GPU
  fed with big batches instead of starving it with tiny CPU-prepared ones.

The script times BOTH methods on the same inline and verifies their outputs
agree, then saves the v2 result + plot.
"""

import os
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

# Enable GPU memory growth to avoid OOM during prediction.
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
    load_segy, segy_to_index, predict_section, plot_prediction,
)

# Large batch for the vectorized path -> higher GPU utilization.
FAST_BATCH = 1024


def predict_section_fast(data, model, section_idx, num_classes,
                         batch_size=FAST_BATCH):
    """
    Vectorized voxelet extraction using a sliding-window VIEW (no bulk copy).

    For each requested inline, `sliding_window_view` exposes every
    (cs, cs, cs) sub-cube of the volume as a strided view. We then materialize
    only one batch of windows at a time via fancy indexing and predict on the
    whole batch, giving the GPU large, contiguous workloads.
    """
    ci = CUBE_INCR
    cs = 2 * ci + 1
    inl_min, inl_max, xl_min, xl_max, t_min, t_max = section_idx
    n_inl = inl_max - inl_min + 1
    n_xl  = xl_max  - xl_min  + 1
    n_z   = t_max   - t_min   + 1
    total = n_inl * n_xl * n_z

    print(f"\n[FAST] Predicting {n_inl} × {n_xl} × {n_z} = {total:,} voxels "
          f"(batch={batch_size}) ...")

    data3d = data[..., 0]                                   # (ni, nx, nz) view
    windows = sliding_window_view(data3d, (cs, cs, cs))     # strided view

    probs = np.empty((total, num_classes), dtype=np.float32)
    out_ptr = 0

    for i in range(n_inl):
        il = inl_min + i
        # Windows for this inline: shape (n_xl, n_z, cs, cs, cs) -- still a view
        w = windows[il - ci,
                    xl_min - ci: xl_max - ci + 1,
                    t_min  - ci: t_max  - ci + 1]

        # Flat (xl-major, z-minor) index order to match the reference loop
        flat_total = n_xl * n_z
        for start in range(0, flat_total, batch_size):
            end = min(start + batch_size, flat_total)
            fidx = np.arange(start, end)
            xs = fidx // n_z
            zs = fidx % n_z
            # Fancy indexing materializes only this batch (C-level gather)
            batch = w[xs, zs]                     # (b, cs, cs, cs)
            batch = batch[..., np.newaxis].astype(np.float32, copy=False)
            probs[out_ptr + start: out_ptr + end] = model.predict_on_batch(batch)

        out_ptr += flat_total
        print(f"  [FAST] inline {i+1}/{n_inl} done", flush=True)

    return probs.reshape(n_inl, n_xl, n_z, num_classes)


def main():
    data, specs = load_segy(SEGY_PATH)

    print(f"\nLoading saved model: {MODEL_SAVE}")
    model = keras.models.load_model(
        MODEL_SAVE,
        custom_objects={"SpatialDropout3D": SpatialDropout3D},
        compile=False,
    )

    section_idx = segy_to_index(SECTION_SEGY, specs, data.shape)

    cs = 2 * CUBE_INCR + 1
    # Warm up the GPU (JIT / graph build) so timings are fair.
    print("\nWarming up GPU ...")
    model.predict_on_batch(np.zeros((8, cs, cs, cs, 1), dtype=np.float32))

    # ── v2: vectorized fast path ──────────────────────────────────────────────
    t0 = time.perf_counter()
    pred_fast = predict_section_fast(data, model, section_idx, NUM_CLASSES)
    t_fast = time.perf_counter() - t0

    # ── v1: original per-voxel loop (for direct comparison) ───────────────────
    t0 = time.perf_counter()
    pred_slow = predict_section(data, model, section_idx, NUM_CLASSES,
                                batch_size=64)
    t_slow = time.perf_counter() - t0

    # ── Verify equivalence ────────────────────────────────────────────────────
    cls_fast = pred_fast.argmax(axis=-1)
    cls_slow = pred_slow.argmax(axis=-1)
    agree = 100.0 * (cls_fast == cls_slow).mean()

    print("\n" + "=" * 60)
    print("  PREDICTION TIME COMPARISON  (inline "
          f"{int(SECTION_SEGY[0])}, {cls_fast.size:,} voxels)")
    print("=" * 60)
    print(f"  v1  per-voxel loop (batch 64)   : {t_slow:8.2f} s")
    print(f"  v2  vectorized     (batch {FAST_BATCH}) : {t_fast:8.2f} s")
    print(f"  speedup                         : {t_slow / t_fast:8.2f}x")
    print(f"  class-map agreement             : {agree:8.2f}%")
    print("=" * 60)

    # ── Save v2 results + plot ────────────────────────────────────────────────
    inl_tag = int(SECTION_SEGY[0])
    np.save(os.path.join(OUT_DIR, f"F3_multi_prob_{inl_tag}_v2.npy"), pred_fast)
    np.save(os.path.join(OUT_DIR, f"F3_multi_class_{inl_tag}_v2.npy"),
            cls_fast.astype(np.int8))
    print(f"\nSaved  F3_multi_prob_{inl_tag}_v2.npy  shape={pred_fast.shape}")

    print("Plotting ...")
    plot_prediction(pred_fast, section_idx, specs, data)

    print("\nPredicted class distribution (v2):")
    cmap = pred_fast[0].argmax(axis=-1)
    for c in range(NUM_CLASSES):
        pct = 100 * (cmap == c).sum() / cmap.size
        print(f"  class {c} ({FACIES_NAMES[c]:20s}): {pct:5.1f}%")


if __name__ == "__main__":
    main()
