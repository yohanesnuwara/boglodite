"""
Edge-safe MalenoV facies predictor for inlines near the survey boundary.

The standard predictors (predict_only_facies_stable.py, ...) center a
61x61x61 (CUBE_INCR=30) voxelet on every voxel, so they require >= CUBE_INCR
inlines of context on each side. That makes any inline within CUBE_INCR of the
inline start/end unpredictable (e.g. Dutch F3 inline 110 = index 10, only 10
inlines from the start at inline 100).

This wrapper edge-pads the inline axis by CUBE_INCR on both sides (replicating
the boundary inline as synthetic context) so ANY inline can be segmented,
including boundary inlines. All other logic (model, strided inference,
plotting) is reused unchanged from the existing scripts.

Usage:
  bash sandbox/MalenoV/run_facies_gpu.sh predict_inline_edge.py           # SECTION_SEGY[0]
  INLINE=110 bash sandbox/MalenoV/run_facies_gpu.sh predict_inline_edge.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from facies_common import (
    SEGY_PATH, OUT_DIR, MODEL_SAVE, NUM_CLASSES, SECTION_SEGY, CUBE_INCR,
    FACIES_NAMES, SpatialDropout3D,
    load_segy, segy_to_index, plot_prediction, keras,
)
from predict_only_facies_stable import predict_section_strided, STRIDE, BATCH


def main():
    # Target inline: env var INLINE overrides SECTION_SEGY[0].
    inline = int(os.environ.get("INLINE", SECTION_SEGY[0]))
    section = SECTION_SEGY.copy()
    section[0] = inline
    section[1] = inline
    print(f"=== MalenoV facies — inline {inline} (edge-safe) ===")

    data, specs = load_segy(SEGY_PATH)

    # Edge-pad the inline axis so boundary inlines gain valid voxelet context.
    pad = CUBE_INCR
    data = np.pad(data, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
    # Shift the inline origin so padded index 0 still maps to the correct SEGY
    # inline number (keeps segy_to_index + plot labels correct).
    specs = dict(specs)
    specs["inl_start"] -= pad * specs["inl_step"]
    print(f"  Padded inline axis by {pad} each side -> cube {data.shape}")

    print(f"\nLoading saved model: {MODEL_SAVE}")
    model = keras.models.load_model(
        MODEL_SAVE,
        custom_objects={"SpatialDropout3D": SpatialDropout3D},
        compile=False,
    )

    section_idx = segy_to_index(section, specs, data.shape)

    cs = 2 * CUBE_INCR + 1
    print("\nWarming up GPU ...")
    model.predict_on_batch(np.zeros((8, cs, cs, cs, 1), dtype=np.float32))

    t0 = time.perf_counter()
    pred = predict_section_strided(data, model, section_idx, NUM_CLASSES)
    t_pred = time.perf_counter() - t0
    print(f"\nPrediction time: {t_pred:.2f} s")

    cls_map = pred.argmax(axis=-1)
    np.save(os.path.join(OUT_DIR, f"F3_multi_prob_{inline}.npy"), pred)
    np.save(os.path.join(OUT_DIR, f"F3_multi_class_{inline}.npy"),
            cls_map.astype(np.int8))
    print(f"Saved  F3_multi_prob_{inline}.npy  shape={pred.shape}")

    print("Plotting ...")
    plot_prediction(pred, section_idx, specs, data)

    print("\nPredicted class distribution:")
    cmap = pred[0].argmax(axis=-1)
    for c in range(NUM_CLASSES):
        pct = 100 * (cmap == c).sum() / cmap.size
        print(f"  class {c} ({FACIES_NAMES[c]:20s}): {pct:5.1f}%")


if __name__ == "__main__":
    main()
