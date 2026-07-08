"""
Prediction-only script for facies segmentation on a given inline
(configured via SECTION_SEGY in facies_common.py).

Reuses the already-trained model at models/F3_multiclass_model.h5
(saved by sandbox/train_seismic_facies.py) and skips training
entirely -- just loads SEGY, predicts the configured inline, saves .npy + plot.
"""

import os

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras

# Enable GPU memory growth to avoid OOM during the long prediction loop.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import sys
sys.path.insert(0, os.path.dirname(__file__))
from facies_common import (
    SEGY_PATH, OUT_DIR, MODEL_SAVE, NUM_CLASSES, SECTION_SEGY, BATCH_SIZE,
    SpatialDropout3D, load_segy, segy_to_index, predict_section, plot_prediction,
)


def main():
    data, specs = load_segy(SEGY_PATH)

    print(f"\nLoading saved model: {MODEL_SAVE}")
    model = keras.models.load_model(
        MODEL_SAVE,
        custom_objects={"SpatialDropout3D": SpatialDropout3D},
        compile=False,
    )

    section_idx = segy_to_index(SECTION_SEGY, specs, data.shape)
    prediction = predict_section(data, model, section_idx, NUM_CLASSES,
                                  batch_size=BATCH_SIZE)

    inl_tag = int(SECTION_SEGY[0])
    np.save(os.path.join(OUT_DIR, f"F3_multi_prob_{inl_tag}.npy"), prediction)
    np.save(os.path.join(OUT_DIR, f"F3_multi_class_{inl_tag}.npy"),
            prediction.argmax(axis=-1).astype(np.int8))
    print(f"Saved  F3_multi_prob_{inl_tag}.npy  shape={prediction.shape}")
    print(f"Saved  F3_multi_class_{inl_tag}.npy shape={prediction.shape[:3]}")

    print("\nPlotting ...")
    plot_prediction(prediction, section_idx, specs, data)

    cls_map = prediction[0].argmax(axis=-1)
    print("\nPredicted class distribution:")
    from facies_common import FACIES_NAMES
    for c in range(NUM_CLASSES):
        pct = 100 * (cls_map == c).sum() / cls_map.size
        print(f"  class {c} ({FACIES_NAMES[c]:20s}): {pct:5.1f}%")


if __name__ == "__main__":
    main()
