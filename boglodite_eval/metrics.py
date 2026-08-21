"""Numerical metrics for comparing agent outputs with canonical references."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_flat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def nrms_percent(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Symmetric normalized RMS difference in percent (0 = identical).

    NRMS = 200 * RMS(candidate-reference) / (RMS(candidate) + RMS(reference)).
    """
    x, y = _finite_flat(candidate, reference)
    if x.size == 0:
        return float("nan")
    rms_diff = math.sqrt(float(np.mean((x - y) ** 2)))
    rms_x = math.sqrt(float(np.mean(x**2)))
    rms_y = math.sqrt(float(np.mean(y**2)))
    den = rms_x + rms_y
    return 0.0 if den == 0 and rms_diff == 0 else (float("inf") if den == 0 else 200.0 * rms_diff / den)


def pearson(candidate: np.ndarray, reference: np.ndarray) -> float:
    x, y = _finite_flat(candidate, reference)
    if x.size < 2:
        return float("nan")
    sx = float(x.std())
    sy = float(y.std())
    if sx == 0 or sy == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mae(candidate: np.ndarray, reference: np.ndarray) -> float:
    x, y = _finite_flat(candidate, reference)
    return float(np.mean(np.abs(x - y))) if x.size else float("nan")


def rmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    x, y = _finite_flat(candidate, reference)
    return float(np.sqrt(np.mean((x - y) ** 2))) if x.size else float("nan")


def structural_similarity(candidate: np.ndarray, reference: np.ndarray) -> float | None:
    """Return SSIM when arrays are 2-D; None for unsupported shapes."""
    a = np.asarray(candidate)
    b = np.asarray(reference)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        return None
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    lo = min(float(np.nanmin(a)), float(np.nanmin(b)))
    hi = max(float(np.nanmax(a)), float(np.nanmax(b)))
    data_range = hi - lo
    if not np.isfinite(data_range) or data_range == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(ssim(a.astype(np.float64), b.astype(np.float64), data_range=data_range))


def _classes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.union1d(np.unique(a), np.unique(b))


def categorical_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a = np.asarray(candidate).ravel()
    b = np.asarray(reference).ravel()
    if a.shape != b.shape:
        raise ValueError("categorical arrays must have identical shape")
    labels = _classes(a, b)
    agreement = float(np.mean(a == b)) if a.size else float("nan")

    f1s: list[float] = []
    ious: list[float] = []
    for c in labels:
        tp = int(np.sum((a == c) & (b == c)))
        fp = int(np.sum((a == c) & (b != c)))
        fn = int(np.sum((a != c) & (b == c)))
        f1_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        if f1_den:
            f1s.append(2 * tp / f1_den)
        if iou_den:
            ious.append(tp / iou_den)

    # Cohen's kappa, implemented locally to keep the evaluator lightweight.
    if a.size:
        po = agreement
        pe = 0.0
        for c in labels:
            pa = float(np.mean(a == c))
            pb = float(np.mean(b == c))
            pe += pa * pb
        kappa = 1.0 if pe == 1.0 and po == 1.0 else ((po - pe) / (1.0 - pe) if pe < 1.0 else 0.0)
    else:
        kappa = float("nan")

    return {
        "voxel_agreement": agreement,
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "cohen_kappa": float(kappa),
    }


def array_summary(a: np.ndarray) -> dict[str, Any]:
    x = np.asarray(a)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.nanmin(x)) if x.size else None,
        "max": float(np.nanmax(x)) if x.size else None,
        "mean": float(np.nanmean(x)) if x.size else None,
        "finite_fraction": float(np.mean(np.isfinite(x))) if x.size else None,
    }
