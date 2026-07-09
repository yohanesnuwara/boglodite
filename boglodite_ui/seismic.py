"""SEGY slicing and PNG rendering for the Boglodite UI.

The volume is opened once with segyio (strict=False, memory-mapped when
possible) and slices are read on demand — no full-cube load, so the 1.2 GB
F3 volume opens instantly.

Rendering follows the same conventions as the sandbox scripts:
  * inline slice  -> (time, xline) image
  * xline slice   -> (time, inline) image
  * time slice    -> (inline, xline) image
  * amplitudes clipped at symmetric percentiles, seismic grey colormap
Result .npy files from outputs/ are rendered with class/probability-aware
colormaps (discrete tab colormap for integer class maps, inferno for
probabilities).
"""

from __future__ import annotations

import io
import os
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import segyio  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

_DARK = {
    "figure.facecolor": "#12181F",
    "axes.facecolor": "#0B0F13",
    "axes.edgecolor": "#3A4756",
    "axes.labelcolor": "#D8E0E8",
    "xtick.color": "#8A96A0",
    "ytick.color": "#8A96A0",
    "text.color": "#D8E0E8",
    "font.family": "monospace",
    "font.size": 9,
}


class SegyVolume:
    """Lazily-opened SEGY volume with per-slice reads."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._f = segyio.open(path, "r", strict=False)
        if self._f.unstructured or self._f.ilines is None:
            self._f.close()
            raise ValueError(
                f"{os.path.basename(path)} has no inline/xline structure "
                "(unstructured SEGY) — the viewer needs a regular 3D volume."
            )
        try:
            self._f.mmap()
        except Exception:
            pass
        self.ilines = np.asarray(self._f.ilines)
        self.xlines = np.asarray(self._f.xlines)
        self.samples = np.asarray(self._f.samples)

    # ── metadata ────────────────────────────────────────────────────────────
    def meta(self) -> dict:
        def rng(a):
            step = int(a[1] - a[0]) if len(a) > 1 else 1
            return {"min": int(a[0]), "max": int(a[-1]), "step": step, "count": len(a)}

        return {
            "path": self.path,
            "file": os.path.basename(self.path),
            "inline": rng(self.ilines),
            "xline": rng(self.xlines),
            "time": rng(self.samples),
        }

    # ── slicing ─────────────────────────────────────────────────────────────
    def slice(self, axis: str, value: int) -> tuple[np.ndarray, dict]:
        """Return (2D array, plot info) for an inline/xline/time slice.

        Arrays are oriented for display: time increases downward for vertical
        sections; map view (inline vs xline) for time slices.
        """
        with self._lock:
            if axis == "inline":
                value = int(self.ilines[np.abs(self.ilines - value).argmin()])
                data = np.asarray(self._f.iline[value]).T  # (samples, xlines)
                info = {
                    "axis": "inline",
                    "value": value,
                    "xlabel": "Crossline",
                    "ylabel": "Time (ms)",
                    "extent": [
                        float(self.xlines[0]),
                        float(self.xlines[-1]),
                        float(self.samples[-1]),
                        float(self.samples[0]),
                    ],
                }
            elif axis == "xline":
                value = int(self.xlines[np.abs(self.xlines - value).argmin()])
                data = np.asarray(self._f.xline[value]).T  # (samples, inlines)
                info = {
                    "axis": "xline",
                    "value": value,
                    "xlabel": "Inline",
                    "ylabel": "Time (ms)",
                    "extent": [
                        float(self.ilines[0]),
                        float(self.ilines[-1]),
                        float(self.samples[-1]),
                        float(self.samples[0]),
                    ],
                }
            elif axis == "time":
                idx = int(np.abs(self.samples - value).argmin())
                value = int(self.samples[idx])
                data = np.asarray(self._f.depth_slice[idx])  # (inlines, xlines)
                info = {
                    "axis": "time",
                    "value": value,
                    "xlabel": "Crossline",
                    "ylabel": "Inline",
                    "extent": [
                        float(self.xlines[0]),
                        float(self.xlines[-1]),
                        float(self.ilines[-1]),
                        float(self.ilines[0]),
                    ],
                }
            else:
                raise ValueError(f"Unknown axis: {axis}")
        return data.astype(np.float32), info

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ── rendering ────────────────────────────────────────────────────────────────
def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def render_seismic_png(data: np.ndarray, info: dict) -> bytes:
    """Render an amplitude slice with symmetric percentile clipping."""
    vmax = float(np.percentile(np.abs(data), 98)) or 1.0
    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        ax.imshow(data, cmap="gray", vmin=-vmax, vmax=vmax,
                  extent=info["extent"], aspect="auto", interpolation="bilinear")
        ax.set_xlabel(info["xlabel"])
        ax.set_ylabel(info["ylabel"])
        ax.set_title(f"{info['axis'].upper()} {info['value']}", loc="left")
    return _fig_to_png(fig)


def render_npy_png(path: str) -> bytes:
    """Best-effort render of an agent-produced .npy result.

    Handles the shapes the sandbox scripts write:
      * (1, X, Z) int class maps          -> discrete facies colormap
      * (1, X, Z, C) softmax probability  -> argmax class map
      * (X, Z) float in [0, 1]            -> fault-probability inferno map
      * anything else 2D                  -> viridis
    """
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    title = os.path.basename(path)

    if arr.ndim == 3 and arr.shape[-1] <= 16:  # probabilities per class
        arr = np.argmax(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Cannot render array of shape {arr.shape}")

    # Vertical sections are stored (xline, time); show time downward.
    if arr.shape[0] > arr.shape[1]:
        disp = arr.T
    else:
        disp = arr

    is_class = np.issubdtype(arr.dtype, np.integer) or (
        np.unique(arr[:: max(1, arr.size // 5000)]).size <= 24
        and np.allclose(arr, np.round(arr))
    )

    with plt.rc_context(_DARK):
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        if is_class:
            classes = np.unique(disp).astype(int)
            n = max(int(classes.max()) + 1, 2)
            cmap = plt.get_cmap("tab10" if n <= 10 else "tab20", n)
            norm = BoundaryNorm(np.arange(-0.5, n + 0.5), n)
            im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="auto",
                           interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n), shrink=0.85)
            cbar.set_label("Class")
        else:
            lo, hi = float(np.nanmin(disp)), float(np.nanmax(disp))
            if 0.0 <= lo and hi <= 1.0 + 1e-6:
                im = ax.imshow(disp, cmap="inferno", vmin=0, vmax=1,
                               aspect="auto", interpolation="bilinear")
                fig.colorbar(im, ax=ax, shrink=0.85).set_label("Probability")
            else:
                im = ax.imshow(disp, cmap="viridis", aspect="auto",
                               interpolation="bilinear")
                fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    return _fig_to_png(fig)
