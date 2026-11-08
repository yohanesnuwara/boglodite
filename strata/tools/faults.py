"""Fault tools: a trace-to-trace discontinuity attribute, a fault (barrier) map, and the
horizon–fault consistency check used by the fault agent as verifier.

Discontinuity at a trace = 1 − (smallest, over its 4 neighbours, of the best normalised
cross-correlation within ±max_lag samples), computed in time slabs and combined by max.
High values mark faults (and noise); the map is thresholded and cleaned into a barrier.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from strata.data import Cube

TOOL_VERSION = "faults-0.1"


def _slab_similarity(D: np.ndarray, max_lag: int) -> np.ndarray:
    """Min over the 4 neighbours of the best lagged NCC, for a (ni, nj, nt) slab."""
    x = D - D.mean(-1, keepdims=True)
    nrm = np.sqrt((x * x).sum(-1)) + 1e-9
    ni, nj, nt = x.shape
    out = np.ones((ni, nj), np.float32)
    for axis in (0, 1):
        a = x[:-1] if axis == 0 else x[:, :-1]
        b = x[1:] if axis == 0 else x[:, 1:]
        na = nrm[:-1] if axis == 0 else nrm[:, :-1]
        nb = nrm[1:] if axis == 0 else nrm[:, 1:]
        best = np.full(a.shape[:2], -1.0, np.float32)
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                c = (a[..., lag:] * b[..., :nt - lag]).sum(-1)
            else:
                c = (a[..., :nt + lag] * b[..., -lag:]).sum(-1)
            best = np.maximum(best, (c / (na * nb)).astype(np.float32))
        if axis == 0:
            out[:-1] = np.minimum(out[:-1], best)
            out[1:] = np.minimum(out[1:], best)
        else:
            out[:, :-1] = np.minimum(out[:, :-1], best)
            out[:, 1:] = np.minimum(out[:, 1:], best)
    return out


def discontinuity(cube: Cube, t_from: float = 2300.0, t_to: float = 2700.0, slab_ms: float = 100.0,
                  max_lag_ms: float = 8.0) -> np.ndarray:
    """(n_il, n_xl) discontinuity in [0, 2]: max over slabs of 1 − neighbour similarity."""
    k0, k1 = int((t_from - cube.t0) / cube.dt), int((t_to - cube.t0) / cube.dt)
    step = max(4, int(slab_ms / cube.dt))
    lag = max(1, int(round(max_lag_ms / cube.dt)))
    disc = np.zeros(cube.shape[:2], np.float32)
    for k in range(max(0, k0), min(cube.shape[2], k1) - step + 1, step):
        sim = _slab_similarity(np.asarray(cube.data[:, :, k:k + step], np.float32), lag)
        disc = np.maximum(disc, 1.0 - sim)
    return disc


def fault_mask(disc: np.ndarray, percentile: float = 92.0, min_size: int = 30, dilate: int = 1) -> tuple[np.ndarray, dict]:
    """Threshold the discontinuity at a percentile, drop small blobs, optionally dilate."""
    thr = float(np.percentile(disc, percentile))
    m = disc >= thr
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, np.where(sizes >= min_size)[0] + 1)
    if dilate:
        keep = ndimage.binary_dilation(keep, iterations=dilate)
    blocks, nb = ndimage.label(~keep)
    bs = np.bincount(blocks.ravel())[1:]
    return keep, {"threshold": round(thr, 4), "fault_fraction": round(float(keep.mean()), 4),
                  "fault_segments": int(len(np.unique(lab[keep & (lab > 0)]))), "blocks": int(nb),
                  "blocks_over_1000_traces": int((bs > 1000).sum())}


def horizon_vs_faults(twt: np.ndarray, mask: np.ndarray, jump_ms: float = 12.0, offset_ms: float = 4.0) -> dict:
    """Horizon–fault consistency (fault agent as verifier).
    - jumps (neighbour |Δt| > jump_ms) away from faults = cycle skips / leaks;
    - neighbour pairs straddling a fault with |Δt| < offset_ms = horizon crossing a fault
      without the throw the fault should show (tracked straight across)."""
    near = ndimage.binary_dilation(mask, iterations=2)
    res = {}
    on, off, across_flat, across = 0, 0, 0, 0
    for axis in (0, 1):
        d = np.abs(np.diff(twt, axis=axis))
        f = (near[:-1] | near[1:]) if axis == 0 else (near[:, :-1] | near[:, 1:])
        valid = np.isfinite(d)
        jumps = valid & (d > jump_ms)
        on += int((jumps & f).sum())
        off += int((jumps & ~f).sum())
        m = (mask[:-1] ^ mask[1:]) if axis == 0 else (mask[:, :-1] ^ mask[:, 1:])
        across += int((valid & m).sum())
        across_flat += int((valid & m & (d < offset_ms)).sum())
    res = {"jumps_on_faults": on, "jumps_off_faults": off,
           "fault_crossings": across, "crossings_without_offset": across_flat,
           "crossing_without_offset_frac": round(across_flat / across, 3) if across else None,
           "off_fault_jump_frac": round(off / max(on + off, 1), 3)}
    return res
