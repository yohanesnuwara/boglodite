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


# ── throw-aware horizon–fault check (calibrated from the seismic itself) ─────
def _segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, np.where(sizes >= min_len)[0] + 1)


def fault_crossings(mask: np.ndarray, max_width: int = 8):
    """Pairs of traces (a, b) on either side of a fault zone, scanning inlines and crosslines:
    a = last trace before a run of fault pixels, b = first trace after it (run ≤ max_width)."""
    pairs = []
    ni, nj = mask.shape
    for axis in (0, 1):
        lines = range(nj) if axis == 0 else range(ni)
        for L in lines:
            m = mask[:, L] if axis == 0 else mask[L, :]
            n = len(m)
            k = 0
            while k < n:
                if m[k]:
                    s = k
                    while k < n and m[k]:
                        k += 1
                    if s > 0 and k < n and (k - s) <= max_width:
                        a = (s - 1, L) if axis == 0 else (L, s - 1)
                        b = (k, L) if axis == 0 else (L, k)
                        pairs.append((a, b))
                else:
                    k += 1
    return pairs


def _throw_estimate(D, t0, dt, a_ij, b_ij, t, win_ms, L):
    hw = int(win_ms / 2 / dt)
    k = int(round((t - t0) / dt))
    if k - hw - L < 0 or k + hw + L >= D.shape[2]:
        return None
    a = np.asarray(D[a_ij][k - hw:k + hw + 1], float); a -= a.mean()
    best = (-1.0, 0.0)
    for lag in range(-L, L + 1):
        b = np.asarray(D[b_ij][k - hw + lag:k + hw + 1 + lag], float); b = b - b.mean()
        c = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
        if c > best[0]:
            best = (c, lag * dt)
    return best


def throw_consistency(cube: Cube, twt: np.ndarray, mask: np.ndarray, min_segment: int = 40,
                      window_ms: float = 80.0, max_throw_ms: float = 80.0, min_ncc: float = 0.6,
                      tol_ms: float = 8.0, step: int = 2) -> dict:
    """Throw consistency at crossings of long fault segments, using only seismic + horizon.
    The seismic offset across the fault is estimated by lagged correlation of the two side traces.
    Because reflections are periodic, single estimates can lock onto the wrong cycle, so an estimate
    is kept only if it is SELF-CONSISTENT: forward (a→b) and reverse (b→a) estimates agree within
    tol_ms, and a 1.5× longer window gives the same lag. The horizon's offset is compared with these
    reliable estimates only."""
    strong = _segments(mask, min_segment)
    pairs = fault_crossings(strong)[::step]
    D = cube.data
    L = int(max_throw_ms / cube.dt)
    n_eval = n_rel = mism = with_throw = with_throw_mism = 0
    for a, b in pairs:
        ta, tb = twt[a], twt[b]
        if not (np.isfinite(ta) and np.isfinite(tb)):
            continue
        n_eval += 1
        f = _throw_estimate(D, cube.t0, cube.dt, a, b, ta, window_ms, L)
        if not f or f[0] < min_ncc:
            continue
        r = _throw_estimate(D, cube.t0, cube.dt, b, a, tb, window_ms, L)
        g = _throw_estimate(D, cube.t0, cube.dt, a, b, ta, window_ms * 1.5, L)
        if not (r and g) or abs(f[1] + r[1]) > tol_ms or abs(f[1] - g[1]) > tol_ms:
            continue
        n_rel += 1
        bad = abs((tb - ta) - f[1]) > tol_ms
        mism += bad
        if abs(f[1]) > tol_ms:
            with_throw += 1
            with_throw_mism += bad
    return {"strong_fault_crossings_evaluated": n_eval, "reliable_throw_estimates": n_rel,
            "mismatched": int(mism), "mismatch_frac": round(mism / n_rel, 3) if n_rel else None,
            "reliable_crossings_with_throw": with_throw,
            "mismatch_frac_where_throw": round(with_throw_mism / with_throw, 3) if with_throw else None,
            "tolerance_ms": tol_ms, "min_segment_traces": min_segment,
            "note": "only self-consistent throw estimates (forward/reverse/longer window agree) are used"}
