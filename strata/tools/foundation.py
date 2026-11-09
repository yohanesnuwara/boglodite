"""Seismic foundation-model features (NCS-v1, Norwegian Computing Center, Apache-2.0) and a
well-trained "below the top" probability → a coarse, package-based horizon guide.

The NCS 3D ViT-MAE reads 224³ crops in (time, inline, crossline) order, normalised by the cube
standard deviation and clipped at ±3σ, and returns one 768-d embedding per 16³ patch. Patches
summarise 64 ms here, so the model cannot pick a 4-ms horizon; it can tell which stratigraphic
package a location belongs to. We densify the output grid by running shifted passes.

Input data only: the cube and the tied well tops (non-blind wells). No reference horizon.
"""

from __future__ import annotations

import os
import time

import numpy as np

from strata.config import CACHE_DIR
from strata.data import Cube

os.environ.setdefault("USE_TF", "0")                 # transformers must not import TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HOME", str(CACHE_DIR.parent / "hf_cache"))

MODEL_ID = "NorskRegnesentralSTI/NCS-v1-3d-base"
CROP, PATCH = 224, 16
P = CROP // PATCH                                     # 14 patches per axis


def _patch_weights() -> np.ndarray:
    """Triangular weight per patch position in a crop (centre patches count more)."""
    w = np.minimum(np.arange(P) + 1, P - np.arange(P)).astype(np.float32)
    return w / w.max()


def extract_features(cube: Cube, densify_t: int = 4, densify_xy: int = 2, stride_patches: int = 7,
                     batch: int = 4, out_name: str | None = None, log=print) -> tuple[np.ndarray, dict]:
    """Embeddings on a grid with spacing PATCH/densify (time) and PATCH/densify_xy (lateral).
    Returns (features float16 memmap (n_il_f, n_xl_f, n_t_f, 768), coords dict with the sample index
    of every grid node along each axis). Cached in data/strata_cache."""
    import torch
    from NCS.inference.pipeline import _get_volume_index_tensors
    from NCS.models.vit3d import ViT3DModel

    name = out_name or f"ncs3d_t{densify_t}_xy{densify_xy}"
    fpath, cpath = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}_coords.npz"
    if fpath.exists() and cpath.exists():
        z = np.load(cpath)
        return np.load(fpath, mmap_mode="r"), {k: z[k] for k in z.files}

    t0 = time.time()
    model = ViT3DModel.from_pretrained(MODEL_ID).to("cuda", torch.float16).eval()
    vi, vs = _get_volume_index_tensors("cuda", CROP, PATCH, P, batch)

    data = np.asarray(cube.data, np.float32)
    std = float(data.std())
    ni, nj, nt = data.shape
    stride = stride_patches * PATCH
    pad = [(0, CROP + stride) for _ in range(3)]
    D = np.pad(np.clip(data / (std + 1e-10), -3, 3), pad, mode="reflect").astype(np.float16)

    st, sxy = PATCH // densify_t, PATCH // densify_xy
    # grid node k along an axis sits at sample PATCH//2 + k*step
    gi, gj, gt = (np.arange(PATCH // 2, n, s) for n, s in ((ni, sxy), (nj, sxy), (nt, st)))
    acc = np.zeros((len(gi), len(gj), len(gt), 768), np.float32)
    wsum = np.zeros((len(gi), len(gj), len(gt)), np.float32)
    wp = _patch_weights()
    w3 = np.einsum("t,i,j->ijt", wp, wp, wp)           # (il, xl, t) weights within a crop

    jobs = []
    for oi in range(0, PATCH, sxy):
        for oj in range(0, PATCH, sxy):
            for ot in range(0, PATCH, st):
                for si in range(oi, max(ni - PATCH // 2, 1), stride):
                    for sj in range(oj, max(nj - PATCH // 2, 1), stride):
                        for sk in range(ot, max(nt - PATCH // 2, 1), stride):
                            jobs.append((si, sj, sk))
    log(f"NCS features: {len(jobs)} crops, grid {acc.shape[:3]}, std {std:.4g}")

    def flush(buf, meta):
        x = torch.from_numpy(np.stack(buf)).to("cuda")[:, None]          # (B,1,t,il,xl)
        n = x.shape[0]
        v, s = (vi, vs) if n == batch else _get_volume_index_tensors("cuda", CROP, PATCH, P, n)
        with torch.inference_mode():
            out = model(pixel_values=x, volume_indices=v, volume_size=s).last_hidden_state[:, 1:]
        f = out.float().cpu().numpy().reshape(n, P, P, P, 768).transpose(0, 2, 3, 1, 4)  # → (B, il, xl, t, H)
        for b, (si, sj, sk) in enumerate(meta):
            ci, cj, ck = si + PATCH // 2 + PATCH * np.arange(P), sj + PATCH // 2 + PATCH * np.arange(P), \
                sk + PATCH // 2 + PATCH * np.arange(P)
            ii = (ci - PATCH // 2) // sxy
            jj = (cj - PATCH // 2) // sxy
            tt = (ck - PATCH // 2) // st
            mi, mj, mt = ii < len(gi), jj < len(gj), tt < len(gt)
            if not (mi.any() and mj.any() and mt.any()):
                continue
            sub = f[b][np.ix_(mi, mj, mt)]
            w = w3[np.ix_(mi, mj, mt)]
            I, J, T = np.ix_(ii[mi], jj[mj], tt[mt])
            acc[I, J, T] += sub * w[..., None]
            wsum[I, J, T] += w

    buf, meta = [], []
    for n, (si, sj, sk) in enumerate(jobs):
        crop = D[si:si + CROP, sj:sj + CROP, sk:sk + CROP]           # (il, xl, t)
        buf.append(np.ascontiguousarray(np.moveaxis(crop, -1, 0)))  # → (t, il, xl)
        meta.append((si, sj, sk))
        if len(buf) == batch:
            flush(buf, meta); buf, meta = [], []
        if n % 200 == 0:
            log(f"  {n}/{len(jobs)} crops, {time.time() - t0:.0f} s")
    if buf:
        flush(buf, meta)
    feats = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.float16)
    np.save(fpath, feats)
    coords = {"il_sample": gi, "xl_sample": gj, "t_sample": gt, "std": np.array(std),
              "seconds": np.array(time.time() - t0)}
    np.savez(cpath, **coords)
    log(f"NCS features done in {time.time() - t0:.0f} s → {fpath}")
    return np.load(fpath, mmap_mode="r"), coords


# ── well-trained "below the top" probability and the package-boundary surface ──
def _node(coords: dict, cube: Cube, il: float, xl: float) -> tuple[int, int]:
    i = int(round(il - cube.ilines[0])); j = int(round(xl - cube.xlines[0]))
    gi = int(np.clip(np.round((i - coords["il_sample"][0]) / (coords["il_sample"][1] - coords["il_sample"][0])), 0, len(coords["il_sample"]) - 1))
    gj = int(np.clip(np.round((j - coords["xl_sample"][0]) / (coords["xl_sample"][1] - coords["xl_sample"][0])), 0, len(coords["xl_sample"]) - 1))
    return gi, gj


def node_times(cube: Cube, coords: dict) -> np.ndarray:
    return cube.t0 + cube.dt * coords["t_sample"].astype(float)


def training_set(feats, coords, cube: Cube, wells: list[tuple[str, float, float, float]],
                 gap_ms: float = 16.0, span_ms: float = 120.0):
    """X, y, group: feature vectors at each well's grid column, labelled 0 above / 1 below the tied
    top (|Δt| in [gap, span]); group = well index (for leave-one-well-out)."""
    tn = node_times(cube, coords)
    X, y, g = [], [], []
    for k, (_, il, xl, twt) in enumerate(wells):
        gi, gj = _node(coords, cube, il, xl)
        d = tn - twt
        sel = (np.abs(d) >= gap_ms) & (np.abs(d) <= span_ms)
        X.append(np.asarray(feats[gi, gj][sel], np.float32)); y.append((d[sel] > 0).astype(int)); g += [k] * int(sel.sum())
    return np.concatenate(X), np.concatenate(y), np.array(g)


class LinearProbe:
    """Logistic regression on z-scored features (L2, full-batch L-BFGS, PyTorch) — the paper's linear probe."""

    def __init__(self, X: np.ndarray, y: np.ndarray, C: float = 0.05, iters: int = 200):
        import torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.w = torch.zeros(X.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=iters, line_search_fn="strong_wolfe")
        lam = 1.0 / (C * len(y))

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ self.w + self.b, yt) + 0.5 * lam * (self.w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w_np, self.b_np = self.w.detach().numpy(), float(self.b.detach())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mu) / self.sd) @ self.w_np + self.b_np
        p = 1 / (1 + np.exp(-z))
        return np.stack([1 - p, p], 1)

    def score(self, X, y) -> float:
        return float(((self.predict_proba(X)[:, 1] > 0.5) == y).mean())


def fit_probe(X, y, C: float = 0.05) -> LinearProbe:
    return LinearProbe(X, y, C)


def step_boundary(p: np.ndarray, tn: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Best single above→below step in P within [lo, hi]: argmin Σ_{t<b} P + Σ_{t≥b} (1−P).
    Returns (boundary time = midpoint between nodes, contrast = mean P below − mean P above)."""
    m = (tn >= lo) & (tn <= hi)
    if m.sum() < 3:
        return np.nan, 0.0
    q, t = p[m], tn[m]
    c_above = np.concatenate([[0], np.cumsum(q)])              # cost of labelling first k as above
    c_below = np.concatenate([np.cumsum((1 - q)[::-1])[::-1], [0]])
    cost = c_above + c_below
    k = int(np.argmin(cost[1:-1])) + 1
    contrast = float(q[k:].mean() - q[:k].mean())
    return float(0.5 * (t[k - 1] + t[k])), contrast


def leave_one_well_out(feats, coords, cube: Cube, wells, window_ms: float = 100.0, C: float = 0.05) -> list[dict]:
    """For each well: train on the others, predict its column, boundary within the other wells'
    interpolated surface ±window_ms. Compared with that interpolation alone. Input data only."""
    from strata.tools.horizon import guide_surface
    X, y, g = training_set(feats, coords, cube, wells)
    tn = node_times(cube, coords)
    out = []
    for k, (name, il, xl, twt) in enumerate(wells):
        clf = fit_probe(X[g != k], y[g != k], C)
        others = [(a, b, t) for j, (_, a, b, t) in enumerate(wells) if j != k]
        gs = guide_surface(cube, others)
        i, j = int(round(il - cube.ilines[0])), int(round(xl - cube.xlines[0]))
        interp = float(gs[i, j])
        gi, gj = _node(coords, cube, il, xl)
        p = clf.predict_proba(np.asarray(feats[gi, gj], np.float32))[:, 1]
        b, con = step_boundary(p, tn, interp - window_ms, interp + window_ms)
        out.append({"well": name, "tie_twt": round(twt, 1), "interp_err_ms": round(interp - twt, 1),
                    "fm_err_ms": round(b - twt, 1) if np.isfinite(b) else None, "contrast": round(con, 2)})
    return out


def probability_and_surface(feats, coords, cube: Cube, wells, guide: np.ndarray, window_ms: float = 100.0,
                            C: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train on all wells; P(below) on the feature grid; per grid column the step boundary within the
    well guide ±window_ms; upsampled to the full (il, xl) grid. Returns (P grid, boundary grid,
    boundary full-res, info)."""
    from scipy.interpolate import RegularGridInterpolator
    X, y, _ = training_set(feats, coords, cube, wells)
    clf = fit_probe(X, y, C)
    tn = node_times(cube, coords)
    ni, nj, nt, H = feats.shape
    P = np.zeros((ni, nj, nt), np.float32)
    for a in range(ni):
        P[a] = clf.predict_proba(np.asarray(feats[a], np.float32).reshape(-1, H))[:, 1].reshape(nj, nt)
    B = np.full((ni, nj), np.nan, np.float32); CON = np.zeros((ni, nj), np.float32)
    for a in range(ni):
        for b in range(nj):
            g = guide[coords["il_sample"][a], coords["xl_sample"][b]]
            B[a, b], CON[a, b] = step_boundary(P[a, b], tn, g - window_ms, g + window_ms)
    ii, jj = np.arange(cube.shape[0]), np.arange(cube.shape[1])
    f = RegularGridInterpolator((coords["il_sample"], coords["xl_sample"]), np.nan_to_num(B, nan=np.nanmedian(B)),
                                bounds_error=False, fill_value=None)
    full = f(np.stack(np.meshgrid(ii, jj, indexing="ij"), -1)).astype(np.float32)
    info = {"train_samples": int(len(y)), "train_accuracy": round(float(clf.score(X, y)), 3),
            "median_contrast": round(float(np.median(CON)), 3)}
    return P, B, full, {**info, "contrast_grid": CON}


def level_check_retrack(cube: Cube, hz, fm_surface: np.ndarray, contrast_grid: np.ndarray, coords: dict,
                        wells: list[tuple[float, float, float]], fault: np.ndarray | None = None,
                        polarity: str = "trough", disagree_ms: float = 40.0, min_contrast: float = 0.5,
                        window_ms: float = 25.0, min_area: int = 200):
    """Re-track only where the horizon and the NCS package boundary disagree by > disagree_ms and the
    boundary is confident (contrast ≥ min_contrast): Viterbi with the boundary as guide (±window_ms),
    wells as anchors; elsewhere the horizon is kept. Thresholds fixed a priori; input data only."""
    from scipy import ndimage
    from strata.tools.horizon import Horizon, viterbi_track
    ni, nj = cube.shape[:2]
    ia = np.clip(np.searchsorted(coords["il_sample"], np.arange(ni)), 0, len(coords["il_sample"]) - 1)
    ja = np.clip(np.searchsorted(coords["xl_sample"], np.arange(nj)), 0, len(coords["xl_sample"]) - 1)
    con = contrast_grid[np.ix_(ia, ja)]
    d = hz.twt - fm_surface
    flag = np.isfinite(d) & (np.abs(d) > disagree_ms) & (con >= min_contrast)
    flag = ndimage.binary_opening(flag, iterations=2)
    lab, n = ndimage.label(flag)
    sizes = ndimage.sum(flag, lab, range(1, n + 1))
    flag = np.isin(lab, np.where(sizes >= min_area)[0] + 1)
    guide = ndimage.median_filter(fm_surface, size=9)
    new, rep = viterbi_track(cube, guide, polarity, window_ms=window_ms, fault=fault, anchors=wells,
                             fault_mode="skip" if fault is not None else "relax")
    twt = np.where(flag & np.isfinite(new.twt), new.twt, hz.twt).astype(np.float32)
    conf = np.where(flag, np.minimum(new.conf, 0.7), hz.conf).astype(np.float32)
    info = {"flagged_traces": int(flag.sum()), "flagged_frac_of_picked": round(float(flag.sum() / max(np.isfinite(hz.twt).sum(), 1)), 4),
            "regions": int(len(np.unique(lab[flag]))), "mean_shift_in_flagged_ms": round(float(np.nanmean((twt - hz.twt)[flag])), 1) if flag.any() else None,
            "params": {"disagree_ms": disagree_ms, "min_contrast": min_contrast, "window_ms": window_ms, "min_area": min_area}}
    return Horizon(twt, conf), info, flag
