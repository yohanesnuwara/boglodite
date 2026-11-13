"""NCS level-check results vs reference (reporting only) + fault probability maps."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from norne_benchmark import score as S, truth as TR
from strata.data import load_cube
from strata.config import CACHE_DIR
from strata.tools import horizon as H, foundation as FM

RUN = "20261108-201137-9b63"
c = load_cube(); ref = S.Reference(); rh = ref.horizon
i0, j0 = int(c.ilines[0]), int(c.xlines[0])
T = np.full(c.shape[:2], np.nan)
ii = (rh.il.values - i0).astype(int); jj = (rh.xl.values - j0).astype(int)
ok = (ii >= 0) & (ii < T.shape[0]) & (jj >= 0) & (jj < T.shape[1]); T[ii[ok], jj[ok]] = rh.twt.values[ok]
before = H.load(f"runs/{RUN}/store/HorizonPicks/Not2_top/v3/horizon.npz").twt
after = H.load(f"outputs/strata_dev/ncs_retracked_{RUN}_v3.npz").twt
flag = np.load(f"outputs/strata_dev/ncs_flag_{RUN}_v3.npy")
rows = np.where(np.isfinite(T).any(1))[0]; cols = np.where(np.isfinite(T).any(0))[0]
r0, r1 = max(rows.min() - 5, 0), min(rows.max() + 5, T.shape[0]); c0, c1 = max(cols.min() - 5, 0), min(cols.max() + 5, T.shape[1])
ext = [j0 + c0, j0 + c1, i0 + r1, i0 + r0]

# reference fault faces (throw > 10 m) in il/xl — truth, display only
cen, thr = TR.fault_face_table(ref.grid)
big = thr > 10
fil = c.xy_to_ilxl(cen[big, 0], cen[big, 1])

# ── 1. error maps ──
fig, ax = plt.subplots(1, 3, figsize=(19, 7.2), constrained_layout=True)
for a, (title, h) in zip(ax[:2], ((f"Before: Opus + repair (18.8 ms)", before), ("After: + NCS level check (15.7 ms)", after))):
    e = (h - T)[r0:r1, c0:c1]
    im = a.imshow(e, cmap="RdBu_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
    a.set_title(f"{title}\nagent − truth; within ±15 ms: {np.mean(np.abs(e[np.isfinite(e)]) <= 15):.0%}")
    a.set_xlabel("Crossline"); a.set_ylabel("Inline")
ax[1].contour(np.arange(j0 + c0, j0 + c1), np.arange(i0 + r0, i0 + r1), flag[r0:r1, c0:c1].astype(float), [0.5], colors="k", linewidths=.8)
fig.colorbar(im, ax=ax[:2], shrink=.8, label="agent − truth (ms)")
d = (np.abs(after - T) - np.abs(before - T))[r0:r1, c0:c1]
im2 = ax[2].imshow(d, cmap="PiYG_r", vmin=-60, vmax=60, extent=ext, aspect="auto", interpolation="nearest")
ax[2].contour(np.arange(j0 + c0, j0 + c1), np.arange(i0 + r0, i0 + r1), flag[r0:r1, c0:c1].astype(float), [0.5], colors="k", linewidths=.8)
dd = d[np.isfinite(d)]
ax[2].set_title(f"Change in |error| (green = better; black = NCS-flagged areas)\nbetter {np.mean(dd < -4):.0%} · worse {np.mean(dd > 4):.0%} of reservoir area")
ax[2].set_xlabel("Crossline"); ax[2].set_ylabel("Inline")
fig.colorbar(im2, ax=ax[2], shrink=.8, label="|after| − |before| (ms)")
plt.savefig("outputs/strata_dev/ncs_error_maps.png", dpi=100)

# ── 2. sections ──
t = c.t0 + c.dt * np.arange(c.shape[2])
fig, axs = plt.subplots(3, 1, figsize=(15, 15))
for a, IL in zip(axs, (1080, 1120, 1160)):
    i = IL - i0; r = rh[rh.il == IL].sort_values("xl")
    lo, hi = np.nanmin(r.twt) - 120, np.nanmax(r.twt) + 80; kk = (t >= lo) & (t <= hi); v = 2.5 * c.rms
    a.imshow(np.asarray(c.data[i]).T[kk], aspect="auto", cmap="gray_r", vmin=-v, vmax=v, extent=[c.xlines[0] - .5, c.xlines[-1] + .5, t[kk][-1], t[kk][0]])
    a.plot(r.xl, r.twt, color="#ffd400", lw=3, label="True Not 2 top")
    m = dict(zip(r.xl.astype(int), r.twt))
    for h, col, lab in ((before, "#00b7ff", "Before (Opus + repair)"), (after, "#ff2d2d", "After NCS level check")):
        dd_ = np.array([h[i, j] - m.get(int(x), np.nan) for j, x in enumerate(c.xlines)])
        a.plot(c.xlines, h[i], color=col, lw=1.5, label=f"{lab} · inline MAE {np.nanmean(np.abs(dd_)):.1f} ms")
    a.set_xlim(r.xl.min() - 40, r.xl.max() + 40); a.set_ylim(hi, lo); a.set_ylabel("TWT (ms)")
    a.set_title(f"Inline {IL}"); a.legend(loc="lower left", fontsize=9, framealpha=.9)
axs[-1].set_xlabel("Crossline"); plt.tight_layout()
plt.savefig("outputs/strata_dev/ncs_sections_truth.png", dpi=100)

# ── 3. fault probability maps at the Not 2 level ──
fm = np.load(f"runs/{RUN}/store/FaultMap/reservoir/v1/faults.npz")
disc = fm["discontinuity"].astype(np.float32)
p_disc = (np.argsort(np.argsort(disc.ravel())) / disc.size).reshape(disc.shape)      # percentile rank 0–1
feats, co = FM.extract_features(c)
tn = FM.node_times(c, co)
gi_, gj_ = co["il_sample"], co["xl_sample"]
hz_g = ndimage.median_filter(np.where(np.isfinite(after), after, np.nanmedian(after)), 5)[np.ix_(gi_, gj_)]
kt = np.clip(np.searchsorted(tn, hz_g), 0, len(tn) - 1)
Fh = np.take_along_axis(np.asarray(feats, np.float32), kt[..., None, None], axis=2)[:, :, 0]   # (40,125,768)
dg = ndimage.maximum_filter(disc, size=5)[np.ix_(gi_, gj_)]                           # discontinuity near each node
pos, neg = dg >= np.percentile(disc, 97), dg <= np.percentile(disc, 50)
X = np.r_[Fh[pos], Fh[neg]]; y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
probe = FM.fit_probe(X, y, C=0.05)
p_ncs = probe.predict_proba(Fh.reshape(-1, 768))[:, 1].reshape(Fh.shape[:2])
fig, ax = plt.subplots(1, 2, figsize=(18, 7.5), constrained_layout=True)
e_full = [c.xlines[0], c.xlines[-1], c.ilines[-1], c.ilines[0]]
ax[0].imshow(p_disc, cmap="magma", vmin=0, vmax=1, extent=e_full, aspect="auto", interpolation="nearest")
ax[0].set_title("Fault probability (a): discontinuity attribute, 2300–2700 ms\n(percentile rank; what the fault agent uses now)")
im = ax[1].imshow(p_ncs, cmap="magma", vmin=0, vmax=1, extent=[c.xlines[gj_[0]], c.xlines[gj_[-1]], c.ilines[gi_[-1]], c.ilines[gi_[0]]], aspect="auto", interpolation="bilinear")
ax[1].set_title(f"Fault probability (b): NCS-3D features at the Not 2 level\nprobe trained on discontinuity ≥97th pct ({int(pos.sum())}) vs ≤50th pct ({int(neg.sum())}) nodes")
for a in ax:
    a.scatter(fil[:, 1], fil[:, 0], s=1.5, c="#00e5ff", alpha=.5, label="reference model faults, throw > 10 m (truth, display only)")
    a.contour(np.arange(j0 + c0, j0 + c1), np.arange(i0 + r0, i0 + r1), np.isfinite(T[r0:r1, c0:c1]).astype(float), [0.5], colors="w", linewidths=.7)
    a.set_xlabel("Crossline"); a.set_ylabel("Inline"); a.set_xlim(1400, 2200); a.set_ylim(1290, 970)
ax[0].legend(loc="lower left", fontsize=8, markerscale=6)
fig.colorbar(im, ax=ax, shrink=.8, label="P(fault)")
plt.savefig("outputs/strata_dev/fault_probability_maps.png", dpi=100)
np.savez_compressed(CACHE_DIR / "ncs3d_fault_prob_not2.npz", p_ncs=p_ncs, il_sample=gi_, xl_sample=gj_)
print("done")
